# src/api/app.py

import os
import time
import json
import csv
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, RedirectResponse

from prometheus_client import Counter, Histogram, generate_latest

from api.schemas import PredictionRequest, PredictionResponse
from api.predictor import predict
from core.load_model import load_model
from src.training.registry import (
    check_registry_health,
    get_model_uri
)

# =====================================================
# CONFIG
# =====================================================

APP_NAME = "credit-risk-api"
APP_VERSION = "4.0.0"

MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILE = PRED_DIR / "predictions.csv"

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# METRICS
# =====================================================

REQUEST_COUNT = Counter(
    "api_requests_total", "Total requests", ["method", "path", "status"]
)

PRED_COUNT = Counter("prediction_requests_total", "Total predictions")
ERROR_COUNT = Counter("prediction_errors_total", "Prediction failures")

LATENCY = Histogram("api_latency_seconds", "Latency", ["path"])

# =====================================================
# GLOBAL STATE
# =====================================================

model = None
shadow_model = None
feature_schema = None


# =====================================================
# FEATURE SCHEMA
# =====================================================

def load_feature_schema():
    schema_path = BASE_DIR / "models" / "feature_schema.json"

    if not schema_path.exists():
        raise RuntimeError("Feature schema missing - model contract broken")

    with open(schema_path, "r") as f:
        return json.load(f)["features"]


# =====================================================
# EVENT STORE
# =====================================================

def log_event(record: Dict[str, Any]):
    file_exists = PRED_FILE.exists()

    with open(PRED_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "probability",
                "prediction",
                "model",
                "alias",
                "latency_ms"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


# =====================================================
# LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, shadow_model, feature_schema

    logger.info("Starting API service...")

    if not check_registry_health():
        raise RuntimeError("Model registry unavailable")

    try:
        # ---------------------------
        # PRIMARY (CHAMPION)
        # ---------------------------
        model = load_model(
            model_name=MODEL_NAME,
            uri=get_model_uri(alias=MODEL_ALIAS)
        )

        # ---------------------------
        # SHADOW (optional safe load)
        # ---------------------------
        try:
            shadow_model = load_model(
                model_name=MODEL_NAME,
                uri=get_model_uri(alias="shadow")
            )
        except Exception:
            shadow_model = None
            logger.warning("Shadow model not available")

        # ---------------------------
        # FEATURE CONTRACT
        # ---------------------------
        feature_schema = load_feature_schema()

        logger.info("API initialized successfully")

    except Exception as e:
        logger.exception("Startup failed: %s", str(e))
        raise

    yield

    logger.info("API shutdown complete")


# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="Credit Risk API",
    version=APP_VERSION,
    lifespan=lifespan
)


# =====================================================
# MIDDLEWARE
# =====================================================

@app.middleware("http")
async def telemetry(request: Request, call_next):

    start = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code

    except Exception:
        status = 500
        raise

    finally:
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status=status
        ).inc()

        LATENCY.labels(path=request.url.path).observe(duration)

    return response


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME}


@app.get("/ready")
def ready():

    if model is None:
        raise HTTPException(503, "Model not loaded")

    return {
        "status": "ready",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "shadow_enabled": shadow_model is not None
    }


# =====================================================
# METRICS
# =====================================================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


# =====================================================
# DASHBOARD
# =====================================================

@app.get("/dashboard")
def dashboard():
    return RedirectResponse("http://localhost:8501")


# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):

    global model, shadow_model, feature_schema

    if model is None:
        raise HTTPException(503, "Model not ready")

    start = time.perf_counter()

    try:
        payload = request.model_dump()

        # ---------------------------
        # CONTRACT VALIDATION
        # ---------------------------
        missing = [f for f in feature_schema if f not in payload]

        if missing:
            raise HTTPException(
                422,
                f"Missing features: {missing}"
            )

        # ---------------------------
        # MAIN MODEL
        # ---------------------------
        prob, pred = predict(model, payload)

        # ---------------------------
        # SHADOW MODEL (optional)
        # ---------------------------
        if shadow_model:
            try:
                shadow_model.predict_proba(payload)
            except Exception:
                logger.warning("Shadow inference failed")

        latency_ms = (time.perf_counter() - start) * 1000

        PRED_COUNT.inc()

        log_event({
            "timestamp": datetime.utcnow().isoformat(),
            "probability": float(prob),
            "prediction": int(pred),
            "model": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "latency_ms": round(latency_ms, 3)
        })

        return PredictionResponse(
            probability=float(prob),
            prediction=int(pred)
        )

    except HTTPException:
        raise

    except Exception as e:
        ERROR_COUNT.inc()
        logger.exception("Prediction failed: %s", str(e))
        raise HTTPException(500, "Inference error")


# =====================================================
# GLOBAL ERROR HANDLER
# =====================================================

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):

    logger.exception("Unhandled error: %s", str(exc))

    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"}
    )