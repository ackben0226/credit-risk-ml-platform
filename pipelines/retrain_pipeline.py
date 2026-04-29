# src/pipelines/retrain_pipeline.py

import os
import json
import logging
import joblib
import mlflow

from datetime import datetime

from src.training.train_model import (
    load_data,
    split_data,
    build_logistic,
    build_xgboost,
    build_lightgbm,
    train_and_log
)

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/processed/train_features.parquet"

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# LOAD CURRENT MODEL METRICS
# =====================================================

def load_current_auc():
    if not os.path.exists(METRICS_PATH):
        return None

    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f).get("auc")
    except Exception:
        return None


# =====================================================
# DECISION LOGIC
# =====================================================

def should_replace(new_auc: float, current_auc: float | None) -> bool:
    """
    Only replace if meaningful improvement exists.
    """
    if current_auc is None:
        return True

    return new_auc > current_auc + 0.005


# =====================================================
# RETRAIN PIPELINE
# =====================================================

def run_retrain():

    logger.info("Starting retraining pipeline")

    # ----------------------------
    # DATA
    # ----------------------------
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # ----------------------------
    # MODELS (UPDATED)
    # ----------------------------
    models = {
        "logistic": build_logistic(X_train),
        "xgboost": build_xgboost(X_train),
        "lightgbm": build_lightgbm(X_train)
    }

    # ----------------------------
    # TRAIN ALL CANDIDATES
    # ----------------------------
    results = []

    for name, model in models.items():

        logger.info("Training %s", name)

        result = train_and_log(
            model=model,
            name=f"retrain_{name}",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

        results.append(result)

    # ----------------------------
    # SELECT BEST MODEL
    # ----------------------------
    best_candidate = max(results, key=lambda x: x["auc"])

    new_auc = best_candidate["auc"]

    logger.info("Best candidate: AUC=%.4f", new_auc)

    # ----------------------------
    # CURRENT MODEL COMPARISON
    # ----------------------------
    current_auc = load_current_auc()

    logger.info("Current AUC: %s", current_auc)

    # ----------------------------
    # PROMOTION DECISION
    # ----------------------------
    if should_replace(new_auc, current_auc):

        logger.info("New model approved for promotion")

        joblib.dump(best_candidate["model"], BEST_MODEL_PATH)

        with open(METRICS_PATH, "w") as f:
            json.dump(
                {
                    "auc": float(new_auc),
                    "model": best_candidate["name"],
                    "updated_at": datetime.utcnow().isoformat()
                },
                f,
                indent=2
            )

        logger.info("Model promoted successfully")

    else:
        logger.info("Model rejected (no meaningful improvement)")

    logger.info("Retraining pipeline complete")


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    run_retrain()