import os
import json
import time
import uuid
import logging
from datetime import datetime

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import create_engine

from src.features.feature_pipeline import FeaturePipeline
from core.score import predict_batch


# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "models/best_model.pkl"
SCHEMA_PATH = "models/feature_schema.pkl"
META_PATH = "models/metadata.json"

TRAIN_STATS_PATH = "models/train_reference.parquet"
CONFIG_PATH = "config/scoring_config.json"

OUTPUT_DIR = "data/scored"
REPORT_DIR = "reports"

DEFAULT_THRESHOLD = 0.5
DEFAULT_TABLE = "credit_risk_scores"
MAX_NULL_RATE = 0.40

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# CONFIG LOADER
# =====================================================

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {
            "threshold": DEFAULT_THRESHOLD,
            "export_to_db": False,
            "table_name": DEFAULT_TABLE,
            "drift_alpha": 0.05
        }

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# ARTIFACT LOADING
# =====================================================

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found")

    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError("Schema not found")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(SCHEMA_PATH)

    metadata = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, feature_columns, metadata


# =====================================================
# DATA LOADING
# =====================================================

def load_batch_data(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Only CSV or parquet supported")

    if df.empty:
        raise ValueError("Empty batch input")

    return df


# =====================================================
# NULL VALIDATION
# =====================================================

def validate_null_rate(df: pd.DataFrame) -> float:

    missing_rate = float(df.isna().mean().mean())

    logger.info("Null rate: %.4f", missing_rate)

    if missing_rate > MAX_NULL_RATE:
        raise ValueError(f"Batch rejected due to high null rate: {missing_rate:.2%}")

    return missing_rate


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def build_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:

    pipeline = FeaturePipeline(feature_columns)

    X = pipeline.transform(df.to_dict(orient="records"))

    return X


# =====================================================
# DRIFT DETECTION
# =====================================================

def detect_drift(batch_df: pd.DataFrame, alpha: float):

    if not os.path.exists(TRAIN_STATS_PATH):
        logger.warning("No training reference data for drift")
        return {}

    train_df = pd.read_parquet(TRAIN_STATS_PATH)

    report = {}

    numeric_cols = batch_df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:

        if col not in train_df.columns:
            continue

        ref = train_df[col].dropna()
        cur = batch_df[col].dropna()

        if len(ref) < 10 or len(cur) < 10:
            continue

        stat, p = ks_2samp(ref, cur)

        report[col] = {
            "ks_stat": float(stat),
            "p_value": float(p),
            "drift_detected": bool(p < alpha)
        }

    return report


# =====================================================
# SAVE DRIFT REPORT
# =====================================================

def save_drift_report(run_id: str, drift_report: dict):

    path = os.path.join(
        REPORT_DIR,
        f"drift_{datetime.utcnow().strftime('%Y%m%d')}.json"
    )

    payload = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat(),
        "features": drift_report
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


# =====================================================
# SAVE OUTPUT
# =====================================================

def save_results(df: pd.DataFrame):

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    path = os.path.join(OUTPUT_DIR, f"batch_score_{ts}.parquet")

    df.to_parquet(path, index=False)

    return path


# =====================================================
# DATABASE EXPORT
# =====================================================

def export_to_db(df: pd.DataFrame, table_name: str):

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        logger.warning("DATABASE_URL not set")
        return

    engine = create_engine(db_url)

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000
    )


# =====================================================
# MAIN PIPELINE
# =====================================================

def batch_score(input_path: str):

    start = time.time()
    run_id = str(uuid.uuid4())

    logger.info("Batch scoring started | run_id=%s", run_id)

    config = load_config()
    threshold = config.get("threshold", DEFAULT_THRESHOLD)

    model, feature_columns, metadata = load_artifacts()

    raw_df = load_batch_data(input_path)

    validate_null_rate(raw_df)

    X = build_features(raw_df, feature_columns)

    drift_report = detect_drift(raw_df, config.get("drift_alpha", 0.05))
    drift_path = save_drift_report(run_id, drift_report)

    probs, preds = predict_batch(model, X, threshold)

    result = raw_df.copy()
    result["probability"] = probs
    result["prediction"] = preds
    result["run_id"] = run_id
    result["scored_at"] = datetime.utcnow()

    output_path = save_results(result)

    if config.get("export_to_db", False):
        export_to_db(result, config.get("table_name", DEFAULT_TABLE))

    runtime = round(time.time() - start, 2)

    summary = {
        "run_id": run_id,
        "model_version": metadata.get("trained_at", "unknown"),
        "rows_scored": len(result),
        "threshold": threshold,
        "runtime_sec": runtime,
        "null_rate": validate_null_rate(raw_df),
        "output_path": output_path,
        "drift_report_path": drift_path
    }

    logger.info("Batch scoring completed | %s", summary)

    return summary


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)

    args = parser.parse_args()

    batch_score(args.input)