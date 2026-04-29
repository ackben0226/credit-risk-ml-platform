import numpy as np
from config.settings import Config
from core.logger import get_logger
from src.features.feature_pipeline import FeaturePipeline

logger = get_logger(__name__)


# =====================================================
# FEATURE RESOLUTION
# =====================================================

def _resolve_features(model, input_data: dict):
    """
    Resolve feature schema from highest-priority source.
    """

    if hasattr(model, "metadata") and model.metadata:
        cols = model.metadata.get("feature_columns")
        if cols:
            return cols

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    logger.warning("Falling back to input keys for feature schema")
    return list(input_data.keys())


# =====================================================
# SINGLE ROW PREDICTION (API USE CASE)
# =====================================================

def predict(model, input_data: dict, threshold: float | None = None):
    """
    Single-record inference (FastAPI / real-time scoring).
    """

    threshold = threshold or Config.THRESHOLD

    feature_columns = _resolve_features(model, input_data)

    pipeline = FeaturePipeline(feature_columns)
    X = pipeline.transform(input_data)

    proba = model.predict_proba(X)

    prob = float(proba[0, 1] if proba.ndim == 2 else proba[0])
    pred = int(prob >= threshold)

    return prob, pred


# =====================================================
# BATCH PREDICTION (PIPELINES / OFFLINE SCORING)
# =====================================================

def predict_batch(model, X, threshold: float | None = None):
    """
    Batch inference (pre-built feature matrix).
    """

    threshold = threshold or Config.THRESHOLD

    proba = model.predict_proba(X)[:, 1]

    preds = (proba >= threshold).astype(int)

    return proba, preds