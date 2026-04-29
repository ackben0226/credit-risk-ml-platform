import logging
import numpy as np
import pandas as pd

from src.features.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


# =====================================================
# SCHEMA RESOLUTION (HARD CONTRACT)
# =====================================================

def resolve_feature_columns(model):
    """
    Extract feature contract from model metadata.
    """

    metadata = getattr(model, "metadata", None)

    if not metadata:
        raise ValueError("Model missing metadata contract")

    feature_cols = metadata.get("feature_columns")

    if not feature_cols:
        raise ValueError("feature_columns not found in model metadata")

    if not isinstance(feature_cols, list):
        raise ValueError("feature_columns must be a list")

    return feature_cols


# =====================================================
# INPUT VALIDATION (STRICT)
# =====================================================

def validate_raw_input(input_data: dict, expected_cols: list):

    input_cols = set(input_data.keys())
    expected_cols = set(expected_cols)

    missing = expected_cols - input_cols
    extra = input_cols - expected_cols

    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    if extra:
        logger.warning("Extra features ignored: %s", sorted(extra))


# =====================================================
# OUTPUT VALIDATION (ROBUST)
# =====================================================

def validate_prediction_output(proba: np.ndarray):

    if proba is None:
        raise ValueError("Model returned None")

    if not isinstance(proba, np.ndarray):
        raise ValueError("Prediction output must be numpy array")

    if proba.ndim != 2:
        raise ValueError(f"Invalid shape: {proba.shape}")

    if proba.shape[1] != 2:
        raise ValueError("Binary classifier must return 2 probability columns")

    if np.isnan(proba).any():
        raise ValueError("NaN detected in prediction output")

    if np.isinf(proba).any():
        raise ValueError("Inf detected in prediction output")


# =====================================================
# MAIN PREDICTION PIPELINE
# =====================================================

def predict(model, input_data: dict, threshold: float = 0.5):

    logger.debug("Starting inference pipeline")

    # -------------------------------------------------
    # 1. Resolve schema
    # -------------------------------------------------
    feature_columns = resolve_feature_columns(model)

    # -------------------------------------------------
    # 2. Validate input contract
    # -------------------------------------------------
    validate_raw_input(input_data, feature_columns)

    # -------------------------------------------------
    # 3. Transform features
    # -------------------------------------------------
    pipeline = FeaturePipeline(feature_columns)
    X = pipeline.transform(input_data)

    if X is None or len(X) == 0:
        raise ValueError("Empty feature matrix after transformation")

    if list(X.columns) != list(feature_columns):
        raise ValueError("Feature column misalignment detected")

    # -------------------------------------------------
    # 4. Inference
    # -------------------------------------------------
    proba = model.predict_proba(X)

    validate_prediction_output(proba)

    # -------------------------------------------------
    # 5. Extract probability
    # -------------------------------------------------
    prob = float(proba[0, 1])

    # production safety clamp
    prob = np.clip(prob, 0.0, 1.0)

    # -------------------------------------------------
    # 6. Decision
    # -------------------------------------------------
    pred = int(prob >= threshold)

    logger.info(
        "Inference complete | prob=%.4f pred=%s",
        prob,
        pred
    )

    return prob, pred