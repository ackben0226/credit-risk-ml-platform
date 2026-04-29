# src/api/shadow.py

from __future__ import annotations

import logging
import threading
from typing import Any

import pandas as pd

from core.load_model import load_model

logger = logging.getLogger(__name__)

# =====================================================
# GLOBAL CACHE
# =====================================================

_shadow_model = None
_shadow_lock = threading.Lock()


# =====================================================
# LOAD SHADOW MODEL (LAZY + THREAD SAFE)
# =====================================================

def get_shadow_model():
    """
    Load shadow model once and cache in memory.

    Registry alias expected:
        models:/credit-risk-model@shadow
    """

    global _shadow_model

    if _shadow_model is None:
        with _shadow_lock:
            if _shadow_model is None:
                logger.info("Loading shadow model")
                _shadow_model = load_model(alias="shadow")
                logger.info("Shadow model loaded")

    return _shadow_model


# =====================================================
# RESET CACHE (FOR HOT RELOADS)
# =====================================================

def refresh_shadow_model():
    """
    Force reload on next request.
    Useful after alias repointing.
    """
    global _shadow_model

    with _shadow_lock:
        _shadow_model = None

    logger.info("Shadow model cache cleared")


# =====================================================
# INPUT NORMALIZATION
# =====================================================

def _to_frame(input_data: Any) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])

    if isinstance(input_data, list):
        return pd.DataFrame(input_data)

    raise ValueError("Unsupported input format for shadow inference")


# =====================================================
# INFERENCE
# =====================================================

def shadow_predict(input_data) -> float:
    """
    Run silent shadow prediction.

    Returns:
        probability for positive class
    """

    model = get_shadow_model()

    X = _to_frame(input_data)

    probs = model.predict_proba(X)

    if probs.ndim == 2:
        return float(probs[0, 1])

    return float(probs[0])


# =====================================================
# SAFE WRAPPER (NON-BLOCKING CALLS)
# =====================================================

def safe_shadow_predict(input_data):
    """
    Never raise into production request path.
    """

    try:
        return shadow_predict(input_data)

    except Exception as exc:
        logger.exception("Shadow inference failed: %s", str(exc))
        return None