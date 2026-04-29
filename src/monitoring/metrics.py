# src/monitoring/inference_metrics.py

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, asdict
from functools import wraps
from typing import Any, Dict, Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================
# CONFIG
# =====================================================

DEFAULT_THRESHOLD = 0.50
LATENCY_WARN_MS = 500
NULL_WARN_RATE = 0.20


# =====================================================
# METRICS MODEL
# =====================================================

@dataclass(slots=True)
class InferenceMetrics:
    request_id: str
    run_id: str
    model_name: str
    model_version: Optional[str]

    n_rows: int
    n_features: int

    latency_ms: float

    positive_rate: float
    mean_probability: float
    min_probability: float
    max_probability: float

    null_rate: float
    duplicate_rate: float

    drifted_features: int

    threshold: float
    created_at: float


# =====================================================
# TIMER
# =====================================================

class Timer:
    def __init__(self):
        self._start = 0.0
        self._end = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000


# =====================================================
# HELPERS
# =====================================================

def generate_request_id() -> str:
    return str(uuid.uuid4())


def _safe_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 0:
        arr = np.array([float(arr)])

    return arr


def _safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


# =====================================================
# CORE METRICS
# =====================================================

def compute_inference_metrics(
    *,
    probabilities,
    predictions,
    input_df: pd.DataFrame,
    latency_ms: float,
    model_name: str,
    model_version: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    run_id: Optional[str] = None,
    request_id: Optional[str] = None,
    drift_report: Optional[Dict[str, Any]] = None
) -> InferenceMetrics:
    """
    Central production inference metrics object.
    """

    probs = _safe_array(probabilities)
    preds = _safe_array(predictions)

    run_id = run_id or generate_request_id()
    request_id = request_id or generate_request_id()

    drifted_features = 0

    if drift_report:
        drifted_features = sum(
            1
            for value in drift_report.values()
            if isinstance(value, dict)
            and value.get("drift_detected") is True
        )

    metrics = InferenceMetrics(
        request_id=request_id,
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        n_rows=len(input_df),
        n_features=input_df.shape[1],
        latency_ms=float(latency_ms),
        positive_rate=float(preds.mean()) if len(preds) else 0.0,
        mean_probability=float(probs.mean()) if len(probs) else 0.0,
        min_probability=float(probs.min()) if len(probs) else 0.0,
        max_probability=float(probs.max()) if len(probs) else 0.0,
        null_rate=float(input_df.isna().mean().mean()),
        duplicate_rate=float(input_df.duplicated().mean()),
        drifted_features=drifted_features,
        threshold=float(threshold),
        created_at=time.time()
    )

    _emit_alerts(metrics)

    logger.info("Inference metrics | %s", asdict(metrics))

    return metrics


# =====================================================
# ALERTING SIGNALS
# =====================================================

def _emit_alerts(metrics: InferenceMetrics) -> None:
    if metrics.latency_ms > LATENCY_WARN_MS:
        logger.warning(
            "High latency detected | request_id=%s latency_ms=%.2f",
            metrics.request_id,
            metrics.latency_ms
        )

    if metrics.null_rate > NULL_WARN_RATE:
        logger.warning(
            "High null rate detected | request_id=%s null_rate=%.4f",
            metrics.request_id,
            metrics.null_rate
        )

    if metrics.drifted_features > 0:
        logger.warning(
            "Feature drift detected | request_id=%s drifted=%s",
            metrics.request_id,
            metrics.drifted_features
        )


# =====================================================
# SERIALIZATION
# =====================================================

def metrics_to_dict(metrics: InferenceMetrics) -> Dict[str, Any]:
    return asdict(metrics)


# =====================================================
# HEALTH SIGNALS
# =====================================================

def compute_health_signals(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "row_count": 0.0,
            "null_rate": 0.0,
            "duplicate_rate": 0.0
        }

    return {
        "row_count": float(len(df)),
        "null_rate": float(df.isna().mean().mean()),
        "duplicate_rate": float(df.duplicated().mean())
    }


# =====================================================
# DECORATOR
# =====================================================

def track_latency(func: Callable):
    """
    Decorator for inference functions.
    Returns (result, latency_ms)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer() as timer:
            result = func(*args, **kwargs)

        latency = timer.elapsed_ms

        logger.info(
            "Function latency | fn=%s latency_ms=%.2f",
            func.__name__,
            latency
        )

        return result, latency

    return wrapper