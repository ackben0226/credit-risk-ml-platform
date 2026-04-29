# src/evaluation/evaluate.py

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.50


# =====================================================
# CORE EVALUATION
# =====================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, float]:

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "precision": float(
            precision_score(y_test, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, preds, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, preds, zero_division=0)
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold)
    }

    logger.info(
        "Evaluation | AUC=%.4f PR_AUC=%.4f F1=%.4f Recall=%.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1"],
        metrics["recall"]
    )

    return metrics


# =====================================================
# MODEL COMPARISON
# =====================================================

def compare_models(
    results: Dict[str, dict],
    metric: str = "roc_auc"
) -> Tuple[str, dict]:

    if not results:
        raise ValueError("No model results provided")

    if metric not in next(iter(results.values())):
        raise ValueError(f"Metric not found: {metric}")

    best_name = max(
        results,
        key=lambda name: results[name][metric]
    )

    logger.info(
        "Best model selected | %s | %s=%.4f",
        best_name,
        metric,
        results[best_name][metric]
    )

    return best_name, results[best_name]