# =====================================================
# src/monitoring/graph.py
# ML Observability Visualization Layer
# =====================================================

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# PREDICTION OVER TIME
# =====================================================

def plot_prediction_trend(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    prob_col: str = "probability",
    window: int = 300
):
    """
    Line chart of prediction probabilities over time.
    """

    if df.empty or prob_col not in df.columns:
        return go.Figure()

    data = df.tail(window)

    fig = px.line(
        data,
        x=time_col,
        y=prob_col,
        title="Prediction Probability Over Time"
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Probability",
        template="plotly_white"
    )

    return fig


# =====================================================
# PROBABILITY DISTRIBUTION
# =====================================================

def plot_probability_distribution(
    df: pd.DataFrame,
    prob_col: str = "probability",
    bins: int = 30
):
    """
    Distribution of model outputs.
    """

    if df.empty or prob_col not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df,
        x=prob_col,
        nbins=bins,
        title="Prediction Probability Distribution"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# =====================================================
# CLASS DISTRIBUTION
# =====================================================

def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "prediction"
):
    """
    Bar chart of predicted class counts.
    """

    if df.empty or target_col not in df.columns:
        return go.Figure()

    counts = df[target_col].value_counts().reset_index()
    counts.columns = ["class", "count"]

    fig = px.bar(
        counts,
        x="class",
        y="count",
        title="Prediction Class Distribution"
    )

    fig.update_layout(template="plotly_white")

    return fig


# =====================================================
# DRIFT VISUALIZATION (PSI-BASED VIEW)
# =====================================================

def plot_drift_comparison(
    df: pd.DataFrame,
    prob_col: str = "probability",
    window: int = 200,
    bins: int = 10
):
    """
    Visual comparison between baseline and recent window.
    """

    if df.empty or prob_col not in df.columns:
        return go.Figure()

    if len(df) < window * 2:
        return go.Figure()

    recent = df.tail(window)
    baseline = df.iloc[:-window]

    def hist(arr):
        h, _ = np.histogram(arr, bins=bins, range=(0, 1))
        return h / (h.sum() + 1e-8)

    b = hist(baseline[prob_col])
    r = hist(recent[prob_col])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=b,
        name="Baseline"
    ))

    fig.add_trace(go.Bar(
        y=r,
        name="Recent"
    ))

    fig.update_layout(
        title="Drift Comparison (Baseline vs Recent)",
        barmode="group",
        template="plotly_white"
    )

    return fig


# =====================================================
# FEATURE DRIFT SNAPSHOT
# =====================================================

def plot_feature_drift_summary(drift_report: dict):
    """
    Visual summary of drifted features.
    """

    if not drift_report:
        return go.Figure()

    features = []
    values = []

    for k, v in drift_report.items():
        if isinstance(v, dict):
            features.append(k)
            values.append(int(v.get("drift_detected", 0)))

    fig = px.bar(
        x=features,
        y=values,
        title="Feature Drift Summary"
    )

    fig.update_layout(template="plotly_white")

    return fig


# =====================================================
# LATENCY TREND (OPTIONAL FUTURE EXTENSION)
# =====================================================

def plot_latency_series(df: pd.DataFrame):
    """
    If latency logs are stored per request.
    """

    if df.empty or "latency_ms" not in df.columns:
        return go.Figure()

    fig = px.line(
        df,
        y="latency_ms",
        title="API Latency Over Time"
    )

    fig.update_layout(template="plotly_white")

    return fig