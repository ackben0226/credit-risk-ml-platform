# =====================================================
# dashboard.py
# Production ML Monitoring Dashboard (v2)
# =====================================================

import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="ML Production Monitoring",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

HEALTH_URL = f"{API_URL}/health"
READY_URL = f"{API_URL}/ready"
METRICS_URL = f"{API_URL}/metrics"

BASE_DIR = Path(__file__).resolve().parent

PRED_PATH = BASE_DIR / "data" / "predictions" / "predictions.csv"
META_PATH = BASE_DIR / "models" / "metadata.json"


# =====================================================
# HELPERS (CLEANED)
# =====================================================

def get_json(url):
    try:
        r = requests.get(url, timeout=3)
        return r.status_code, r.json()
    except Exception:
        return 500, {}


def get_text(url):
    try:
        return requests.get(url, timeout=3).text
    except Exception:
        return ""


def load_predictions():
    if not PRED_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(PRED_PATH)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def load_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


# =====================================================
# TITLE
# =====================================================

st.title("ML Production Monitoring Dashboard")


# =====================================================
# SYSTEM HEALTH
# =====================================================

st.header("System Health")

col1, col2 = st.columns(2)

with col1:
    code, health = get_json(HEALTH_URL)
    st.metric("Health Status", code)
    st.json(health)

with col2:
    code, ready = get_json(READY_URL)
    st.metric("Readiness Status", code)
    st.json(ready)


# =====================================================
# MODEL METADATA
# =====================================================

st.header("Model Registry Snapshot")

meta = load_metadata()

if meta:
    st.json(meta)
else:
    st.warning("No model metadata available")


# =====================================================
# PREDICTION DATA
# =====================================================

df = load_predictions()

st.header("Prediction Stream")

if df.empty:
    st.warning("No prediction logs found.")
    st.stop()


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Predictions", len(df))

with col2:
    st.metric("Avg Probability", round(df["probability"].mean(), 4))

with col3:
    st.metric("Positive Rate", round(df["prediction"].mean(), 4))

with col4:
    st.metric("Unique Runs", df["run_id"].nunique() if "run_id" in df else 1)

st.dataframe(df.tail(50), use_container_width=True)


# =====================================================
# DRIFT MONITORING (IMPROVED)
# =====================================================

st.header("Drift Monitoring (Windowed)")

WINDOW = 200

if "probability" in df.columns:

    recent = df.tail(WINDOW)
    baseline = df.iloc[:-WINDOW] if len(df) > WINDOW else df

    def compute_psi(baseline, recent, bins=10):
        import numpy as np

        hist_b, _ = np.histogram(baseline, bins=bins, range=(0, 1))
        hist_r, _ = np.histogram(recent, bins=bins, range=(0, 1))

        hist_b = hist_b / (hist_b.sum() + 1e-6)
        hist_r = hist_r / (hist_r.sum() + 1e-6)

        psi = np.sum((hist_b - hist_r) * np.log((hist_b + 1e-6) / (hist_r + 1e-6)))
        return float(psi)

    psi_score = compute_psi(
        baseline["probability"],
        recent["probability"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline Mean", round(baseline["probability"].mean(), 4))
    col2.metric("Recent Mean", round(recent["probability"].mean(), 4))
    col3.metric("PSI Drift Score", round(psi_score, 4))

    st.plotly_chart(
        px.line(df.tail(300), x="timestamp", y="probability",
                title="Prediction Trend Over Time"),
        use_container_width=True
    )

    st.plotly_chart(
        px.histogram(df, x="probability", nbins=30,
                     title="Probability Distribution"),
        use_container_width=True
    )


# =====================================================
# LATENCY + ERROR METRICS
# =====================================================

st.header("API Performance")

metrics_raw = get_text(METRICS_URL)


def safe_extract(metric):
    import re
    match = re.search(rf"{metric}\s+([0-9\.eE+-]+)", metrics_raw)
    return float(match.group(1)) if match else 0.0


lat_sum = safe_extract("api_request_latency_seconds_sum")
lat_count = safe_extract("api_request_latency_seconds_count")

avg_latency = lat_sum / lat_count if lat_count else 0

error_count = safe_extract("prediction_errors_total")
req_count = safe_extract("prediction_requests_total")

error_rate = error_count / req_count if req_count else 0

c1, c2, c3 = st.columns(3)

c1.metric("Avg Latency (s)", round(avg_latency, 5))
c2.metric("Errors", int(error_count))
c3.metric("Error Rate", round(error_rate, 4))


# =====================================================
# CLASS DISTRIBUTION
# =====================================================

st.header("Prediction Distribution")

if "prediction" in df.columns:

    dist = df["prediction"].value_counts().reset_index()
    dist.columns = ["class", "count"]

    st.plotly_chart(
        px.bar(dist, x="class", y="count", title="Class Distribution"),
        use_container_width=True
    )


# =====================================================
# RAW METRICS
# =====================================================

with st.expander("Raw Metrics"):
    st.code(metrics_raw[:8000])