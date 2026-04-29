# Credit Risk ML Platform  
## Production-Grade MLOps System with Model Registry, Monitoring & Automated Decisioning

---

## 📌 Overview

This project implements a **full end-to-end machine learning platform** for credit risk prediction. It is designed as a **production ML system**, not a standalone model.

The platform covers the complete ML lifecycle:

- Data ingestion & validation
- Feature engineering pipeline
- Training (Logistic Regression, XGBoost, LightGBM)
- Model registry with lifecycle control (MLflow)
- FastAPI inference service
- Shadow deployment system
- Monitoring & observability layer
- Automation engine for model promotion and rollback

The system reflects real-world **regulated fintech ML infrastructure patterns**.

---

## 🧠 System Architecture
```                  Raw Data
                        ↓
Feature Engineering Layer (Batch + Reusable Transformations)
                        ↓
Feature Store (Parquet-based versioned datasets)
                        ↓
Training Pipeline (Logistic Regression / XGBoost / LightGBM)
                        ↓
MLflow Model Registry (Versioning + Aliases)
                        ↓
FastAPI Serving Layer (Real-time inference)
                        ↓
Shadow Model Inference (Parallel evaluation)
                        ↓
Monitoring Layer (Prometheus + Streamlit Dashboard)
                        ↓
Automation Engine (Prefect + Rule-based Decision System)
                        ↓
Rollback / Promotion System
```

## 🏗️ Core Components

### 1. Feature Engineering System
- Modular transformation pipeline:
  - Raw ingestion
  - Bureau feature aggregation
  - Derived feature generation
- Parquet-based feature storage
- Strict schema enforcement between training and inference
- Eliminates training-serving skew

---

### 2. Training Pipeline
Supports multiple models:

- Logistic Regression (baseline)
- XGBoost (primary production model)
- LightGBM (alternative boosting model)

Key features:
- MLflow experiment tracking
- Metric logging (ROC-AUC, PR-AUC)
- Model comparison framework
- Reproducible training runs

---

### 3. Model Registry (MLflow-based)
Extended registry abstraction:

- Versioned model storage
- Alias system:
  - `champion`
  - `challenger`
- Promotion logic with performance gates:
  - AUC threshold validation
  - PR-AUC validation
- Registry health checks at runtime

---

### 4. Model Serving (FastAPI)
Production inference API:

- Strict schema validation (training contract enforcement)
- Central prediction controller
- Probabilistic outputs (risk scoring)
- Event logging for every prediction
- Prometheus metrics:
  - request count
  - latency tracking
  - error rates

---

### 5. Shadow Deployment System
- Dual inference pipeline:
  - Champion model (production traffic)
  - Shadow model (parallel evaluation)
- No impact on production decisions
- Enables safe model testing in real traffic conditions

---

### 6. Monitoring & Observability
Full observability stack:

- Streamlit dashboard:
  - prediction drift tracking
  - probability distribution monitoring
  - latency trends
- Prometheus metrics:
  - API latency histograms
  - request counters
  - error tracking
- Prediction event store (CSV lineage logging)

---

### 7. Automation & Decision Engine
Rule-based ML lifecycle controller:

- Executes full pipeline:
  - Feature pipeline → Training → Registry → Promotion
- Model promotion rules:
  - Minimum ROC-AUC threshold
  - Minimum PR-AUC threshold
- Supports:
  - Model promotion
  - Challenger assignment
  - Rollback logic (extendable)

---

## 🚀 How to Run the Project

### 1. Clone repository
```bash
git https://github.com/ackben0226/ML-Systems-Credit-Risk-Platform-Engineering-Portfolio.git
cd ML-Systems-Credit-Risk-Platform-Engineering-Portfolio

2. Install dependencies
```make install```

3. Run full system using Docker
```docker-compose up --build```

Services:
- API → http://localhost:8000
- MLflow → http://localhost:5000


4. Run API locally
make run

5. Run feature + training pipeline
make pipeline

6. Launch monitoring dashboard
make dashboard

🔍 Example API Request
POST /predict
{"feature_1": 1.2,  "feature_2": 0.8,  "feature_3": 3.1}
Response:
{"probability": 0.73,  "prediction": 1}

🧠 Run Core ML Pipeline
Feature engineering + training pipeline
make pipeline

📊 Monitoring
- Streamlit dashboard: prediction drift + latency
make dashboard

- Prometheus: system metrics
- CSV event store: full prediction lineage tracking

📉 Run Advanced Graph Visualisations
streamlit run dashboard.py

📡 MLflow Tracking UI
make mlflow

⚙️ Engineering Principles
1. Reproducibility
- Versioned datasets
- MLflow tracking
- Deterministic feature pipelines

2. Separation of Concerns
- Feature engineering ≠ training ≠ serving
- Registry acts as control plane

3. Production Safety
- Schema enforcement at inference boundary
- Shadow model validation before promotion
- Controlled model lifecycle transitions

4. Observability First
- Every prediction logged
- System metrics exposed via Prometheus
- Drift monitoring in dashboard

5. Failure Isolation
- API independent of training system
- Registry failures do not affect inference
- Monitoring decoupled from model logic

⚠️ Key Problems Solved
- Training–Serving Skew → solved via schema contract
- Model Lifecycle Control → solved via MLflow registry + aliases
- Production Monitoring Gap → solved via Prometheus + Streamlit
- Safe Model Updates → solved via shadow deployment system

📈 System Maturity
ComponentMaturityFeature Pipeline8/10Training System8/10Model Registry8.5/10API Serving8.5/10Monitoring8/10Shadow Deployment7.5/10Automation Engine7.5/10Overall MLOps System8.5/10

🧭 Future Improvements
- True feature store (Feast/Tecton-style)
- Kubernetes deployment (autoscaling inference)
- Automated rollback system based on drift detection
- CI/CD pipeline for ML training (GitHub Actions)
- Canary deployment for model rollout

🧠 Summary
This project demonstrates a production-grade ML platform architecture implementing:
- End-to-end MLOps lifecycle
- Model governance and registry control
- Real-time inference system
- Shadow deployment strategy
- Monitoring and drift detection
- Automation-driven model promotion

It is designed to reflect systems used in fintech credit scoring, regulated lending platforms, and enterprise ML infrastructure environments.

