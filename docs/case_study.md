# Advanced Credit Risk Decision System

**Production-Grade ML Platform | End-to-End MLOps | Real-Time Decisioning**

---

## 1. Problem Context

Credit risk assessment in lending requires **accurate, explainable, and production-ready models** that can operate reliably under real-world constraints (data quality issues, latency requirements, model lifecycle control).

This project was designed to simulate a **regulated fintech environment**, focusing not just on model accuracy, but on **deployment, monitoring, and governance**.

---

## 2. System Architecture

```
Raw Data (Application + Bureau)
        ↓
Feature Engineering Pipeline
        ↓
Feature Store (Versioned Parquet)
        ↓
Model Training (LR / XGBoost / LightGBM)
        ↓
MLflow Registry (Champion / Challenger)
        ↓
FastAPI Inference Service
        ↓
Shadow Model Evaluation
        ↓
Monitoring (Prometheus + Dashboard)
        ↓
Automation (Promotion / Rollback Logic)
```

### Key Architectural Decisions

* **Separation of concerns**: feature engineering, training, and serving decoupled
* **Registry-driven control plane**: MLflow governs model lifecycle
* **Deterministic pipelines**: eliminates training–serving skew
* **Shadow deployment**: safe validation before promotion

---

## 3. Data & Feature Engineering

* Dataset: **~300K+ records**, 100+ features (application + bureau data)
* Engineered features:

  * Credit-to-income ratio
  * Annuity ratios
  * External risk score aggregations
  * Bureau debt and overdue metrics

### Design Choices

* Built **modular feature pipelines** (aggregation + derivation layers)
* Enforced **strict schema contracts** at training and inference
* Applied **data validation rules** (range checks, anomaly handling)

### Outcome

* Reduced data inconsistencies and pipeline failures
* Enabled **consistent feature generation across environments**

---

## 4. Modelling Approach

### Models Evaluated

* Logistic Regression (baseline)
* XGBoost (primary)
* LightGBM (alternative)

### Evaluation Strategy

* Cross-validation with **ROC-AUC as primary metric**
* Stability across folds used for model selection

### Results

* Baseline (Logistic Regression): **ROC-AUC = 0.72**
* Final Model (XGBoost): **ROC-AUC = 0.81**
* ~**12.5% relative performance improvement**

### Decision Rationale

* XGBoost selected for:

  * Higher predictive power
  * Better handling of non-linear relationships
  * Stable performance across validation splits

---

## 5. Production System Design

### Inference Layer

* Built using **FastAPI**
* Real-time scoring with:

  * Input schema validation
  * Feature alignment checks
  * Prediction output validation

### Performance

* **<100ms latency per request (local benchmark)**
* Supports both **single and batch predictions**

### Deployment

* Fully **containerised using Docker**
* Ensures reproducibility across environments

---

## 6. Model Lifecycle & Governance

### MLflow Registry

* Versioned model tracking
* Alias system:

  * `champion` (production)
  * `challenger` (candidate)

### Promotion Logic

* Models promoted only if:

  * ROC-AUC threshold met
  * Performance validated against baseline

### Benefit

* Enables **controlled, auditable model updates**
* Prevents unverified models reaching production

---

## 7. Monitoring & Risk Control

### Observability Stack

* **Prometheus metrics**:

  * API latency
  * request volume
  * error rates

* Dashboard (Streamlit):

  * Prediction distribution tracking
  * Drift monitoring

### Shadow Deployment

* Runs **challenger model in parallel**
* No impact on live decisions
* Enables **safe real-world evaluation**

### Outcome

* Reduced risk of model degradation
* Enabled early detection of performance drift

---

## 8. Business Impact Simulation

Using threshold analysis on model outputs:

* Potential **12–18% reduction in high-risk approvals**
* Improved decision boundary control between:

  * approval rates
  * default risk

### Interpretation

* System supports **risk-aware lending decisions**
* Demonstrates how ML can influence:

  * credit policy
  * portfolio risk exposure

---

## 9. Engineering Maturity

| Component         | Maturity |
| ----------------- | -------- |
| Feature Pipeline  | 8/10     |
| Training System   | 8/10     |
| Model Registry    | 8.5/10   |
| API Serving       | 8.5/10   |
| Monitoring        | 8/10     |
| Shadow Deployment | 7.5/10   |
| Automation Engine | 7.5/10   |

---

## 10. Key Takeaways

* Built a **full ML lifecycle system**, not just a model
* Demonstrated **production-readiness**:

  * deployment
  * monitoring
  * governance
* Solved critical industry problems:

  * training–serving skew
  * model lifecycle control
  * safe deployment strategies

---

## 11. Next Improvements

* Introduce **true feature store (e.g. Feast)**
* Deploy on **Kubernetes for scalability**
* Implement **automated rollback based on drift detection**
* Add **CI/CD pipelines for training and deployment**

---

## Final Positioning

This project demonstrates capability to:

* Build and productionise ML systems
* Apply data science in **risk-sensitive environments**
* Bridge **modelling, engineering, and business decisioning**

It aligns directly with real-world systems used in:

* Credit scoring platforms
* Lending risk engines
* Financial decision support systems
