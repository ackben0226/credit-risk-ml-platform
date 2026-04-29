import json
import logging
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from mlflow.models.signature import infer_signature

from src.training.train_model import (
    build_logistic,
    build_xgboost,
    build_lightgbm,
    split_data,
    load_data,
    save_artifacts
)

from config.paths import MODEL_DIR

# =====================================================
# CONFIG
# =====================================================

SEED = 42
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_experiment("credit-risk-tuning")

# =====================================================
# PARAM GRIDS (CORRECTED)
# =====================================================

LOGISTIC_GRID = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": ["balanced"]
}

XGB_GRID = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma": [0, 0.5, 1.0]
}

LGM_GRID = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [-1, 5, 7],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__num_leaves": [31, 63, 127],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0],
    "model__min_child_samples": [20, 40, 60],
    "model__reg_alpha": [0, 0.1, 0.5],
    "model__reg_lambda": [0, 1, 5]
}

# =====================================================
# CORE TUNING FUNCTION
# =====================================================

def tune_model(model, grid, X_train, y_train):

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_score_, search.best_params_

# =====================================================
# PIPELINE
# =====================================================

def run_tuning():

    logger.info("Starting tuning pipeline")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    models = {
        "logistic": (build_logistic(X_train), LOGISTIC_GRID),
        "xgboost": (build_xgboost(X_train), XGB_GRID),
        "lightgbm": (build_lightgbm(X_train), LGM_GRID)
    }

    results = {}

    for name, (model, grid) in models.items():

        with mlflow.start_run(run_name=f"tune_{name}"):

            best_model, cv_auc, best_params = tune_model(
                model, grid, X_train, y_train
            )

            probs = best_model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            test_auc = roc_auc_score(y_test, probs)
            report = classification_report(y_test, preds)

            mlflow.log_metrics({
                "cv_auc": cv_auc,
                "test_auc": test_auc
            })

            mlflow.log_params(best_params)

            signature = infer_signature(X_test, probs)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(5),
                registered_model_name="credit-risk-model"
            )

            results[name] = {
                "model": best_model,
                "cv_auc": cv_auc,
                "test_auc": test_auc,
                "params": best_params,
                "report": report
            }

            logger.info("%s | CV=%.4f | TEST=%.4f", name, cv_auc, test_auc)

    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best = results[best_name]

    logger.info("BEST MODEL: %s | AUC=%.4f", best_name, best["test_auc"])

    save_artifacts(
        {
            "name": best_name,
            "model": best["model"],
            "auc": best["test_auc"],
            "report": best["report"]
        },
        feature_columns=list(X_train.columns)
    )

    summary_path = MODEL_DIR / "tuning_summary.json"

    json.dump(
        {
            k: {
                "cv_auc": v["cv_auc"],
                "test_auc": v["test_auc"],
                "params": v["params"]
            }
            for k, v in results.items()
        },
        open(summary_path, "w"),
        indent=2
    )

    return best