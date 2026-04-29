from pathlib import Path
import json
import logging
import random
from datetime import datetime, timezone

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "train_features.parquet"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCHEMA_PATH = MODEL_DIR / "feature_schema.pkl"
META_PATH = MODEL_DIR / "metadata.json"
REPORT_PATH = MODEL_DIR / "classification_report.txt"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

TEST_SIZE = 0.20
RANDOM_STATE = 42
THRESHOLD = 0.50

MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_experiment("credit-risk-models")


# =====================================================
# REPRODUCIBILITY
# =====================================================

def set_seeds() -> None:
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)


# =====================================================
# DATA
# =====================================================

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty")

    if TARGET not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET}")

    logger.info("Loaded dataset | rows=%s cols=%s", df.shape[0], df.shape[1])

    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


# =====================================================
# PREPROCESSING
# =====================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    logger.info(
        "Preprocessor | numeric=%s categorical=%s",
        len(numeric_cols),
        len(categorical_cols)
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]
                ),
                numeric_cols
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False
                            )
                        )
                    ]
                ),
                categorical_cols
            )
        ],
        remainder="drop"
    )


# =====================================================
# MODEL BUILDERS
# =====================================================

def build_logistic(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )


def build_xgboost(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(X)),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    min_child_weight=3,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist"
                )
            )
        ]
    )


def build_lightgbm(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(X)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    num_leaves=31,
                    min_child_samples=30,
                    class_weight="balanced",
                    objective="binary",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1
                )
            )
        ]
    )


# =====================================================
# TRAIN / EVALUATE
# =====================================================

def evaluate_predictions(y_true, probs, threshold=THRESHOLD):
    preds = (probs >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "report": classification_report(y_true, preds)
    }


def train_and_log(model, name, X_train, X_test, y_train, y_test):

    logger.info("Training model: %s", name)

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, probs)

        mlflow.log_params(
            {
                "model_name": name,
                "rows_train": X_train.shape[0],
                "features": X_train.shape[1],
                "threshold": THRESHOLD
            }
        )

        mlflow.log_metrics(
            {
                "auc": metrics["auc"],
                "pr_auc": metrics["pr_auc"]
            }
        )

        signature = infer_signature(X_test, probs)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="credit-risk-model",
            signature=signature,
            input_example=X_test.head(5),
            serialization_format="cloudpickle"
        )

        logger.info(
            "%s complete | AUC=%.4f | PR_AUC=%.4f",
            name,
            metrics["auc"],
            metrics["pr_auc"]
        )

        return {
            "name": name,
            "model": model,
            "auc": metrics["auc"],
            "pr_auc": metrics["pr_auc"],
            "report": metrics["report"]
        }


# =====================================================
# SAVE ARTIFACTS
# =====================================================

def save_artifacts(best, feature_columns):

    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(list(feature_columns), SCHEMA_PATH)

    REPORT_PATH.write_text(best["report"], encoding="utf-8")

    metadata = {
        "model": best["name"],
        "auc": round(best["auc"], 6),
        "pr_auc": round(best["pr_auc"], 6),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "threshold": THRESHOLD
    }

    META_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    logger.info("Artifacts saved -> %s", MODEL_DIR)


# =====================================================
# MAIN
# =====================================================

def main():

    set_seeds()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model_builders = {
        "logistic_regression": build_logistic,
        "xgboost": build_xgboost,
        "lightgbm": build_lightgbm,
    }

    candidates = []

    for name, builder in model_builders.items():

        try:
            result = train_and_log(
                model=builder(X_train),
                name=name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

            candidates.append(result)

        except Exception as e:
            logger.exception(
                "Training failed for %s: %s",
                name,
                str(e)
            )

    if not candidates:
        raise RuntimeError("No models trained successfully")

    best = max(
        candidates,
        key=lambda x: x["auc"]
    )

    logger.info(
        "Champion selected | %s | AUC=%.4f | PR_AUC=%.4f",
        best["name"],
        best["auc"],
        best["pr_auc"]
    )

    save_artifacts(best, X_train.columns)


if __name__ == "__main__":
    main()