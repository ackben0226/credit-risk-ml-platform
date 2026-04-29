import logging

from src.training.train_model import (
    load_data,
    split_data,
    build_logistic,
    build_xgboost,
    build_lightgbm,
    train_and_log,
    save_artifacts
)

logger = logging.getLogger(__name__)


# =====================================================
# MODEL REGISTRY (EXTENSIBLE DESIGN)
# =====================================================

MODEL_BUILDERS = {
    "logistic_regression": build_logistic,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm
}


# =====================================================
# TRAINING PIPELINE
# =====================================================

def run_training_pipeline():

    logger.info("Starting training pipeline")

    # -------------------
    # DATA
    # -------------------
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    results = {}

    # -------------------
    # TRAIN ALL MODELS
    # -------------------
    for name, builder in MODEL_BUILDERS.items():

        logger.info("Training model: %s", name)

        try:
            model = builder(X_train)

            result = train_and_log(
                model=model,
                name=name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

            results[name] = result

        except Exception as e:
            logger.exception("Failed training %s: %s", name, str(e))
            continue

    if not results:
        raise RuntimeError("No models trained successfully")

    # -------------------
    # MODEL SELECTION (ROBUST)
    # -------------------
    best_name, best_result = max(
        results.items(),
        key=lambda x: x[1]["auc"]
    )

    logger.info(
        "Best model selected: %s | AUC=%.4f",
        best_name,
        best_result["auc"]
    )

    # -------------------
    # ARTIFACT STORAGE
    # -------------------
    save_artifacts(
        best_result,
        feature_columns=list(X_train.columns)
    )

    logger.info("Training pipeline completed successfully")