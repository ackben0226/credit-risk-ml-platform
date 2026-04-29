import os
import tempfile
import pandas as pd
from unittest.mock import patch

from src.training.train_model import run_training


# -----------------------------------------------------
# FIXTURE VALIDATION HELPERS
# -----------------------------------------------------

def _validate_artifacts(model_dir: str):

    model_path = os.path.join(model_dir, "best_model.pkl")
    schema_path = os.path.join(model_dir, "feature_schema.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")

    assert os.path.exists(model_path), "Model artifact missing"
    assert os.path.exists(schema_path), "Schema artifact missing"
    assert os.path.exists(metadata_path), "Metadata artifact missing"

    return model_path, schema_path, metadata_path


def _validate_result(result: dict):

    required_keys = {"auc", "model_name"}

    assert isinstance(result, dict), "Result must be a dict"
    assert required_keys.issubset(result.keys()), "Missing training metrics"

    assert isinstance(result["auc"], float), "AUC must be float"
    assert 0.0 <= result["auc"] <= 1.0, "AUC out of valid range"


# -----------------------------------------------------
# SMOKE TEST
# -----------------------------------------------------

def test_training_pipeline_smoke():

    with tempfile.TemporaryDirectory() as tmpdir:

        # isolate environment completely
        with patch("src.training.train_model.MODEL_DIR", tmpdir), \
             patch("src.training.train_model.DATA_PATH", "tests/fixtures/sample_train.parquet"), \
             patch("src.training.train_model.mlflow.set_experiment"), \
             patch("src.training.train_model.mlflow.start_run"), \
             patch("src.training.train_model.mlflow.log_metric"), \
             patch("src.training.train_model.mlflow.sklearn.log_model"):

            result = run_training(
                data_path="tests/fixtures/sample_train.parquet",
                model_dir=tmpdir,
                mlflow_enabled=False
            )

            # ----------------------------
            # artifact validation
            # ----------------------------
            model_path, schema_path, metadata_path = _validate_artifacts(tmpdir)

            # ----------------------------
            # output validation
            # ----------------------------
            _validate_result(result)

            # ----------------------------
            # sanity checks on model file size
            # ----------------------------
            assert os.path.getsize(model_path) > 0, "Model file is empty"
            assert os.path.getsize(schema_path) > 0, "Schema file is empty"
            assert os.path.getsize(metadata_path) > 0, "Metadata file is empty"