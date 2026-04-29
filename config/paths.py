from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODEL_DIR = BASE_DIR / "models"

# derived paths (ADD THIS)
MODEL_PATH = MODEL_DIR / "best_model.pkl"
META_PATH = MODEL_DIR / "metadata.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.pkl"