import joblib
import json
from functools import lru_cache

from config.paths import MODEL_PATH, META_PATH
from core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model(model_name=None, version=None, stage=None):

    logger.info(
        "Loading model | name=%s version=%s stage=%s",
        model_name, version, stage
    )

    # current fallback: local model
    model = joblib.load(MODEL_PATH)

    metadata = {}

    try:
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
    except Exception:
        logger.warning("Metadata not found")

    model.metadata = metadata

    return model