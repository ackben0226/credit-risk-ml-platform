# src/warehouse/warehouse.py

import os
import time
import logging
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# =====================================================
# CONFIG
# =====================================================

REQUIRED_COLUMNS = {
    "SK_ID_CURR",
    "probability",
    "prediction"
}

DEFAULT_TABLE = "credit_risk_scores"
FAILED_TABLE = "credit_risk_scores_failed"


# =====================================================
# CONNECTION
# =====================================================

def get_engine():
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError("DATABASE_URL not configured")

    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=3600
    )


# =====================================================
# VALIDATION
# =====================================================

def validate_schema(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Empty dataframe cannot be written")

    if not df["probability"].between(0, 1).all():
        raise ValueError("Probability values must be in [0,1]")

    if not set(df["prediction"].unique()).issubset({0, 1}):
        raise ValueError("Prediction must be binary (0/1)")


# =====================================================
# PREPROCESS SANITIZATION
# =====================================================

def sanitize(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.drop_duplicates()

    df["probability"] = df["probability"].clip(0, 1)
    df["prediction"] = df["prediction"].astype(int)

    df["written_at"] = pd.Timestamp.utcnow()

    return df


# =====================================================
# DEAD LETTER LOGGING
# =====================================================

def write_failed_batch(df: pd.DataFrame, engine):
    try:
        df.to_sql(
            FAILED_TABLE,
            engine,
            if_exists="append",
            index=False,
            method="multi"
        )
        logger.warning("Failed batch stored in dead-letter table")

    except Exception as e:
        logger.error("Failed to write dead-letter batch: %s", str(e))


# =====================================================
# MAIN WRITER (RETRY + SAFE WRITE)
# =====================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _write_batch(df: pd.DataFrame, engine, table_name: str):

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000
    )


# =====================================================
# PUBLIC API
# =====================================================

def write_scores(
    df: pd.DataFrame,
    table_name: str = DEFAULT_TABLE,
    request_id: str | None = None
):
    """
    Production-grade warehouse writer:
    - schema validation
    - sanitization
    - retry-safe writes
    - observability
    """

    start = time.perf_counter()
    engine = get_engine()

    try:
        # -------------------------
        # VALIDATION
        # -------------------------
        validate_schema(df)

        # -------------------------
        # SANITIZE
        # -------------------------
        df = sanitize(df)

        if request_id:
            df["request_id"] = request_id

        # -------------------------
        # WRITE
        # -------------------------
        _write_batch(df, engine, table_name)

        duration = time.perf_counter() - start

        logger.info(
            "WAREHOUSE_WRITE_SUCCESS | rows=%s table=%s latency_ms=%.2f",
            len(df),
            table_name,
            duration * 1000
        )

        return True

    except (SQLAlchemyError, ValueError) as e:

        logger.error(
            "WAREHOUSE_WRITE_FAILED | table=%s error=%s",
            table_name,
            str(e)
        )

        # store for debugging / replay
        write_failed_batch(df, engine)

        raise

    except Exception as e:

        logger.exception("Unexpected warehouse failure: %s", str(e))
        write_failed_batch(df, engine)

        raise