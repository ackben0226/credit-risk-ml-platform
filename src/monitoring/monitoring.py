# src/monitoring/monitoring.py

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "credit_risk_scores"

_engine: Optional[Engine] = None


# =====================================================
# ENGINE MANAGEMENT
# =====================================================

def get_database_url() -> str:
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable not configured")

    return db_url


def get_engine() -> Engine:
    """
    Singleton SQLAlchemy engine with pooling.
    """

    global _engine

    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=5,
            max_overflow=10,
            future=True
        )

        logger.info("Database engine initialized")

    return _engine


def dispose_engine() -> None:
    global _engine

    if _engine is not None:
        _engine.dispose()
        _engine = None
        logger.info("Database engine disposed")


# =====================================================
# CONNECTION HELPERS
# =====================================================

@contextmanager
def get_connection():
    engine = get_engine()

    with engine.begin() as conn:
        yield conn


def healthcheck() -> bool:
    """
    Verify warehouse connectivity.
    """

    try:
        with get_connection() as conn:
            conn.execute(text("SELECT 1"))

        return True

    except Exception as exc:
        logger.exception("Warehouse healthcheck failed: %s", str(exc))
        return False


# =====================================================
# DATA VALIDATION
# =====================================================

def validate_scores(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")

    required_cols = {"prediction", "probability"}

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not df["probability"].between(0, 1).all():
        raise ValueError("Probability values must be between 0 and 1")


# =====================================================
# WRITE SCORES
# =====================================================

def write_scores(
    df: pd.DataFrame,
    table_name: str = DEFAULT_TABLE,
    if_exists: str = "append",
    chunksize: int = 1000
) -> int:
    """
    Persist scored predictions into warehouse.

    Returns:
        rows written
    """

    validate_scores(df)

    try:
        engine = get_engine()

        rows = len(df)

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=chunksize
        )

        logger.info(
            "Warehouse write complete | table=%s rows=%s",
            table_name,
            rows
        )

        return rows

    except SQLAlchemyError as exc:
        logger.exception(
            "Warehouse write failed | table=%s error=%s",
            table_name,
            str(exc)
        )
        raise


# =====================================================
# READ HELPERS
# =====================================================

def read_recent_scores(
    table_name: str = DEFAULT_TABLE,
    limit: int = 100
) -> pd.DataFrame:
    """
    Read recent scored rows.
    Assumes timestamp column exists.
    """

    query = text(
        f"""
        SELECT *
        FROM {table_name}
        ORDER BY timestamp DESC
        LIMIT :limit
        """
    )

    try:
        engine = get_engine()

        return pd.read_sql(
            query,
            con=engine,
            params={"limit": limit}
        )

    except Exception as exc:
        logger.exception("Failed reading scores: %s", str(exc))
        raise


# =====================================================
# METRIC AGGREGATES
# =====================================================

def score_summary(
    table_name: str = DEFAULT_TABLE
) -> dict:
    """
    Basic monitoring summary.
    """

    query = text(
        f"""
        SELECT
            COUNT(*) AS row_count,
            AVG(probability) AS avg_probability,
            MIN(probability) AS min_probability,
            MAX(probability) AS max_probability
        FROM {table_name}
        """
    )

    try:
        with get_connection() as conn:
            row = conn.execute(query).mappings().first()

        return dict(row) if row else {}

    except Exception as exc:
        logger.exception("Failed summary query: %s", str(exc))
        raise