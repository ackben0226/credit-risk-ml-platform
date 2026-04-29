# src/pipelines/pipeline.py

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.data.ingestion import CreditDataPipeline
from src.features.bureau_features import BureauFeatures
from src.data_merge import DataMerger
from src.features.derived_features import DeriveFeatures

# =====================================================
# CONFIG
# =====================================================

PIPELINE_VERSION = "3.0.0"

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_PATH = PROCESSED_DIR / "train_features.parquet"
META_PATH = PROCESSED_DIR / "train_features.meta.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =====================================================
# RESULT MODEL
# =====================================================

@dataclass
class PipelineResult:
    success: bool
    rows: int
    cols: int
    output_path: str
    duration_seconds: float
    pipeline_version: str
    created_at: str


# =====================================================
# TIMER
# =====================================================

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self.end - self.start


# =====================================================
# VALIDATION
# =====================================================

def validate_output(raw_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    if final_df.empty:
        raise ValueError("Final dataset is empty")

    if "TARGET" not in final_df.columns:
        raise ValueError("TARGET column missing after pipeline")

    if final_df.shape[0] != raw_df.shape[0]:
        raise ValueError(
            f"Row mismatch after merge "
            f"(raw={raw_df.shape[0]}, final={final_df.shape[0]})"
        )

    if "SK_ID_CURR" in final_df.columns:
        if not final_df["SK_ID_CURR"].is_unique:
            raise ValueError("Duplicate SK_ID_CURR detected after merge")

    if final_df.shape[1] < 10:
        raise ValueError("Too few output features produced")


# =====================================================
# STATS / LINEAGE
# =====================================================

def build_metadata(
    df: pd.DataFrame,
    duration: float
) -> Dict[str, Any]:

    target_rate = None
    if "TARGET" in df.columns:
        target_rate = float(df["TARGET"].mean())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    return {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "duration_seconds": round(duration, 4),
        "target_rate": target_rate,
        "numeric_feature_count": len(numeric_cols),
        "column_names": df.columns.tolist(),
        "null_rate": float(df.isna().mean().mean())
    }


# =====================================================
# SAFE WRITE
# =====================================================

def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(".tmp.parquet")

    df.to_parquet(tmp_path, index=False)

    tmp_path.replace(path)


def write_metadata(meta: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(".tmp.json")

    tmp_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    tmp_path.replace(path)


# =====================================================
# MAIN PIPELINE
# =====================================================

def run_pipeline() -> Dict[str, Any]:

    logger.info("Starting feature pipeline v%s", PIPELINE_VERSION)

    with Timer() as timer:

        try:
            # -------------------------------------------------
            # LOAD DATA
            # -------------------------------------------------

            logger.info("Loading source datasets")

            data = CreditDataPipeline(DATA_DIR).load_data()

            train_df = data.get_train()
            bureau_df = data.get_bureau()

            logger.info(
                "Loaded train=%s bureau=%s",
                train_df.shape,
                bureau_df.shape
            )

            # -------------------------------------------------
            # BUREAU FEATURES
            # -------------------------------------------------

            logger.info("Building bureau features")

            bureau_features = BureauFeatures(
                bureau_df
            ).build()

            # -------------------------------------------------
            # MERGE
            # -------------------------------------------------

            logger.info("Merging datasets")

            merged = DataMerger(
                train_df,
                bureau_features
            ).merge()

            # -------------------------------------------------
            # DERIVED FEATURES
            # -------------------------------------------------

            logger.info("Generating derived features")

            final_df = DeriveFeatures(
                merged
            ).build()

            # -------------------------------------------------
            # VALIDATE
            # -------------------------------------------------

            validate_output(
                raw_df=train_df,
                final_df=final_df
            )

            # -------------------------------------------------
            # WRITE OUTPUT
            # -------------------------------------------------

            logger.info("Writing parquet artifact")

            atomic_write_parquet(
                final_df,
                OUTPUT_PATH
            )

            metadata = build_metadata(
                final_df,
                timer.elapsed
            )

            write_metadata(
                metadata,
                META_PATH
            )

            result = PipelineResult(
                success=True,
                rows=final_df.shape[0],
                cols=final_df.shape[1],
                output_path=str(OUTPUT_PATH),
                duration_seconds=round(timer.elapsed, 4),
                pipeline_version=PIPELINE_VERSION,
                created_at=datetime.now(timezone.utc).isoformat()
            )

            logger.info(
                "Pipeline complete | rows=%s cols=%s duration=%.2fs",
                result.rows,
                result.cols,
                result.duration_seconds
            )

            return asdict(result)

        except Exception:
            logger.exception("Feature pipeline failed")
            raise


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    run_pipeline()