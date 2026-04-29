# src/pipelines/prefect_pipeline.py

from __future__ import annotations

import os
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from prefect import flow, task, get_run_logger

from src.data.ingestion import CreditDataPipeline
from src.data.cleaning import CleanData
from src.features.bureau_features import BureauFeatures
from src.data_merge import DataMerger
from src.features.derived_features import DeriveFeatures

from src.training.train_model import main as train_model_pipeline
from src.training.registry import register_model


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = PROCESSED_DIR / "metadata"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# =====================================================
# HELPERS
# =====================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return str(uuid.uuid4())


def parquet_path(run_id: str, name: str) -> Path:
    return PROCESSED_DIR / f"{name}_{run_id}.parquet"


def meta_path(run_id: str) -> Path:
    return META_DIR / f"{run_id}.json"


# =====================================================
# TASKS
# =====================================================

@task(retries=2, retry_delay_seconds=10)
def extract_raw_data(data_dir: str) -> dict:
    """
    Load raw source datasets.
    """
    log = get_run_logger()

    data = CreditDataPipeline(data_dir).load_data()

    log.info(
        "Raw data loaded | train_rows=%s bureau_rows=%s",
        data.get_train().shape[0],
        data.get_bureau().shape[0]
    )

    return {
        "train": data.get_train(),
        "bureau": data.get_bureau()
    }


@task(retries=2, retry_delay_seconds=10)
def clean_datasets(payload: dict, run_id: str) -> dict:
    """
    Clean and persist datasets.
    """
    log = get_run_logger()

    train_df = CleanData(payload["train"]).build()
    bureau_df = CleanData(payload["bureau"]).build()

    train_path = parquet_path(run_id, "train_clean")
    bureau_path = parquet_path(run_id, "bureau_clean")

    train_df.to_parquet(train_path, index=False)
    bureau_df.to_parquet(bureau_path, index=False)

    log.info("Clean datasets saved")

    return {
        "train_path": str(train_path),
        "bureau_path": str(bureau_path)
    }


@task
def build_bureau_features_task(paths: dict, run_id: str) -> str:
    """
    Build bureau aggregates.
    """
    log = get_run_logger()

    bureau_df = pd.read_parquet(paths["bureau_path"])

    feat_df = BureauFeatures(bureau_df).build()

    out = parquet_path(run_id, "bureau_features")
    feat_df.to_parquet(out, index=False)

    log.info("Bureau features saved | rows=%s", feat_df.shape[0])

    return str(out)


@task
def merge_features_task(paths: dict, bureau_feat_path: str, run_id: str) -> str:
    """
    Merge train + bureau features.
    """
    log = get_run_logger()

    train_df = pd.read_parquet(paths["train_path"])
    bureau_feat = pd.read_parquet(bureau_feat_path)

    merged = DataMerger(train_df, bureau_feat).merge()

    out = parquet_path(run_id, "merged")
    merged.to_parquet(out, index=False)

    log.info("Merged dataset saved | rows=%s", merged.shape[0])

    return str(out)


@task
def derive_features_task(merged_path: str, run_id: str) -> str:
    """
    Build final model features.
    """
    log = get_run_logger()

    df = pd.read_parquet(merged_path)

    final_df = DeriveFeatures(df).build()

    out = parquet_path(run_id, "train_features")

    final_df.to_parquet(out, index=False)

    log.info(
        "Final feature set saved | rows=%s cols=%s",
        final_df.shape[0],
        final_df.shape[1]
    )

    return str(out)


@task
def validate_dataset_task(dataset_path: str) -> dict:
    """
    Hard validation gate before training.
    """
    log = get_run_logger()

    df = pd.read_parquet(dataset_path)

    if df.empty:
        raise ValueError("Dataset empty")

    if "TARGET" not in df.columns:
        raise ValueError("TARGET missing")

    null_rate = float(df.isna().mean().mean())

    if null_rate > 0.35:
        raise ValueError(f"Null rate too high: {null_rate:.2%}")

    metrics = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "null_rate": null_rate
    }

    log.info("Validation passed | %s", metrics)

    return metrics


@task
def persist_metadata_task(run_id: str, dataset_path: str, stats: dict) -> str:
    """
    Save lineage metadata.
    """
    payload = {
        "run_id": run_id,
        "created_at": utc_now(),
        "dataset_path": dataset_path,
        **stats
    }

    path = meta_path(run_id)

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return str(path)


@task(retries=1)
def train_model_task(dataset_path: str) -> dict:
    """
    Trigger training pipeline using exact dataset artifact.
    Assumes train_model_pipeline(path=...)
    """
    log = get_run_logger()

    log.info("Training started | dataset=%s", dataset_path)

    result = train_model_pipeline(dataset_path=str(dataset_path))

    log.info("Training complete")

    return result


@task
def register_model_task(training_result: dict) -> dict:
    """
    Register trained model to registry.
    Expects training_result contains run_id.
    """
    log = get_run_logger()

    run_id = training_result["run_id"]

    version = register_model(run_id=run_id)

    log.info("Model registered | version=%s", version)

    return {
        "run_id": run_id,
        "version": version
    }


# =====================================================
# FLOW
# =====================================================

@flow(
    name="credit-risk-ml-pipeline",
    log_prints=True
)
def ml_pipeline(
    data_dir: str = str(DATA_DIR),
    run_training: bool = True,
    register_after_train: bool = True
):
    """
    Production-grade orchestration flow.
    """

    log = get_run_logger()

    run_id = make_run_id()

    log.info("Pipeline started | run_id=%s", run_id)

    # -----------------------------------------
    # 1. Extract
    # -----------------------------------------

    raw = extract_raw_data(data_dir)

    # -----------------------------------------
    # 2. Clean
    # -----------------------------------------

    clean_paths = clean_datasets(raw, run_id)

    # -----------------------------------------
    # 3. Parallelizable branch
    # -----------------------------------------

    bureau_feat_path = build_bureau_features_task(
        clean_paths,
        run_id
    )

    # -----------------------------------------
    # 4. Merge
    # -----------------------------------------

    merged_path = merge_features_task(
        clean_paths,
        bureau_feat_path,
        run_id
    )

    # -----------------------------------------
    # 5. Derive final features
    # -----------------------------------------

    dataset_path = derive_features_task(
        merged_path,
        run_id
    )

    # -----------------------------------------
    # 6. Validation Gate
    # -----------------------------------------

    stats = validate_dataset_task(dataset_path)

    # -----------------------------------------
    # 7. Metadata / Lineage
    # -----------------------------------------

    metadata_file = persist_metadata_task(
        run_id,
        dataset_path,
        stats
    )

    log.info("Metadata saved | %s", metadata_file)

    output = {
        "run_id": run_id,
        "dataset_path": dataset_path,
        "metadata_path": metadata_file,
        "stats": stats
    }

    # -----------------------------------------
    # 8. Train + Register
    # -----------------------------------------

    if run_training:

        train_result = train_model_task(dataset_path)

        output["training"] = train_result

        if register_after_train:
            reg = register_model_task(train_result)
            output["registry"] = reg

    log.info("Pipeline completed | run_id=%s", run_id)

    return output


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    ml_pipeline()