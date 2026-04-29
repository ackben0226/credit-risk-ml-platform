# src/pipelines/scheduler.py

from __future__ import annotations

import logging
from pathlib import Path
from prefect import flow, task

from api.batch_score import batch_score
from src.alerts.alerts import alert_on_failure

# =====================================================
# CONFIG
# =====================================================

INPUT_PATH = Path("data/incoming/today.parquet")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================
# TASK
# =====================================================

@task(
    name="run-batch-score",
    retries=3,
    retry_delay_seconds=60
)
def run_batch(input_path: str) -> dict:
    """
    Execute batch scoring job.
    """

    logger.info("Running batch score | file=%s", input_path)

    result = batch_score(input_path)

    logger.info("Batch scoring complete | result=%s", result)

    return result


# =====================================================
# FLOW
# =====================================================

@flow(
    name="daily-credit-risk-scoring",
    log_prints=True
)
def daily_scoring() -> dict:
    """
    Daily scheduled inference pipeline.
    Intended for Prefect deployment scheduler / cron.
    """

    logger.info("Starting scheduled scoring flow")

    try:
        if not INPUT_PATH.exists():
            raise FileNotFoundError(
                f"Input file not found: {INPUT_PATH}"
            )

        result = run_batch(str(INPUT_PATH))

        logger.info("Scoring flow completed successfully")

        return result

    except Exception as exc:
        logger.exception("Scoring flow failed")

        try:
            alert_on_failure(str(exc))
        except Exception:
            logger.exception("Failed to send alert")

        raise


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    daily_scoring()