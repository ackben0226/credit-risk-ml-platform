# pipelines/scheduler.py

import logging
from datetime import datetime
from prefect import flow, task, get_run_logger

from api.batch_score import batch_score
from src.monitoring.monitoring import write_scores
from src.alerts.alerts import alert_on_failure

# ----------------------------
# LOGGING
# ----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================================================
# TASK: BATCH SCORING (IDEMPOTENT)
# ====================================================

@task(
    retries=3,
    retry_delay_seconds=60,
    log_prints=True
)
def run_daily_batch(input_path: str):

    logger = get_run_logger()

    logger.info(f"Starting batch scoring | input={input_path}")

    results = batch_score(input_path)

    if results is None or len(results) == 0:
        raise ValueError("Batch scoring returned empty results")

    logger.info(f"Batch scoring completed | rows={len(results)}")

    return results


# ====================================================
# TASK: PERSIST RESULTS
# ====================================================

@task
def persist_results(df):

    logger = get_run_logger()

    logger.info("Writing results to warehouse")

    write_scores(
        df=df,
        table_name="credit_risk_scores"
    )

    return True


# ====================================================
# FLOW: DAILY SCORING PIPELINE
# ====================================================

@flow(
    name="daily-credit-risk-scoring",
    log_prints=True
)
def daily_scoring(input_path: str = "data/incoming/today.parquet"):

    logger = get_run_logger()

    run_id = datetime.utcnow().isoformat()

    try:
        logger.info(f"[{run_id}] Daily scoring pipeline started")

        # ----------------------------
        # STEP 1: SCORE
        # ----------------------------
        scored_df = run_daily_batch(input_path)

        # ----------------------------
        # STEP 2: STORE RESULTS
        # ----------------------------
        persist_results(scored_df)

        logger.info(f"[{run_id}] Pipeline completed successfully")

        return {
            "run_id": run_id,
            "status": "success",
            "rows": len(scored_df)
        }

    except Exception as exc:

        logger.exception(f"[{run_id}] Pipeline failed")

        alert_on_failure(
            message=str(exc),
            context={
                "run_id": run_id,
                "pipeline": "daily_scoring"
            }
        )

        raise


# ====================================================
# ENTRYPOINT
# ====================================================

if __name__ == "__main__":
    daily_scoring()