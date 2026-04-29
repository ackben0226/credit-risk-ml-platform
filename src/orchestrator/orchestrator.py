# src/automation/orchestrator.py

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from prefect.client.orchestration import get_client

from src.training.registry import (
    check_registry_health,
    register_model,
    promote_if_valid,
    set_challenger,
)
from src.automation.rollback import rollback_to_previous_champion


# =====================================================
# CONFIG
# =====================================================

DEPLOYMENT_NAME = "credit-risk-ml-pipeline"
MIN_AUC = 0.75
MIN_PR_AUC = 0.20

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =====================================================
# RESULT MODEL
# =====================================================

@dataclass
class StageResult:
    success: bool
    stage: str
    message: str
    metadata: Optional[Dict[str, Any]] = None


# =====================================================
# HELPERS
# =====================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def result(stage, success, message, metadata=None):
    return StageResult(
        success=success,
        stage=stage,
        message=message,
        metadata=metadata or {}
    )


# =====================================================
# PREFECT EXECUTION
# =====================================================

async def run_prefect_deployment() -> Dict[str, Any]:
    """
    Trigger Prefect deployment and wait for completion.
    """

    async with get_client() as client:

        deployments = await client.read_deployments()

        deployment = next(
            d for d in deployments
            if d.name == DEPLOYMENT_NAME
        )

        flow_run = await client.create_flow_run_from_deployment(
            deployment.id
        )

        logger.info("Triggered Prefect run_id=%s", flow_run.id)

        while True:
            run = await client.read_flow_run(flow_run.id)

            if run.state.is_completed():
                logger.info("Prefect pipeline completed")
                return {
                    "success": True,
                    "flow_run_id": str(flow_run.id)
                }

            if run.state.is_failed():
                return {
                    "success": False,
                    "error": str(run.state.message)
                }


# =====================================================
# TRAINING RESULTS READER
# =====================================================

def read_training_results() -> Dict[str, float]:
    """
    Replace with MLflow metrics fetch.
    """

    # Example placeholder
    return {
        "auc": 0.782,
        "pr_auc": 0.241
    }


# =====================================================
# MAIN DECISION ENGINE
# =====================================================

async def run_orchestrator():

    summary = {
        "started_at": utc_now(),
        "success": False,
        "stages": []
    }

    # -------------------------------------------------
    # Registry health
    # -------------------------------------------------

    if not check_registry_health():
        summary["stages"].append(
            asdict(result(
                "healthcheck",
                False,
                "Registry unavailable"
            ))
        )
        return summary

    # -------------------------------------------------
    # Trigger Prefect Pipeline
    # -------------------------------------------------

    logger.info("Launching Prefect deployment")

    prefect_result = await run_prefect_deployment()

    if not prefect_result["success"]:
        summary["stages"].append(
            asdict(result(
                "pipeline",
                False,
                prefect_result["error"]
            ))
        )
        return summary

    summary["stages"].append(
        asdict(result(
            "pipeline",
            True,
            "Prefect pipeline completed",
            prefect_result
        ))
    )

    # -------------------------------------------------
    # Read metrics
    # -------------------------------------------------

    metrics = read_training_results()

    auc = metrics["auc"]
    pr_auc = metrics["pr_auc"]

    # -------------------------------------------------
    # Register model
    # -------------------------------------------------

    version = register_model(
        run_id=prefect_result["flow_run_id"]
    )

    summary["stages"].append(
        asdict(result(
            "registry",
            True,
            f"Registered version {version}",
            {"version": version}
        ))
    )

    # -------------------------------------------------
    # Challenger first
    # -------------------------------------------------

    set_challenger(version)

    # -------------------------------------------------
    # Promotion Gate
    # -------------------------------------------------

    try:
        promote_if_valid(
            version=version,
            auc=auc,
            pr_auc=pr_auc,
            min_auc=MIN_AUC,
            min_pr_auc=MIN_PR_AUC
        )

        summary["stages"].append(
            asdict(result(
                "promotion",
                True,
                "Promoted to champion",
                {
                    "version": version,
                    "auc": auc,
                    "pr_auc": pr_auc
                }
            ))
        )

        summary["success"] = True

    except Exception as exc:

        logger.warning("Promotion failed: %s", str(exc))

        rollback_to_previous_champion()

        summary["stages"].append(
            asdict(result(
                "rollback",
                True,
                "Rollback executed"
            ))
        )

    summary["finished_at"] = utc_now()

    return summary


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    import asyncio

    output = asyncio.run(run_orchestrator())

    logger.info(output)