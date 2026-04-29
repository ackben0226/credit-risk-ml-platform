# =====================================================
# src/automation/controller.py
# ML Lifecycle Control Plane (Decision Brain)
# =====================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from prefect.deployments import run_deployment

from src.training.registry import ModelRegistry
from src.automation.rollback import auto_rollback
from src.alerts.alerts import notify, Severity

logger = logging.getLogger(__name__)


# =====================================================
# CONFIG
# =====================================================

PIPELINE_DEPLOYMENT = "credit-risk-ml-pipeline/main"

MIN_AUC = 0.75
MIN_PR_AUC = 0.20
MAX_DRIFT = 5
MAX_ERROR_RATE = 0.05


# =====================================================
# STATE
# =====================================================

@dataclass
class ModelState:
    run_id: str
    version: str
    auc: float
    pr_auc: float
    error_rate: float
    drifted_features: int
    rows: int


# =====================================================
# CONTROLLER (CORE BRAIN)
# =====================================================

class MLController:

    def __init__(self):
        self.registry = ModelRegistry("credit-risk-model")

    # -------------------------------------------------
    # STEP 1: Trigger Prefect pipeline
    # -------------------------------------------------

    def run_pipeline(self) -> Dict[str, Any]:
        logger.info("Triggering Prefect pipeline")

        result = run_deployment(name=PIPELINE_DEPLOYMENT)

        return result

    # -------------------------------------------------
    # STEP 2: Extract metrics
    # -------------------------------------------------

    def build_state(self, result: Dict[str, Any]) -> ModelState:

        metrics = result.get("metrics", {})

        return ModelState(
            run_id=result.get("run_id", "unknown"),
            version=metrics.get("model_version", "latest"),
            auc=float(metrics.get("auc", 0.0)),
            pr_auc=float(metrics.get("pr_auc", 0.0)),
            error_rate=float(metrics.get("error_rate", 0.0)),
            drifted_features=int(metrics.get("drifted_features", 0)),
            rows=int(result.get("rows", 0))
        )

    # -------------------------------------------------
    # STEP 3: Decision rules engine
    # -------------------------------------------------

    def should_promote(self, state: ModelState) -> tuple[bool, str]:

        if state.auc < MIN_AUC:
            return False, "AUC below threshold"

        if state.pr_auc < MIN_PR_AUC:
            return False, "PR-AUC below threshold"

        if state.error_rate > MAX_ERROR_RATE:
            return False, "Error rate too high"

        if state.drifted_features > MAX_DRIFT:
            return False, "Drift too high"

        return True, "all_checks_passed"

    # -------------------------------------------------
    # STEP 4: Registry operations
    # -------------------------------------------------

    def promote(self, state: ModelState):

        self.registry.set_alias(
            version=state.version,
            alias="champion"
        )

        notify(
            Severity.INFO,
            "Model Promoted",
            f"Version {state.version} is now CHAMPION",
            run_id=state.run_id
        )

    def assign_shadow(self, state: ModelState):

        self.registry.set_alias(
            version=state.version,
            alias="shadow"
        )

    # -------------------------------------------------
    # STEP 5: Decision execution
    # -------------------------------------------------

    def execute(self) -> Dict[str, Any]:

        logger.info("===== ML CONTROL PLANE STARTED =====")

        # 1. Run pipeline
        pipeline_result = self.run_pipeline()

        # 2. Build state
        state = self.build_state(pipeline_result)

        # 3. Shadow always updated first
        self.assign_shadow(state)

        # 4. Promotion decision
        promote, reason = self.should_promote(state)

        decision = "rejected"

        if promote:
            self.promote(state)
            decision = "promoted"
        else:
            notify(
                Severity.WARNING,
                "Model Rejected",
                reason,
                run_id=state.run_id,
                metadata=state.__dict__
            )

        # 5. Safety rollback layer
        rollback = auto_rollback(
            error_rate=state.error_rate,
            auc_live=state.auc,
            drifted_features=state.drifted_features
        )

        logger.info("===== CONTROL PLANE FINISHED =====")

        return {
            "decision": decision,
            "reason": reason,
            "run_id": state.run_id,
            "version": state.version,
            "rollback": rollback.__dict__
        }


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":

    controller = MLController()
    result = controller.execute()

    logger.info("FINAL RESULT: %s", result)