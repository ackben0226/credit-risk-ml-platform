# =====================================================
# src/automation/decision_engine.py
# ML Lifecycle Decision Engine (Control Plane)
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

PREFECT_DEPLOYMENT_NAME = "credit-risk-ml-pipeline/main"

MIN_AUC = 0.75
MIN_PR_AUC = 0.20
MAX_DRIFTED_FEATURES = 5
MAX_ERROR_RATE = 0.05


# =====================================================
# STATE MODEL
# =====================================================

@dataclass
class DecisionContext:
    run_id: str
    model_version: str
    auc: float
    pr_auc: float
    error_rate: float
    drifted_features: int
    rows_scored: int


# =====================================================
# DECISION ENGINE
# =====================================================

class DecisionEngine:

    def __init__(self):
        self.registry = ModelRegistry("credit-risk-model")

    # -------------------------------------------------
    # STEP 1 — Trigger pipeline (Prefect)
    # -------------------------------------------------

    def run_pipeline(self) -> Dict[str, Any]:
        logger.info("Triggering Prefect deployment")

        result = run_deployment(name=PREFECT_DEPLOYMENT_NAME)

        logger.info("Pipeline execution completed")

        return result

    # -------------------------------------------------
    # STEP 2 — Load model metrics (from run output / MLflow)
    # -------------------------------------------------

    def extract_metrics(self, pipeline_result: Dict[str, Any]) -> DecisionContext:

        # In production this should come from MLflow / monitoring store
        metrics = pipeline_result.get("metrics", {})

        return DecisionContext(
            run_id=pipeline_result.get("run_id", "unknown"),
            model_version=metrics.get("model_version", "latest"),
            auc=float(metrics.get("auc", 0.0)),
            pr_auc=float(metrics.get("pr_auc", 0.0)),
            error_rate=float(metrics.get("error_rate", 0.0)),
            drifted_features=int(metrics.get("drifted_features", 0)),
            rows_scored=int(pipeline_result.get("rows", 0))
        )

    # -------------------------------------------------
    # STEP 3 — Promotion Rules Engine
    # -------------------------------------------------

    def should_promote(self, ctx: DecisionContext) -> tuple[bool, str]:

        if ctx.auc < MIN_AUC:
            return False, f"AUC too low: {ctx.auc:.4f}"

        if ctx.pr_auc < MIN_PR_AUC:
            return False, f"PR-AUC too low: {ctx.pr_auc:.4f}"

        if ctx.error_rate > MAX_ERROR_RATE:
            return False, f"Error rate too high: {ctx.error_rate:.4f}"

        if ctx.drifted_features > MAX_DRIFTED_FEATURES:
            return False, f"Drift too high: {ctx.drifted_features}"

        return True, "all_checks_passed"

    # -------------------------------------------------
    # STEP 4 — Promotion Execution
    # -------------------------------------------------

    def promote(self, ctx: DecisionContext):

        logger.info("Promoting model version=%s", ctx.model_version)

        self.registry.set_alias(
            version=ctx.model_version,
            alias="champion"
        )

        notify(
            Severity.INFO,
            "Model Promoted",
            f"Version {ctx.model_version} became champion",
            run_id=ctx.run_id,
            metadata={
                "auc": ctx.auc,
                "pr_auc": ctx.pr_auc
            }
        )

    # -------------------------------------------------
    # STEP 5 — Shadow Assignment
    # -------------------------------------------------

    def assign_shadow(self, ctx: DecisionContext):

        logger.info("Assigning shadow model version=%s", ctx.model_version)

        self.registry.set_alias(
            version=ctx.model_version,
            alias="shadow"
        )

    # -------------------------------------------------
    # STEP 6 — Full Decision Cycle
    # -------------------------------------------------

    def execute(self) -> Dict[str, Any]:

        # 1. Run pipeline
        pipeline_result = self.run_pipeline()

        # 2. Extract metrics
        ctx = self.extract_metrics(pipeline_result)

        # 3. Always assign shadow first
        self.assign_shadow(ctx)

        # 4. Promotion decision
        promote, reason = self.should_promote(ctx)

        if promote:
            self.promote(ctx)

            decision = "promoted"

        else:
            decision = "rejected"

            notify(
                Severity.WARNING,
                "Model Rejected",
                reason,
                run_id=ctx.run_id,
                metadata=ctx.__dict__
            )

        # 5. Auto rollback check (live safety layer)
        rollback_result = auto_rollback(
            error_rate=ctx.error_rate,
            auc_live=ctx.auc,
            drifted_features=ctx.drifted_features
        )

        return {
            "decision": decision,
            "reason": reason,
            "run_id": ctx.run_id,
            "model_version": ctx.model_version,
            "rollback": rollback_result.__dict__
        }


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":

    engine = DecisionEngine()
    result = engine.execute()

    logger.info("FINAL DECISION: %s", result)