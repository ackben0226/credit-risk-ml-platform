from __future__ import annotations

import logging
import time
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Production-grade MLflow registry service.
    """

    ALIAS_CHAMPION = "champion"
    ALIAS_CHALLENGER = "challenger"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = MlflowClient()

    # =====================================================
    # HEALTH
    # =====================================================

    def is_healthy(self) -> bool:
        try:
            self.client.search_registered_models(max_results=1)
            return True
        except Exception as exc:
            logger.exception("Registry unavailable: %s", exc)
            return False

    # =====================================================
    # REGISTRATION
    # =====================================================

    def register(self, run_id: str, artifact_path: str = "model") -> str:
        model_uri = f"runs:/{run_id}/{artifact_path}"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name
        )

        version = str(result.version)
        self._wait_until_ready(version)

        logger.info("Registered %s v%s", self.model_name, version)
        return version

    # =====================================================
    # ALIASES
    # =====================================================

    def set_alias(self, version: str, alias: str):
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=alias,
            version=version
        )

        logger.info("Alias set | %s -> v%s", alias, version)

    def get_model_uri(self, alias: str) -> str:
        return f"models:/{self.model_name}@{alias}"

    # =====================================================
    # PROMOTION
    # =====================================================

    def promote(self, version: str):
        self.set_alias(version, self.ALIAS_CHAMPION)

        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.info("Promoted v%s -> champion", version)

    def set_challenger(self, version: str):
        self.set_alias(version, self.ALIAS_CHALLENGER)

    # =====================================================
    # GATED PROMOTION
    # =====================================================

    def promote_if_valid(
        self,
        version: str,
        auc: float,
        pr_auc: float,
        min_auc: float = 0.75,
        min_pr_auc: float = 0.20
    ):

        if auc < min_auc:
            raise ValueError("AUC gate failed")

        if pr_auc < min_pr_auc:
            raise ValueError("PR-AUC gate failed")

        self.promote(version)

        logger.info(
            "Model v%s promoted (AUC=%.4f PR_AUC=%.4f)",
            version, auc, pr_auc
        )

    # =====================================================
    # INTERNAL UTILS
    # =====================================================

    def _wait_until_ready(self, version: str, timeout: int = 60):

        start = time.time()

        while time.time() - start < timeout:

            mv = self.client.get_model_version(
                name=self.model_name,
                version=version
            )

            if mv.status == "READY":
                return

            if mv.status == "FAILED_REGISTRATION":
                raise RuntimeError(f"Registration failed v{version}")

            time.sleep(2)

        raise TimeoutError(f"Timeout waiting for v{version}")