# =====================================================
# src/automation/rollback.py
# Production-grade autonomous rollback controller
# =====================================================

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from mlflow.tracking import MlflowClient

from src.training.registry import (
    MODEL_NAME,
    ALIAS_CHAMPION,
    ALIAS_CHALLENGER
)

from src.alerts.alerts import notify, Severity

logger = logging.getLogger(__name__)

client = MlflowClient()

# =====================================================
# CONFIG
# =====================================================

MAX_ERROR_RATE = 0.05
MIN_AUC = 0.75
MAX_DRIFTED_FEATURES = 5

COOLDOWN_MINUTES = 60

STATE_DIR = "runtime"
STATE_FILE = f"{STATE_DIR}/rollback_state.json"

os.makedirs(STATE_DIR, exist_ok=True)


# =====================================================
# MODELS
# =====================================================

@dataclass
class RollbackResult:
    success: bool
    action: str
    reason: str
    current_version: Optional[str]
    target_version: Optional[str]
    timestamp: str
    metadata: Dict[str, Any]


# =====================================================
# HELPERS
# =====================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_dt():
    return datetime.now(timezone.utc)


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}

    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(data: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def cooldown_active() -> bool:
    state = load_state()

    ts = state.get("last_rollback")

    if not ts:
        return False

    last = datetime.fromisoformat(ts)

    return now_dt() - last < timedelta(minutes=COOLDOWN_MINUTES)


def get_alias_version(alias: str) -> Optional[str]:
    try:
        mv = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=alias
        )
        return str(mv.version)

    except Exception:
        return None


def set_alias(alias: str, version: str):
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version
    )


def verify_alias(alias: str, version: str) -> bool:
    actual = get_alias_version(alias)
    return actual == version


def get_last_stable_version(
    exclude: Optional[str] = None
) -> Optional[str]:

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    versions = sorted(
        versions,
        key=lambda x: int(x.version),
        reverse=True
    )

    for mv in versions:
        version = str(mv.version)

        if version == exclude:
            continue

        tags = getattr(mv, "tags", {}) or {}

        if tags.get("approved") == "true":
            return version

    for mv in versions:
        version = str(mv.version)

        if version != exclude:
            return version

    return None


# =====================================================
# DECISION ENGINE
# =====================================================

def should_rollback(
    *,
    error_rate: float,
    auc_live: Optional[float],
    drifted_features: int
) -> tuple[bool, str]:

    if error_rate > MAX_ERROR_RATE:
        return True, f"error_rate={error_rate:.2%}"

    if auc_live is not None and auc_live < MIN_AUC:
        return True, f"auc_live={auc_live:.4f}"

    if drifted_features > MAX_DRIFTED_FEATURES:
        return True, f"drifted_features={drifted_features}"

    return False, "healthy"


# =====================================================
# EXECUTION
# =====================================================

def execute_rollback(reason: str) -> RollbackResult:

    current = get_alias_version(ALIAS_CHAMPION)

    target = get_last_stable_version(
        exclude=current
    )

    if not target:
        return RollbackResult(
            success=False,
            action="rollback_failed",
            reason="No fallback version found",
            current_version=current,
            target_version=None,
            timestamp=utc_now(),
            metadata={}
        )

    # atomic-ish move
    set_alias(ALIAS_CHAMPION, target)

    if not verify_alias(ALIAS_CHAMPION, target):
        raise RuntimeError("Alias verification failed")

    save_state({
        "last_rollback": utc_now(),
        "from": current,
        "to": target,
        "reason": reason
    })

    notify(
        Severity.CRITICAL,
        "Automatic Rollback Executed",
        f"Champion moved v{current} -> v{target}\nReason: {reason}"
    )

    logger.warning(
        "Rollback executed | %s -> %s",
        current,
        target
    )

    return RollbackResult(
        success=True,
        action="rollback",
        reason=reason,
        current_version=current,
        target_version=target,
        timestamp=utc_now(),
        metadata={}
    )


def auto_rollback(
    *,
    error_rate: float,
    auc_live: Optional[float] = None,
    drifted_features: int = 0
) -> RollbackResult:

    rollback, reason = should_rollback(
        error_rate=error_rate,
        auc_live=auc_live,
        drifted_features=drifted_features
    )

    current = get_alias_version(ALIAS_CHAMPION)

    if not rollback:
        return RollbackResult(
            success=True,
            action="none",
            reason="healthy",
            current_version=current,
            target_version=None,
            timestamp=utc_now(),
            metadata={}
        )

    if cooldown_active():
        notify(
            Severity.WARNING,
            "Rollback Suppressed",
            "Cooldown active"
        )

        return RollbackResult(
            success=False,
            action="suppressed",
            reason="cooldown_active",
            current_version=current,
            target_version=None,
            timestamp=utc_now(),
            metadata={}
        )

    return execute_rollback(reason)