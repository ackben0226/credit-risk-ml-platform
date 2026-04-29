# =====================================================
# src/alerts/alerts.py
# Production-grade alerting + deduplication + routing
# =====================================================

from __future__ import annotations

import os
import json
import time
import logging
import hashlib
import smtplib
import requests
from enum import Enum
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# =====================================================
# CONFIG
# =====================================================

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

STATE_DIR = "runtime"
ALERT_STATE_FILE = f"{STATE_DIR}/alert_state.json"

os.makedirs(STATE_DIR, exist_ok=True)

# =====================================================
# SEVERITY
# =====================================================

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# =====================================================
# ALERT MODEL
# =====================================================

@dataclass
class AlertEvent:
    severity: Severity
    title: str
    message: str
    fingerprint: str
    timestamp: str
    metadata: Dict[str, Any]


# =====================================================
# STATE (DEDUP / COOLDOWN)
# =====================================================

def _load_state() -> dict:
    try:
        if os.path.exists(ALERT_STATE_FILE):
            with open(ALERT_STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_state(state: dict):
    with open(ALERT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _now() -> float:
    return time.time()


def _make_fingerprint(title: str, message: str, severity: str) -> str:
    raw = f"{title}|{message}|{severity}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _is_duplicate(fingerprint: str, cooldown_sec: int = 900) -> bool:
    state = _load_state()

    last = state.get(fingerprint)

    if not last:
        return False

    return (_now() - last) < cooldown_sec


def _update_state(fingerprint: str):
    state = _load_state()
    state[fingerprint] = _now()
    _save_state(state)


# =====================================================
# SLACK
# =====================================================

def _send_slack(payload: str):

    if not SLACK_WEBHOOK_URL:
        return

    try:
        r = requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": payload},
            timeout=5
        )
        if r.status_code >= 300:
            logger.warning("Slack failed status=%s", r.status_code)

    except Exception as e:
        logger.warning("Slack error: %s", str(e))


# =====================================================
# EMAIL
# =====================================================

def _send_email(subject: str, body: str):

    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, ALERT_EMAIL_FROM]):
        return

    recipients = [
        r.strip()
        for r in ALERT_EMAIL_TO.split(",")
        if r.strip()
    ]

    if not recipients:
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, recipients, msg.as_string())

    except Exception as e:
        logger.warning("Email error: %s", str(e))


# =====================================================
# CORE DISPATCHER
# =====================================================

def notify(
    severity: Severity,
    title: str,
    message: str,
    *,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):

    fingerprint = _make_fingerprint(title, message, severity.value)

    if _is_duplicate(fingerprint):
        logger.info("Duplicate alert suppressed | %s", title)
        return

    _update_state(fingerprint)

    emoji = {
        Severity.INFO: "ℹ️",
        Severity.WARNING: "⚠️",
        Severity.CRITICAL: "🚨"
    }[severity]

    enriched = {
        "severity": severity.value,
        "title": title,
        "message": message,
        "run_id": run_id,
        "metadata": metadata or {}
    }

    slack_payload = (
        f"{emoji} {title}\n"
        f"{message}\n"
        f"{json.dumps(enriched, indent=2)}"
    )

    # Slack always
    _send_slack(slack_payload)

    # Email only for elevated severity
    if severity in (Severity.WARNING, Severity.CRITICAL):
        _send_email(
            subject=f"[{severity.value.upper()}] {title}",
            body=json.dumps(enriched, indent=2)
        )

    logger.info("Alert sent | %s | %s", severity.value, title)


# =====================================================
# DOMAIN ALERTS
# =====================================================

def alert_failure(run_id: str, error: str):

    notify(
        Severity.CRITICAL,
        "Pipeline Failure",
        error,
        run_id=run_id,
        metadata={"type": "pipeline_failure"}
    )


def alert_drift(run_id: str, drift_count: int, features: Optional[Dict[str, Any]] = None):

    if drift_count <= 0:
        return

    notify(
        Severity.WARNING,
        "Data Drift Detected",
        f"{drift_count} features drifting",
        run_id=run_id,
        metadata={
            "type": "drift",
            "drift_count": drift_count,
            "features": features or {}
        }
    )


def alert_null_rate(run_id: str, null_rate: float):

    if null_rate < 0.40:
        return

    notify(
        Severity.WARNING,
        "High Null Rate",
        f"Null rate = {null_rate:.2%}",
        run_id=run_id,
        metadata={
            "type": "data_quality",
            "null_rate": null_rate
        }
    )


def alert_success(run_id: str, rows: int):

    notify(
        Severity.INFO,
        "Batch Completed",
        f"Rows processed = {rows}",
        run_id=run_id,
        metadata={
            "type": "success",
            "rows": rows
        }
    )