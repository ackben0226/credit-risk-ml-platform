import os
import pandas as pd
from datetime import datetime

PRED_PATH = "data/predictions/predictions.parquet"


def init_store():
    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)

    if not os.path.exists(PRED_PATH):
        df = pd.DataFrame(columns=[
            "timestamp",
            "probability",
            "prediction",
            "model_name",
            "model_version",
            "AMT_CREDIT",
            "AMT_INCOME_TOTAL"
        ])
        df.to_parquet(PRED_PATH, index=False)


def log_prediction(record: dict):
    """
    Append a prediction event safely
    """

    if os.path.exists(PRED_PATH):
        df = pd.read_parquet(PRED_PATH)
    else:
        df = pd.DataFrame()

    record["timestamp"] = record.get(
        "timestamp",
        datetime.utcnow().isoformat()
    )

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    df.to_parquet(PRED_PATH, index=False)