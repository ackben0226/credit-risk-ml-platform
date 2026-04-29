import pandas as pd
import logging

logger = logging.getLogger(__name__)


# =====================================================
# NOTE:
# This must match model metadata schema in production
# =====================================================

def validate_and_prepare(input_dict: dict, expected_features: list) -> pd.DataFrame:

    df = pd.DataFrame([input_dict])

    # -------------------------------------------------
    # 1. CHECK FOR MISSING FEATURES
    # -------------------------------------------------
    missing = [c for c in expected_features if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # -------------------------------------------------
    # 2. IGNORE EXTRA FEATURES (LOG ONLY)
    # -------------------------------------------------
    extra = [c for c in df.columns if c not in expected_features]

    if extra:
        logger.warning("Extra features ignored: %s", extra)

    # -------------------------------------------------
    # 3. ENFORCE STRICT COLUMN ORDER
    # -------------------------------------------------
    df = df.reindex(columns=expected_features)

    # -------------------------------------------------
    # 4. TYPE VALIDATION (NO SILENT COERCION)
    # -------------------------------------------------
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                raise ValueError(f"Invalid non-numeric value in column: {col}")

    # -------------------------------------------------
    # 5. HANDLE MISSING VALUES EXPLICITLY
    # -------------------------------------------------
    # IMPORTANT: no silent imputation
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        logger.warning("Missing values detected in: %s", null_cols)

    return df