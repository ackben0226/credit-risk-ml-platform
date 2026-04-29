import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    Production-grade feature validation layer.

    Responsibilities:
    - Enforce schema correctness
    - Detect data quality issues
    - Fail fast on invalid data contracts
    - Provide observable diagnostics
    """

    def __init__(self, expected_features: list[str], strict: bool = False):
        self.expected_features = list(expected_features)
        self.strict = strict

    # =====================================================
    # ENTRY POINT
    # =====================================================

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Starting feature validation")

        self._check_schema(df)
        self._check_data_types(df)
        self._check_missingness(df)

        logger.info("Feature validation passed")

        return True

    # =====================================================
    # SCHEMA VALIDATION
    # =====================================================

    def _check_schema(self, df: pd.DataFrame):

        missing = [c for c in self.expected_features if c not in df.columns]
        extra = [c for c in df.columns if c not in self.expected_features]

        if missing:
            msg = f"Missing required features: {missing}"
            logger.error(msg)

            if self.strict:
                raise ValueError(msg)

        if extra:
            logger.warning("Unexpected features detected: %s", extra)

    # =====================================================
    # TYPE VALIDATION
    # =====================================================

    def _check_data_types(self, df: pd.DataFrame):

        for col in self.expected_features:

            if col not in df.columns:
                continue

            if df[col].dtype == "object":

                sample = df[col].dropna().unique()[:5]

                logger.warning(
                    "Non-numeric feature detected: %s | sample values: %s",
                    col, sample
                )

                # attempt safe coercion (diagnostic only)
                coerced = pd.to_numeric(df[col], errors="coerce")

                invalid_ratio = coerced.isna().mean()

                if invalid_ratio > 0.2:
                    logger.error(
                        "High invalid conversion rate in %s: %.2f%%",
                        col,
                        invalid_ratio * 100
                    )

                    if self.strict:
                        raise ValueError(f"Unreliable numeric conversion: {col}")

    # =====================================================
    # MISSINGNESS CHECK
    # =====================================================

    def _check_missingness(self, df: pd.DataFrame):

        missing_ratio = df[self.expected_features].isnull().mean()

        bad_cols = missing_ratio[missing_ratio > 0.6].index.tolist()

        if bad_cols:
            msg = f"High missingness (>60%) in: {bad_cols}"
            logger.warning(msg)

            if self.strict:
                raise ValueError(msg)