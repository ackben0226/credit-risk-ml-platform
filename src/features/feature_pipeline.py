import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Production inference feature pipeline.

    Responsibilities:
    - Enforce training-time schema consistency
    - Ensure deterministic column ordering
    - Apply safe numeric coercion
    - Provide observable data quality logging
    """

    def __init__(self, feature_columns, strict: bool = False):
        self.feature_columns = list(feature_columns)
        self.strict = strict

    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, data) -> pd.DataFrame:

        df = self._to_dataframe(data)

        df = self._enforce_schema(df)
        df = self._coerce_types(df)
        df = self._handle_missing(df)

        self._final_sanity_check(df)

        return df

    # =====================================================
    # INPUT NORMALISATION
    # =====================================================

    def _to_dataframe(self, data):

        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, list):
            return pd.DataFrame(data)

        raise TypeError(
            "Unsupported input type. Expected dict, list[dict], or DataFrame"
        )

    # =====================================================
    # SCHEMA ENFORCEMENT
    # =====================================================

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:

        missing = [c for c in self.feature_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in self.feature_columns]

        if missing:
            logger.warning("Missing features injected as NaN: %s", missing)

            if self.strict:
                raise ValueError(f"Missing required features: {missing}")

        if extra:
            logger.warning("Extra features ignored: %s", extra)

        # enforce exact training order
        df = df.reindex(columns=self.feature_columns)

        return df

    # =====================================================
    # TYPE COERCION
    # =====================================================

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in self.feature_columns:

            if col not in df.columns:
                continue

            before_na = df[col].isna().sum()

            df[col] = pd.to_numeric(df[col], errors="coerce")

            new_na = df[col].isna().sum() - before_na

            if new_na > 0:
                logger.warning(
                    "Column %s: %s invalid values coerced to NaN",
                    col, new_na
                )

        return df

    # =====================================================
    # MISSING VALUE STRATEGY
    # =====================================================

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:

        missing_before = df.isna().sum().sum()

        if missing_before > 0:
            logger.warning("Total missing values before imputation: %s", missing_before)

        # safer strategy: explicit imputation
        df = df.fillna(0)

        return df

    # =====================================================
    # FINAL SAFETY CHECK
    # =====================================================

    def _final_sanity_check(self, df: pd.DataFrame):

        if df.isna().any().any():
            bad_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(f"NaNs still present in columns: {bad_cols}")