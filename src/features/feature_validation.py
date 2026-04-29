# ============================================================
# src/features/feature_validation.py
# ============================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class ValidationError(Exception):
    """Base validation error."""


class SchemaValidationError(ValidationError):
    """Raised when schema contract fails."""


class DataTypeValidationError(ValidationError):
    """Raised when dtype contract fails."""


class MissingnessValidationError(ValidationError):
    """Raised when missingness exceeds threshold."""


class DuplicateColumnError(ValidationError):
    """Raised when duplicate columns are detected."""


# ============================================================
# CONFIG
# ============================================================

@dataclass(slots=True)
class FeatureValidationConfig:
    required_columns: list[str]
    optional_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    nullable_columns: list[str] = field(default_factory=list)

    max_missing_ratio: float = 0.60
    max_invalid_numeric_ratio: float = 0.20

    allow_extra_columns: bool = True
    strict: bool = True


# ============================================================
# VALIDATOR
# ============================================================

class FeatureValidator:
    """
    Production-grade feature contract validator.

    Validates:
    - duplicate columns
    - required schema
    - unexpected columns
    - numeric coercion quality
    - column missingness
    """

    def __init__(self, config: FeatureValidationConfig):
        self.config = config

    # --------------------------------------------------------

    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate dataframe against feature contract.

        Returns original dataframe if valid.
        Raises ValidationError subclasses on failure in strict mode.
        """
        start = time.perf_counter()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")

        logger.info(
            "feature_validation_started rows=%s cols=%s",
            len(df),
            len(df.columns)
        )

        self._check_duplicate_columns(df)
        self._check_schema(df)
        self._check_numeric_contract(df)
        self._check_missingness(df)

        elapsed = round(time.perf_counter() - start, 4)

        logger.info(
            "feature_validation_passed rows=%s cols=%s runtime_sec=%s",
            len(df),
            len(df.columns),
            elapsed
        )

        return df

    # --------------------------------------------------------

    def _raise_or_log(
        self,
        exc_type: type[Exception],
        message: str,
        level: str = "error"
    ) -> None:
        getattr(logger, level)(message)

        if self.config.strict:
            raise exc_type(message)

    # --------------------------------------------------------

    def _check_duplicate_columns(self, df: pd.DataFrame) -> None:
        duplicates = df.columns[df.columns.duplicated()].tolist()

        if duplicates:
            self._raise_or_log(
                DuplicateColumnError,
                f"Duplicate columns detected: {duplicates}"
            )

    # --------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame) -> None:
        required = set(self.config.required_columns)
        optional = set(self.config.optional_columns)
        actual = set(df.columns)

        missing = sorted(required - actual)

        if missing:
            self._raise_or_log(
                SchemaValidationError,
                f"Missing required columns: {missing}"
            )

        if not self.config.allow_extra_columns:
            allowed = required | optional
            extra = sorted(actual - allowed)

            if extra:
                self._raise_or_log(
                    SchemaValidationError,
                    f"Unexpected columns detected: {extra}",
                    level="warning"
                )

    # --------------------------------------------------------

    def _check_numeric_contract(self, df: pd.DataFrame) -> None:
        for col in self.config.numeric_columns:

            if col not in df.columns:
                continue

            s = df[col]

            if pd.api.types.is_numeric_dtype(s):
                continue

            non_null_mask = s.notna()

            if non_null_mask.sum() == 0:
                continue

            coerced = pd.to_numeric(s, errors="coerce")

            invalid_ratio = coerced[non_null_mask].isna().mean()

            if invalid_ratio > self.config.max_invalid_numeric_ratio:
                self._raise_or_log(
                    DataTypeValidationError,
                    (
                        f"Numeric coercion failure in {col}: "
                        f"{invalid_ratio:.2%} invalid"
                    )
                )

    # --------------------------------------------------------

    def _check_missingness(self, df: pd.DataFrame) -> None:
        cols = [
            c for c in self.config.required_columns
            if c in df.columns and c not in self.config.nullable_columns
        ]

        if not cols:
            return

        ratios = df[cols].isna().mean()

        offenders = ratios[
            ratios > self.config.max_missing_ratio
        ].to_dict()

        if offenders:
            self._raise_or_log(
                MissingnessValidationError,
                f"High missingness detected: {offenders}",
                level="warning"
            )