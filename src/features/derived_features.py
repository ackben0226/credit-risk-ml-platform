import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DeriveFeatures:
    """
    Feature derivation layer for credit risk modelling.

    Responsibility:
    - Create stable, model-ready engineered features
    - Avoid leakage and division instability
    - Handle missing/edge-case safe transformations
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def build(self) -> pd.DataFrame:

        logger.info("Starting derived feature engineering")

        self._fix_known_anomalies()

        self._ratios()
        self._time_features()
        self._household_features()
        self._external_score_features()
        self._bureau_features()

        logger.info("Derived features created: %s columns", self.df.shape[1])

        return self.df

    # =====================================================
    # CLEANING
    # =====================================================

    def _fix_known_anomalies(self):

        if "DAYS_EMPLOYED" in self.df.columns:
            self.df["DAYS_EMPLOYED"] = self.df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # =====================================================
    # RATIO FEATURES (CORE CREDIT SIGNALS)
    # =====================================================

    def _ratios(self):

        self.df["credit_income_ratio"] = self._safe_div(
            self.df.get("AMT_CREDIT"),
            self.df.get("AMT_INCOME_TOTAL")
        )

        self.df["annuity_income_ratio"] = self._safe_div(
            self.df.get("AMT_ANNUITY"),
            self.df.get("AMT_INCOME_TOTAL")
        )

        self.df["annuity_credit_ratio"] = self._safe_div(
            self.df.get("AMT_ANNUITY"),
            self.df.get("AMT_CREDIT")
        )

        if "AMT_GOODS_PRICE" in self.df.columns:
            self.df["goods_credit_ratio"] = self._safe_div(
                self.df.get("AMT_GOODS_PRICE"),
                self.df.get("AMT_CREDIT")
            )

    # =====================================================
    # TIME-BASED FEATURES
    # =====================================================

    def _time_features(self):

        if "DAYS_BIRTH" in self.df.columns:
            self.df["age_years"] = -self.df["DAYS_BIRTH"] / 365

        if "DAYS_EMPLOYED" in self.df.columns:
            self.df["employment_years"] = -self.df["DAYS_EMPLOYED"] / 365

        if {"employment_years", "age_years"}.issubset(self.df.columns):
            self.df["employment_age_ratio"] = self._safe_div(
                self.df["employment_years"],
                self.df["age_years"]
            )

    # =====================================================
    # HOUSEHOLD FEATURES
    # =====================================================

    def _household_features(self):

        if "CNT_FAM_MEMBERS" in self.df.columns:

            self.df["income_per_family_member"] = self._safe_div(
                self.df.get("AMT_INCOME_TOTAL"),
                self.df.get("CNT_FAM_MEMBERS")
            )

        if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(self.df.columns):

            self.df["children_ratio"] = self._safe_div(
                self.df["CNT_CHILDREN"],
                self.df["CNT_FAM_MEMBERS"]
            )

    # =====================================================
    # EXTERNAL SCORES (STRONG CREDIT SIGNAL)
    # =====================================================

    def _external_score_features(self):

        cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        available = [c for c in cols if c in self.df.columns]

        if len(available) == 0:
            return

        self.df["ext_sources_mean"] = self.df[available].mean(axis=1)
        self.df["ext_sources_min"] = self.df[available].min(axis=1)
        self.df["ext_sources_max"] = self.df[available].max(axis=1)

    # =====================================================
    # BUREAU FEATURES (SAFE JOIN FEATURES)
    # =====================================================

    def _bureau_features(self):

        if {
            "AMT_CREDIT_SUM_DEBT_sum",
            "AMT_CREDIT_SUM_sum"
        }.issubset(self.df.columns):

            self.df["bureau_debt_credit_ratio"] = self._safe_div(
                self.df["AMT_CREDIT_SUM_DEBT_sum"],
                self.df["AMT_CREDIT_SUM_sum"]
            )

    # =====================================================
    # SAFE DIVISION (CRITICAL FOR PRODUCTION)
    # =====================================================

    @staticmethod
    def _safe_div(numerator, denominator):

        if numerator is None or denominator is None:
            return np.nan

        return np.where(
            denominator == 0,
            np.nan,
            numerator / denominator
        )