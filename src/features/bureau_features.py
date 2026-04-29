import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BureauFeatures:
    """
    Production-safe bureau feature engineering module.

    Converts raw bureau credit history into stable,
    model-ready aggregated features grouped by SK_ID_CURR.
    """

    def __init__(self, bureau: pd.DataFrame):
        self.bureau = bureau.copy()

        if "SK_ID_CURR" not in self.bureau.columns:
            raise ValueError("SK_ID_CURR is required for bureau aggregation")

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    def build(self) -> pd.DataFrame:

        logger.info("Starting bureau feature engineering")

        if self.bureau.empty:
            logger.warning("Empty bureau dataset received")
            return pd.DataFrame(columns=["SK_ID_CURR"])

        # =================================================
        # SAFE AGGREGATION MAP
        # =================================================
        agg_map = {
            "DAYS_CREDIT": ["min", "max", "mean", "std"],
            "CREDIT_DAY_OVERDUE": ["max", "mean"],
            "AMT_CREDIT_SUM": ["sum", "mean", "max"],
            "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
            "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
            "CNT_CREDIT_PROLONG": ["sum", "max"],
        }

        # keep only available columns (prevents runtime failure)
        agg_map = {
            col: funcs
            for col, funcs in agg_map.items()
            if col in self.bureau.columns
        }

        if not agg_map:
            raise ValueError("No valid bureau columns found for aggregation")

        # =================================================
        # AGGREGATION
        # =================================================
        grouped = self.bureau.groupby("SK_ID_CURR", as_index=False).agg(agg_map)

        # =================================================
        # FLATTEN COLUMN STRUCTURE (CRITICAL FOR ML STABILITY)
        # =================================================
        grouped.columns = self._flatten_columns(grouped.columns)

        logger.info("Bureau feature shape: %s", grouped.shape)

        return grouped

    # =====================================================
    # COLUMN NORMALISATION
    # =====================================================

    @staticmethod
    def _flatten_columns(columns) -> list:

        flat_cols = []

        for col in columns:
            if isinstance(col, tuple):
                base, stat = col
                if stat:
                    flat_cols.append(f"{base.lower()}_{stat.lower()}")
                else:
                    flat_cols.append(base.lower())
            else:
                flat_cols.append(col.lower())

        return flat_cols