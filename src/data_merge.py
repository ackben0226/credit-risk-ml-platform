from __future__ import annotations

import logging
import pandas as pd


logger = logging.getLogger(__name__)

JOIN_KEY = "SK_ID_CURR"


class DataMerger:
    """
    Safely merges base application data with engineered bureau features.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        bureau_features: pd.DataFrame
    ):
        self.train = train.copy()
        self.bureau_features = bureau_features.copy()

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_inputs(self):

        for name, df in {
            "train": self.train,
            "bureau_features": self.bureau_features
        }.items():

            if df.empty:
                raise ValueError(f"{name} dataframe is empty")

            if JOIN_KEY not in df.columns:
                raise ValueError(
                    f"{JOIN_KEY} missing in {name}"
                )

        # train should be unique customer rows
        if self.train[JOIN_KEY].duplicated().any():
            raise ValueError(
                "Duplicate SK_ID_CURR found in train data"
            )

        # aggregated bureau features should also be unique
        if self.bureau_features[JOIN_KEY].duplicated().any():
            raise ValueError(
                "Duplicate SK_ID_CURR found in bureau_features"
            )

    def _check_column_collisions(self):

        collisions = (
            set(self.train.columns)
            & set(self.bureau_features.columns)
        ) - {JOIN_KEY}

        if collisions:
            logger.warning(
                "Overlapping columns detected: %s",
                sorted(collisions)
            )

    # =====================================================
    # MERGE
    # =====================================================

    def merge(self) -> pd.DataFrame:

        logger.info("Starting merge")

        self._validate_inputs()
        self._check_column_collisions()

        train_rows = len(self.train)

        df = self.train.merge(
            self.bureau_features,
            on=JOIN_KEY,
            how="left",
            validate="one_to_one"
        )

        if len(df) != train_rows:
            raise ValueError(
                "Row count changed after merge "
                f"before={train_rows}, after={len(df)}"
            )

        logger.info(
            "Merge complete | rows=%s cols=%s",
            df.shape[0],
            df.shape[1]
        )

        return df