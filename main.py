import logging
import os
from datetime import datetime

import pandas as pd

from src.data.ingestion import CreditDataPipeline
from src.data.cleaning import CleanData
from src.features.bureau_features import BureauFeatures
from src.data_merge import DataMerger
from src.features.derived_features import DeriveFeatures

# ----------------------------
# CONFIG
# ----------------------------

DATA_PATH = "data"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOGGING
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ----------------------------
# PIPELINE
# ----------------------------

def main():

    try:
        # ----------------------------
        # 1. LOAD DATA
        # ----------------------------
        logger.info("Loading raw data")
        pipeline = CreditDataPipeline(DATA_PATH).load_data()

        # ----------------------------
        # 2. CLEAN DATA (CRITICAL STEP)
        # ----------------------------
        logger.info("Cleaning train and bureau datasets")

        cleaner = CleanData()

        train_df = cleaner.transform(pipeline.train)
        bureau_df = cleaner.transform(pipeline.bureau)

        # ----------------------------
        # 3. VALIDATION (EARLY FAILURE)
        # ----------------------------
        required_cols = ["SK_ID_CURR"]

        for col in required_cols:
            assert col in train_df.columns, f"{col} missing in train"
            assert col in bureau_df.columns, f"{col} missing in bureau"

        # ----------------------------
        # 4. FEATURE ENGINEERING (BUREAU)
        # ----------------------------
        logger.info("Building bureau features")
        bureau_features = BureauFeatures(bureau_df).build_feature()

        # ----------------------------
        # 5. MERGE DATASETS
        # ----------------------------
        logger.info("Merging datasets")

        merged_df = DataMerger(train_df, bureau_features).merge()

        # ----------------------------
        # 6. DERIVED FEATURES
        # ----------------------------
        logger.info("Engineering derived features")

        final_df = DeriveFeatures(merged_df).build()

        # ----------------------------
        # 7. FINAL VALIDATION (CRITICAL FOR SKLEARN)
        # ----------------------------
        logger.info("Running final validations")

        assert final_df.shape[0] == train_df.shape[0], "Row mismatch after processing"
        assert "TARGET" in final_df.columns, "TARGET missing"

        # Ensure no raw object leakage into model stage
        object_cols = final_df.select_dtypes(include=["object"]).columns.tolist()
        if len(object_cols) > 0:
            logger.warning(f"Object columns detected: {object_cols}")

        # ----------------------------
        # 8. SAVE VERSIONED OUTPUT
        # ----------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = os.path.join(
            OUTPUT_DIR,
            f"train_features_{timestamp}.parquet"
        )

        logger.info(f"Saving dataset to {output_path}")

        final_df.to_parquet(output_path, index=False)

        # ----------------------------
        # 9. SUMMARY
        # ----------------------------
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Bureau features shape: {bureau_features.shape}")
        logger.info(f"Final dataset shape: {final_df.shape}")

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")

        return output_path

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


# ----------------------------
# ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    main()