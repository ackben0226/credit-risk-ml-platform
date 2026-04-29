import pandas as pd
import numpy as np

from src.features.feature_pipeline import FeaturePipeline


# -----------------------------------------------------
# FEATURE CONTRACT (SOURCE OF TRUTH)
# -----------------------------------------------------

FEATURE_COLUMNS = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]


BASE_INPUT = {
    "AMT_CREDIT": 100000,
    "AMT_INCOME_TOTAL": 50000,
    "AMT_ANNUITY": 10000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -3000,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.7,
}


# -----------------------------------------------------
# TEST 1: STRICT SCHEMA CONTRACT
# -----------------------------------------------------

def test_feature_pipeline_schema_contract():

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X = pipeline.transform([BASE_INPUT])

    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[1] == len(FEATURE_COLUMNS)


# -----------------------------------------------------
# TEST 2: MISSING FEATURES ARE ZERO-FILLED (NO NULLS ALLOWED)
# -----------------------------------------------------

def test_feature_pipeline_missing_values_are_safe():

    incomplete = BASE_INPUT.copy()
    del incomplete["EXT_SOURCE_3"]

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X = pipeline.transform([incomplete])

    assert "EXT_SOURCE_3" in X.columns
    assert X["EXT_SOURCE_3"].isna().sum() == 0
    assert (X["EXT_SOURCE_3"] == 0).all()


# -----------------------------------------------------
# TEST 3: EXTRA FEATURES MUST BE DROPPED
# -----------------------------------------------------

def test_feature_pipeline_extra_features_removed():

    noisy = BASE_INPUT.copy()
    noisy.update({
        "UNUSED_FEATURE": 999,
        "TEXT_FIELD": "invalid"
    })

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X = pipeline.transform([noisy])

    assert "UNUSED_FEATURE" not in X.columns
    assert "TEXT_FIELD" not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS


# -----------------------------------------------------
# TEST 4: COLUMN ORDER MUST BE DETERMINISTIC
# -----------------------------------------------------

def test_feature_pipeline_column_order_stable():

    shuffled = dict(reversed(list(BASE_INPUT.items())))

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X = pipeline.transform([shuffled])

    assert list(X.columns) == FEATURE_COLUMNS


# -----------------------------------------------------
# TEST 5: BATCH CONSISTENCY (ROW INTEGRITY)
# -----------------------------------------------------

def test_feature_pipeline_batch_consistency():

    batch = [BASE_INPUT, BASE_INPUT, BASE_INPUT]

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X = pipeline.transform(batch)

    assert X.shape == (3, len(FEATURE_COLUMNS))


# -----------------------------------------------------
# TEST 6: NUMERIC DTYPE ENFORCEMENT
# -----------------------------------------------------

def test_feature_pipeline_numeric_only():

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X = pipeline.transform([BASE_INPUT])

    assert all(np.issubdtype(X[col].dtype, np.number) for col in X.columns)


# -----------------------------------------------------
# TEST 7: NULL SAFETY GUARANTEE (STRICT)
# -----------------------------------------------------

def test_feature_pipeline_no_nulls_allowed():

    broken_input = {k: None for k in FEATURE_COLUMNS}

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X = pipeline.transform([broken_input])

    assert not X.isnull().any().any()
    assert X.shape == (1, len(FEATURE_COLUMNS))


# -----------------------------------------------------
# TEST 8: INPUT ORDER INDEPENDENCE (REAL WORLD EDGE CASE)
# -----------------------------------------------------

def test_feature_pipeline_input_order_independence():

    reversed_input = dict(reversed(list(BASE_INPUT.items())))

    pipeline = FeaturePipeline(FEATURE_COLUMNS)

    X1 = pipeline.transform([BASE_INPUT])
    X2 = pipeline.transform([reversed_input])

    pd.testing.assert_frame_equal(X1, X2)