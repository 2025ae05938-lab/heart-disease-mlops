"""Unit tests for data loading and cleaning.

These tests build a tiny fake dataset, so they run fast and need no internet.
"""
import numpy as np
import pandas as pd

from data_utils import FEATURES, TARGET, clean, has_missing, split_xy


def make_raw_df() -> pd.DataFrame:
    """A small hand-made DataFrame that looks like the raw data."""
    return pd.DataFrame(
        {
            "age": [63, 67, 41, 56],
            "sex": [1, 1, 0, 1],
            "cp": [1, 4, 2, 2],
            "trestbps": [145, 160, 130, 120],
            "chol": [233, 286, 204, 236],
            "fbs": [1, 0, 0, 0],
            "restecg": [2, 2, 2, 0],
            "thalach": [150, 108, 172, 178],
            "exang": [0, 1, 0, 0],
            "oldpeak": [2.3, 1.5, 1.4, 0.8],
            "slope": [3, 2, 1, 1],
            "ca": [0.0, 3.0, np.nan, 0.0],      # one missing value
            "thal": [6.0, 3.0, 3.0, np.nan],    # one missing value
            "target": [0, 2, 0, 1],             # multi-class original labels
        }
    )


def test_clean_removes_missing_values():
    cleaned = clean(make_raw_df())
    assert not has_missing(cleaned)  # no NaN should remain


def test_target_is_binary():
    cleaned = clean(make_raw_df())
    assert set(cleaned[TARGET].unique()) <= {0, 1}  # only 0 or 1 allowed


def test_clean_keeps_expected_columns():
    cleaned = clean(make_raw_df())
    for col in FEATURES + [TARGET]:
        assert col in cleaned.columns  # every needed column is present


def test_split_xy_shapes():
    cleaned = clean(make_raw_df())
    X, y = split_xy(cleaned)
    assert X.shape[1] == len(FEATURES)  # X has all feature columns
    assert len(X) == len(y)  # one label per row
