"""Shared helpers for loading, cleaning and splitting the heart disease data.

Both train.py, the tests and the notebook import from here, so the cleaning
logic lives in exactly one place.
"""
import os

import numpy as np
import pandas as pd

RAW_PATH = os.path.join("data", "heart_raw.csv")
CLEAN_PATH = os.path.join("data", "heart_clean.csv")

# Columns that are real numbers -> will be scaled
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Columns that are categories (codes, not quantities) -> will be one-hot encoded
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
# Everything the model uses as input
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV. The file marks missing values with '?'."""
    return pd.read_csv(path, na_values="?")  # treat '?' as NaN (missing)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data: fix the target, drop duplicates, fill missing values."""
    df = df.copy()  # never modify the caller's DataFrame
    # Original target is 0 (healthy) or 1-4 (disease severity). Make it binary.
    df[TARGET] = (df[TARGET] > 0).astype(int)
    df = df.drop_duplicates()  # remove exact duplicate rows
    # 'ca' and 'thal' have a few missing values: fill with the most common value
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    # Safety net: fill any other missing numeric value with the median
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    # Categorical columns are stored as whole numbers (e.g. 3.0 -> 3)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    """Convenience: load the raw CSV and clean it in one call."""
    return clean(load_raw(path))


def save_clean(df: pd.DataFrame, path: str = CLEAN_PATH) -> None:
    """Save the cleaned dataset (one of the assignment deliverables)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def split_xy(df: pd.DataFrame):
    """Split a cleaned DataFrame into features X and label y."""
    return df[FEATURES], df[TARGET]


def has_missing(df: pd.DataFrame) -> bool:
    """True if any value is missing; used by the tests."""
    return bool(np.any(df.isna().values))
