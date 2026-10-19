"""Unit tests for the model pipeline.

We train a tiny model on synthetic data (fast, no files needed) and check
that it behaves sensibly.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_utils import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from train import build_pipeline


def make_training_data(n: int = 120):
    """Random fake patients with realistic column types."""
    rng = np.random.RandomState(0)
    data = {col: rng.normal(size=n) for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.randint(0, 3, size=n)
    X = pd.DataFrame(data)[FEATURES]
    y = pd.Series(rng.randint(0, 2, size=n))
    return X, y


def test_pipeline_trains_and_predicts():
    X, y = make_training_data()
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)  # one prediction per row
    assert set(preds) <= {0, 1}  # predictions are class labels


def test_probabilities_between_0_and_1():
    X, y = make_training_data()
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()  # valid probabilities


def test_preprocessing_output_has_no_nan():
    X, y = make_training_data()
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    transformed = pipe.named_steps["prep"].transform(X)
    assert not np.isnan(transformed).any()  # preprocessing yields clean numbers
