"""Unit tests for machine learning modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from income_classifier.ml.data import process_data
from income_classifier.ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def sample_data():
    """Load sample census data for testing."""
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "census.csv"
    data = pd.read_csv(csv_path)
    return data.head(100)


def test_process_data_training(sample_data):
    """Test process_data in training mode produces valid arrays and objects."""
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == len(sample_data)
    assert len(y) == len(sample_data)
    assert encoder is not None
    assert lb is not None


def test_process_data_inference(sample_data):
    """Test process_data in inference mode without target label."""
    X_train, _, encoder, lb = process_data(
        sample_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    infer_df = sample_data.drop(columns=["salary"]).head(10)
    X_infer, y_infer, _, _ = process_data(
        infer_df,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert isinstance(X_infer, np.ndarray)
    assert X_infer.shape[0] == len(infer_df)
    assert X_infer.shape[1] == X_train.shape[1]
    assert len(y_infer) == 0


def test_train_model(sample_data):
