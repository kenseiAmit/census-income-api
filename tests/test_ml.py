"""Unit tests for machine learning modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
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
    """Test that train_model returns a trained RandomForestClassifier."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y, n_estimators=10, max_depth=5)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_inference(sample_data):
    """Test model inference returns expected binary predictions."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y, n_estimators=10, max_depth=5)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics():
    """Test compute_model_metrics returns valid metric values."""
    y = np.array([0, 1, 1, 0, 1, 0])
    preds = np.array([0, 1, 0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_save_and_load_model(tmp_path, sample_data):
    """Test saving and loading model artifacts."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y, n_estimators=10, max_depth=5)
    target_path = tmp_path / "test_model.joblib"
    save_model(model, target_path)
    assert target_path.exists()

    loaded = load_model(target_path)
    preds_orig = inference(model, X)
    preds_loaded = inference(loaded, X)
    np.testing.assert_array_equal(preds_orig, preds_loaded)
