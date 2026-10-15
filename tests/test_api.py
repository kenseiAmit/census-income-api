"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    """Create test client for FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def low_income_payload():
    """Sample payload expected to result in <=50K prediction."""
    return {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }


@pytest.fixture(scope="module")
def high_income_payload():
    """Sample payload expected to result in >50K prediction."""
    return {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }


def test_get_root(client):
    """Test GET endpoint on root path for status code and welcome greeting."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["greeting"] == "Welcome to the Census Income Prediction API!"


def test_post_predict_low_income(client, low_income_payload):
    """Test POST endpoint predicting <=50K income bracket."""
    response = client.post("/predict", json=low_income_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "<=50K"


def test_post_predict_high_income(client, high_income_payload):
    """Test POST endpoint predicting >50K income bracket."""
    response = client.post("/predict", json=high_income_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == ">50K"
