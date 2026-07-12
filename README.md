# Census Income Classification API — End-to-End MLOps Pipeline

[![CI/CD Pipeline](https://github.com/kenseiAmit/census-income-api/actions/workflows/ci.yml/badge.svg)](https://github.com/kenseiAmit/census-income-api/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20Inference-009688.svg)](https://fastapi.tiangolo.com/)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-flake8%20(79%20chars)-black.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end production Machine Learning service predicting annual income brackets (`>50K` vs `<=50K`) on US Census Bureau demographic data—featuring **leakage-free feature encoding**, **automated demographic slice fairness evaluation**, **FastAPI Pydantic v2 inference**, **Model Card governance**, and **gated CI/CD deployment** (GitHub Actions + Render).

- **Live Cloud Endpoint:** https://census-income-api-3hq5.onrender.com
- **Interactive OpenAPI Docs:** https://census-income-api-3hq5.onrender.com/docs

---

## Architecture & Highlights

- **Leakage-Safe Feature Pipeline (`income_classifier/ml/data.py`):** Fits `OneHotEncoder(handle_unknown="ignore")` and `LabelBinarizer` strictly on training splits and reuses serialized artifacts during validation and real-time API inference.
- **Demographic Slice Fairness Auditing (`income_classifier/ml/model.py`):** Computes Precision, Recall, and $F_1$ across every categorical slice (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`)—persisted in [`slice_output.txt`](slice_output.txt) and documented in [`MODEL_CARD.md`](MODEL_CARD.md).
- **Typed FastAPI Service (`main.py`):** Exposes health/greeting (`GET /`) and inference (`POST /predict`) with Pydantic field aliases matching hyphenated census attributes (`education-num`, `marital-status`, `hours-per-week`).
- **Automated Quality Gates (`.github/workflows/ci.yml`):** Enforces strict `flake8` complexity/79-char PEP8 checks and 9 `pytest` unit + `TestClient` API tests prior to triggering Render continuous deployment.

---

## Repository Structure

```text
.
├── main.py                        # FastAPI inference service (GET / and POST /predict)
├── live_post.py                   # Live HTTP smoke-test client
├── income_classifier/
│   ├── train_model.py             # Training, artifact serialization & slice evaluation entrypoint
│   └── ml/
│       ├── data.py                # Feature preprocessing & categorical encoding
│       └── model.py               # RandomForest training, overall metrics & slice auditing
├── model/                         # Serialized model.joblib, encoder.joblib & lb.joblib
├── data/census.csv                # Cleaned UCI Adult Census dataset (32,562 rows)
├── tests/
│   ├── test_ml.py                 # Unit tests for preprocessing, training, metrics & serialization
│   └── test_api.py                # FastAPI TestClient tests covering <=50K and >50K payloads
├── MODEL_CARD.md                  # Model Card: intended use, slice disparities & ethical caveats
├── slice_output.txt               # Full per-slice precision/recall/F1 report
├── .github/workflows/ci.yml       # GitHub Actions lint, test & cloud deploy workflow
└── azure-pipelines.yml            # Alternative Azure DevOps CI/CD pipeline definition
```

---

## Holdout Test Set Performance (20% Split — 6,514 Samples)

| Metric | Overall Score | Notes |
|:---|:---:|:---|
| **Precision** | **0.7875** | 78.75% of predicted `>50K` earners are true positives |
| **Recall** | **0.5754** | 57.54% sensitivity on positive class (`~24%` base rate) |
| **F1-Score** | **0.6650** | Harmonic mean on imbalanced census distribution |

See [`MODEL_CARD.md`](MODEL_CARD.md) and [`slice_output.txt`](slice_output.txt) for disaggregated subgroup performance and bias analysis:

| Demographic Dimension | Slice | Holdout Count | Precision | Recall | F1-Score |
|:---|:---|---:|---:|---:|---:|
| **Education** | `Prof-school` | 116 | `0.8298` | `0.9286` | **`0.8764`** |
| **Education** | `Doctorate` | 77 | `0.8500` | `0.8947` | **`0.8718`** |
