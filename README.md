# Deploying a Scalable Machine Learning Pipeline with FastAPI

## Repository Information

Repository platform: Azure DevOps
Repository URL: https://dev.azure.com/udacity-student/fastapi-ml-model/_git/fastapi-ml-model-submission

This repository is private. The submitted complete-repository ZIP contains the complete project source code, Git history (.git), trained model artifacts, tests, pipeline configuration, and screenshots.

- **GitHub Repository**: https://github.com/udacity/fastapi-census-ml-pipeline
- **Live Application URL**: https://census-income-api.onrender.com
- **Swagger Documentation URL**: https://census-income-api.onrender.com/docs

---

## Project Overview
This project develops and operationalizes an end-to-end Machine Learning classification pipeline using Census Bureau data. The pipeline cleans the dataset, trains a Random Forest Classifier to predict whether an individual's salary exceeds \$50K per year, calculates slice-based fairness and performance metrics, and deploys the model via a production-ready FastAPI REST service with automated CI/CD pipelines (GitHub Actions & Azure Pipelines).

---

## Project Architecture & Directory Layout
```text
nd0821-c3-starter-code/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── azure-pipelines.yml            # Azure DevOps CI/CD pipeline definition
├── data/
│   └── census.csv                 # Cleaned Census Bureau dataset (spaces stripped)
├── model/
│   ├── model.joblib               # Trained Random Forest classifier
│   ├── encoder.joblib             # Fitted OneHotEncoder for categorical features
│   └── lb.joblib                  # Fitted LabelBinarizer for salary targets
├── screenshots/
│   ├── continuous_integration.png # Evidence of passing CI run (pytest & flake8)
│   ├── continuous_deployment.png  # Evidence of successful cloud deployment
│   ├── example.png                # Swagger UI showing POST body example
│   └── live_post.png              # Terminal output from live_post.py execution
├── starter/
│   ├── __init__.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── data.py                # Data preprocessing & feature transformation
│   │   └── model.py               # ML training, evaluation, slice metrics, I/O
│   └── train_model.py             # Model training & slice metrics script
├── tests/
│   ├── __init__.py
│   ├── test_ml.py                 # ML unit tests (6 tests)
│   └── test_api.py                # FastAPI endpoint tests (3 tests)
├── main.py                        # FastAPI application with GET and POST endpoints
├── live_post.py                   # Live API query script using requests module
├── sanitycheck.py                 # Rubric test compliance validation tool
├── slice_output.txt               # Evaluation metrics computed on data slices
├── model_card_template.md         # Comprehensive Model Card documentation
├── model_card.md                  # Model Card specification
├── requirements.txt               # Pinned project dependencies
└── setup.py                       # Local package installer
```

---

## Environment Setup

### 1. Virtual Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

---

## Data Cleaning & Model Training

### 1. Data Cleaning
The Census dataset originally contained leading whitespace in field values and column headers. All commas were sanitized and extraneous spaces removed while strictly preserving original column names and hyphens.

### 2. Model Training & Evaluation
To train the model, evaluate on holdout test data, serialize artifacts to `model/`, and generate slice metrics:
```bash
python -m starter.train_model
```

### 3. Model Performance on Holdout Test Set (20%)
- **Precision**: 0.7875 (78.75%)
- **Recall**: 0.5754 (57.54%)
- **F1-Score**: 0.6650 (66.50%)

### 4. Slicing Performance
Metrics are computed across all categorical slices (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`) and saved to `slice_output.txt`.

---

## FastAPI REST Service

### 1. Running the API Locally
Start the server using `uvicorn`:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API documentation is accessible at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

### 2. API Endpoints
- **GET `/`**: Returns a welcome greeting:
  ```json
  {"greeting": "Welcome to the Census Income Prediction API!"}
  ```
- **POST `/predict`** (and alias **POST `/model`**): Ingests demographic and employment attributes and outputs income classification:
  ```json
  {"prediction": "<=50K"}
  ```

### 3. Testing the Live API Script
Query the running API using `live_post.py`:
```bash
python live_post.py --url http://127.0.0.1:8000/predict
```
Example Output:
```text
Sending POST request to: http://127.0.0.1:8000/predict
Payload:
  age: 31
  workclass: Private
  fnlgt: 45781
  education: Masters
  education-num: 14
  marital-status: Never-married
  occupation: Prof-specialty
  relationship: Not-in-family
  race: White
  sex: Female
  capital-gain: 14084
  capital-loss: 0
  hours-per-week: 50
  native-country: United-States

Response Received:
HTTP Status Code: 200
Inference Result: {'prediction': '>50K'}
```

---

## Testing & Quality Assurance

### 1. Automated Unit Tests with Pytest
Execute the 9 unit tests covering ML functions, edge cases, model persistence, and API GET/POST endpoints:
```bash
pytest -v
```

### 2. PEP8 Code Linting with Flake8
Run flake8 across the repository to verify code conforms to PEP8 and has 0 errors:
```bash
flake8 . --count --max-complexity=10 --max-line-length=79 --statistics
```

### 3. Rubric Sanity Check
Validate test coverage for GET and POST methods with the Udacity sanity checker:
```bash
echo "tests/test_api.py" | python sanitycheck.py
```
Output:
```text
============= Sanity Check Report ===========
Your test cases look good!
```

---

## Continuous Integration & Deployment (CI/CD)

The repository provides both **GitHub Actions** (`.github/workflows/ci.yml`) and **Azure Pipelines** (`azure-pipelines.yml`) workflows:
1. **Triggers**: Automated on pushes to `main` and `master`, as well as Pull Requests.
2. **CI Stage**: Sets up Python environment, installs dependencies, runs `flake8` linting (0 errors required), and runs `pytest` (all tests passing required).
3. **CD Stage**: Gated strictly on successful CI. Automatically triggers deployment to the cloud application platform (Render / Heroku).

---

## Required Screenshots
All screenshots required by the rubric are located in `screenshots/`:
1. `screenshots/continuous_integration.png` — Evidence of passing CI run (pytest and flake8).
2. `screenshots/continuous_deployment.png` — Evidence of successful continuous deployment stage.
3. `screenshots/example.png` — Swagger UI (/docs) displaying POST endpoint request schema with example values.
4. `screenshots/live_post.png` — Terminal execution of `live_post.py` showing HTTP status 200 and model prediction.
