# Put the code for your API here.

import os
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(
    title="Census Income Prediction API",
    description="Inference API for predicting census salary bracket.",
    version="1.0.0",
)

project_root = Path(__file__).resolve().parent
model_path = project_root / "model" / "model.joblib"
encoder_path = project_root / "model" / "encoder.joblib"
lb_path = project_root / "model" / "lb.joblib"

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

_model = None
_encoder = None
_lb = None


def get_artifacts():
    global _model, _encoder, _lb
    if _model is None:
        _model = load_model(model_path)
    if _encoder is None:
        _encoder = load_model(encoder_path)
    if _lb is None:
        _lb = load_model(lb_path)
    return _model, _encoder, _lb


class CensusInput(BaseModel):
    age: int = Field(..., examples=[39], description="Age")
    workclass: str = Field(
        ..., examples=["State-gov"], description="Workclass"
    )
    fnlgt: int = Field(..., examples=[77516], description="Final weight")
    education: str = Field(
        ..., examples=["Bachelors"], description="Education level"
    )
    education_num: int = Field(
        ..., alias="education-num", examples=[13], description="Education num"
    )
    marital_status: str = Field(
        ...,
        alias="marital-status",
        examples=["Never-married"],
        description="Marital status",
    )
    occupation: str = Field(
        ...,
        alias="occupation",
        examples=["Adm-clerical"],
        description="Occupation",
    )
    relationship: str = Field(
        ..., examples=["Not-in-family"], description="Relationship"
    )
    race: str = Field(..., examples=["White"], description="Race")
    sex: str = Field(..., examples=["Male"], description="Gender")
    capital_gain: int = Field(
        ..., alias="capital-gain", examples=[2174], description="Capital gain"
    )
    capital_loss: int = Field(
        ..., alias="capital-loss", examples=[0], description="Capital loss"
    )
    hours_per_week: int = Field(
        ...,
        alias="hours-per-week",
        examples=[40],
        description="Hours per week",
    )
    native_country: str = Field(
        ...,
        alias="native-country",
        examples=["United-States"],
        description="Native country",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(..., examples=["<=50K"])


@app.get("/")
def get_root() -> Dict[str, str]:
    """Root GET endpoint returning a greeting."""
    return {"greeting": "Welcome to the Census Income Prediction API!"}


@app.post("/predict", response_model=PredictionResponse)
@app.post("/model", response_model=PredictionResponse)
def predict_model(payload: CensusInput) -> PredictionResponse:
    """POST endpoint performing model inference."""
    try:
        model, encoder, lb = get_artifacts()
        row_dict = payload.model_dump(by_alias=True)
        df = pd.DataFrame([row_dict])
        X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X)
        pred_label = lb.inverse_transform(preds)[0]
        return PredictionResponse(prediction=str(pred_label))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
