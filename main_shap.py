from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from shap_utils import COLUMNS, explain_row
from report_generator import build_report_pdf


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="StaySense SHAP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Existing trained pipeline
# DO NOT retrain or modify model_pipeline.pkl
# ---------------------------------------------------------
model = joblib.load(BASE_DIR / "model_pipeline.pkl")


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------
class Features(BaseModel):
    latitude: float = Field(
        ..., ge=-90, le=90, description="Latitude coordinate"
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Longitude coordinate"
    )
    price: float = Field(
        ..., gt=0, description="Price per night, must be positive"
    )
    minimum_nights: int = Field(
        ..., ge=1, le=365, description="Minimum nights required for booking"
    )
    number_of_reviews: int = Field(
        ..., ge=0, description="Total number of reviews"
    )
    reviews_per_month: float = Field(
        ..., ge=0, description="Average reviews per month"
    )
    calculated_host_listings_count: int = Field(
        ..., ge=0, description="Number of listings by this host"
    )
    availability_365: int = Field(
        ..., ge=0, le=365, description="Days available out of 365"
    )
    neighbourhood_group: str = Field(
        ..., min_length=1, description="Borough or neighbourhood group"
    )
    neighbourhood: str = Field(
        ..., min_length=1, description="Specific neighbourhood name"
    )


# Shared room is rare (~2.4% of listings) and the model under-detects it.
# Instead of changing the prediction, we flag a "risk" when its probability
# is meaningful but it is not the top class.
SHARED_RISK_THRESHOLD = 0.20


def make_row(features: Features) -> pd.DataFrame:
    """Convert validated API input into the exact model input format."""
    return pd.DataFrame(
        [features.model_dump()],
        columns=COLUMNS,
    )


# ---------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------
@app.get("/")
def home():
    """Serve the StaySense frontend."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css")
def css():
    return FileResponse(
        BASE_DIR / "style.css",
        media_type="text/css",
    )


@app.get("/script.js")
def javascript():
    return FileResponse(
        BASE_DIR / "script.js",
        media_type="application/javascript",
    )


@app.get("/report.html")
def report_page():
    return FileResponse(
        BASE_DIR / "report.html",
        media_type="text/html",
    )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "service": "StaySense SHAP API",
        "status": "online",
    }


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------
@app.get("/meta")
def meta():
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    return {
        "classes": [str(x) for x in classifier.classes_],
        "raw_features": COLUMNS,
        "transformed_feature_count": len(
            preprocessor.get_feature_names_out()
        ),
        "model": type(classifier).__name__,
    }


# ---------------------------------------------------------
# Original prediction behavior
# ---------------------------------------------------------
@app.post("/predict")
def predict(features: Features):
    row = make_row(features)

    prediction = model.predict(row)
    probability = model.predict_proba(row)[0]

    classes = [str(c) for c in model.named_steps["classifier"].classes_]
    shared_p = float(probability[classes.index("Shared room")])
    shared_risk = (
        str(prediction[0]) != "Shared room"
        and shared_p >= SHARED_RISK_THRESHOLD
    )

    return {
        "Predicted_room_type": prediction[0],
        "Probability": probability.tolist(),
        "Shared_room_probability": shared_p,
        "Shared_room_risk": shared_risk,
    }


# ---------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------
@app.post("/explain")
def explain(features: Features):
    try:
        row = make_row(features)
        return explain_row(model, row)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {exc}",
        ) from exc


# ---------------------------------------------------------
# PDF prediction report
# ---------------------------------------------------------
@app.post("/report")
def report(features: Features):
    try:
        row = make_row(features)

        # Use the same SHAP explanation that powers /explain.
        explanation = explain_row(model, row)

        pdf = build_report_pdf(
            features.model_dump(),
            explanation,
        )

        filename = "staysense_prediction_report.pdf"

        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{filename}"'
                )
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {exc}",
        ) from exc