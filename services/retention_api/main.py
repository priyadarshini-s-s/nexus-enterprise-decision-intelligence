from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# NEXUS — Retention Model API
# =============================================================================

MODEL_URI = "models:/logistic_retention/2"


FEATURES = [
    "orders_to_date",
    "total_spend_to_date",
    "total_order_items_to_date",
    "unique_products_to_date",
    "unique_categories_to_date",
    "unique_sellers_to_date",
    "recency_days",
    "average_order_value",
    "average_items_per_order",
    "orders_30d",
    "spend_30d",
    "items_30d",
    "orders_60d",
    "spend_60d",
    "items_60d",
    "orders_90d",
    "spend_90d",
    "items_90d",
    "average_review_score_to_date",
    "payment_observation_available",
    "review_observation_available",
    "item_observation_available",
]


# =============================================================================
# Request schema
# =============================================================================

class RetentionPredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    orders_to_date: int = Field(
        ge=1
    )

    total_spend_to_date: float = Field(
        ge=0
    )

    total_order_items_to_date: int = Field(
        ge=1
    )

    unique_products_to_date: int = Field(
        ge=1
    )

    unique_categories_to_date: int = Field(
        ge=1
    )

    unique_sellers_to_date: int = Field(
        ge=1
    )

    recency_days: float = Field(
        ge=0
    )

    average_order_value: float = Field(
        ge=0
    )

    average_items_per_order: float = Field(
        ge=0
    )

    orders_30d: int = Field(
        ge=0
    )

    spend_30d: float = Field(
        ge=0
    )

    items_30d: int = Field(
        ge=0
    )

    orders_60d: int = Field(
        ge=0
    )

    spend_60d: float = Field(
        ge=0
    )

    items_60d: int = Field(
        ge=0
    )

    orders_90d: int = Field(
        ge=0
    )

    spend_90d: float = Field(
        ge=0
    )

    items_90d: int = Field(
        ge=0
    )

    average_review_score_to_date: float = Field(
        ge=0,
        le=5
    )

    payment_observation_available: int = Field(
        ge=0,
        le=1
    )

    review_observation_available: int = Field(
        ge=0,
        le=1
    )

    item_observation_available: int = Field(
        ge=0,
        le=1
    )


# =============================================================================
# Response schema
# =============================================================================

class RetentionPredictionResponse(BaseModel):

    model: str

    model_version: str

    target: Literal[
        "repeat_purchase_90d"
    ]

    prediction_horizon_days: int

    repeat_purchase_probability: float = Field(
        ge=0,
        le=1
    )


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="NEXUS Retention Intelligence API",
    description=(
        "Serves the registered NEXUS retention propensity model."
    ),
    version="1.0.0",
)


# =============================================================================
# Load registered MLflow model once at startup
# =============================================================================

print(
    f"Loading MLflow model: {MODEL_URI}"
)

model = mlflow.sklearn.load_model(
    MODEL_URI
)

print(
    "MLflow retention model loaded successfully."
)


# =============================================================================
# Health endpoint
# =============================================================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "healthy",
        "model": "logistic_retention",
        "model_version": "2",
    }


# =============================================================================
# Model metadata endpoint
# =============================================================================

@app.get(
    "/model"
)
def model_info():

    return {
        "model": "logistic_retention",
        "model_version": "2",
        "target": "repeat_purchase_90d",
        "prediction_horizon_days": 90,
        "feature_version": "V1",
        "feature_count": len(FEATURES),
        "features": FEATURES,
    }


# =============================================================================
# Prediction endpoint
# =============================================================================

@app.post(
    "/predict",
    response_model=RetentionPredictionResponse,
)
def predict(
    request: RetentionPredictionRequest,
):

    # -------------------------------------------------------------------------
    # Convert validated Pydantic request to DataFrame
    # -------------------------------------------------------------------------

    request_data = request.model_dump()

    input_frame = pd.DataFrame(
        [
            request_data
        ]
    )

    # Explicitly enforce model feature ordering.
    input_frame = input_frame[
        FEATURES
    ]
    # MLflow Version 2 contains an explicit model signature.
    # Cast the DataFrame to the exact dtypes expected by that signature.

    input_frame = input_frame.astype(
        {
            "orders_to_date": "int64",
            "total_spend_to_date": "float64",
            "total_order_items_to_date": "int64",
            "unique_products_to_date": "float64",
            "unique_categories_to_date": "float64",
            "unique_sellers_to_date": "float64",
            "recency_days": "float64",
            "average_order_value": "float64",
            "average_items_per_order": "float64",
            "orders_30d": "float64",
            "spend_30d": "float64",
            "items_30d": "int64",
            "orders_60d": "float64",
            "spend_60d": "float64",
            "items_60d": "int64",
            "orders_90d": "float64",
            "spend_90d": "float64",
            "items_90d": "int64",
            "average_review_score_to_date": "float64",
            "payment_observation_available": "int32",
            "review_observation_available": "int32",
            "item_observation_available": "int32",
        }
    )

    # -------------------------------------------------------------------------
    # Generate probability
    # -------------------------------------------------------------------------

    probability = float(
        model.predict_proba(
            input_frame
        )[0, 1]
    )

    # -------------------------------------------------------------------------
    # Return business-facing response
    # -------------------------------------------------------------------------

    return RetentionPredictionResponse(
        model="logistic_retention",
        model_version="2",
        target="repeat_purchase_90d",
        prediction_horizon_days=90,
        repeat_purchase_probability=probability,
    )