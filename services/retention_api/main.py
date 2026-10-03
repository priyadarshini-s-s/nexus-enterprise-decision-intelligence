from typing import Literal
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from services.decision_engine.population_policy import (
    prioritize_customers,
)


# =============================================================================
# NEXUS — Retention Model + Decision API
# =============================================================================

MODEL_URI = os.getenv(
    "NEXUS_RETENTION_MODEL_URI",
    "models:/logistic_retention/2",
)

MODEL_NAME = "logistic_retention"
MODEL_VERSION = "2"
TARGET = "repeat_purchase_90d"
PREDICTION_HORIZON_DAYS = 90
FEATURE_VERSION = "V1"


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


MODEL_DTYPES = {
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


# =============================================================================
# Request schemas
# =============================================================================

class RetentionPredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    orders_to_date: int = Field(ge=1)

    total_spend_to_date: float = Field(ge=0)

    total_order_items_to_date: int = Field(ge=1)

    unique_products_to_date: int = Field(ge=1)

    unique_categories_to_date: int = Field(ge=1)

    unique_sellers_to_date: int = Field(ge=1)

    recency_days: float = Field(ge=0)

    average_order_value: float = Field(ge=0)

    average_items_per_order: float = Field(ge=0)

    orders_30d: int = Field(ge=0)

    spend_30d: float = Field(ge=0)

    items_30d: int = Field(ge=0)

    orders_60d: int = Field(ge=0)

    spend_60d: float = Field(ge=0)

    items_60d: int = Field(ge=0)

    orders_90d: int = Field(ge=0)

    spend_90d: float = Field(ge=0)

    items_90d: int = Field(ge=0)

    average_review_score_to_date: float = Field(
        ge=0,
        le=5,
    )

    payment_observation_available: int = Field(
        ge=0,
        le=1,
    )

    review_observation_available: int = Field(
        ge=0,
        le=1,
    )

    item_observation_available: int = Field(
        ge=0,
        le=1,
    )


class RetentionDecisionCustomer(
    RetentionPredictionRequest
):

    customer_id: str = Field(
        min_length=1
    )


class RetentionPredictionResponse(BaseModel):

    model: str

    model_version: str

    target: Literal[
        "repeat_purchase_90d"
    ]

    prediction_horizon_days: int

    repeat_purchase_probability: float = Field(
        ge=0,
        le=1,
    )


class RetentionDecisionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    customers: list[
        RetentionDecisionCustomer
    ] = Field(
        min_length=1,
        max_length=100_000,
    )

    capacity_fraction: float = Field(
        gt=0,
        le=1,
    )


class RetentionDecisionCustomerResponse(
    BaseModel
):

    customer_id: str

    repeat_purchase_probability: float = Field(
        ge=0,
        le=1,
    )

    decision_rank: int

    priority: Literal[
        "prioritize",
        "standard",
    ]


class RetentionDecisionResponse(BaseModel):

    model: str

    model_version: str

    target: Literal[
        "repeat_purchase_90d"
    ]

    prediction_horizon_days: int

    capacity_fraction: float

    population_size: int

    target_population_size: int

    customers: list[
        RetentionDecisionCustomerResponse
    ]


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="NEXUS Retention Intelligence API",
    description=(
        "Serves the registered NEXUS retention propensity model "
        "and population-level capacity-based decision policy."
    ),
    version="1.1.0",
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
# Feature preparation
# =============================================================================

def prepare_model_frame(
    requests: list[dict],
) -> pd.DataFrame:

    frame = pd.DataFrame(
        requests
    )

    frame = frame[
        FEATURES
    ]

    frame = frame.astype(
        MODEL_DTYPES
    )

    return frame


# =============================================================================
# Health endpoint
# =============================================================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


# =============================================================================
# Model metadata endpoint
# =============================================================================

@app.get(
    "/model"
)
def model_info():

    return {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET,
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "feature_version": FEATURE_VERSION,
        "feature_count": len(FEATURES),
        "features": FEATURES,
    }


# =============================================================================
# Individual prediction endpoint
# =============================================================================

@app.post(
    "/predict",
    response_model=RetentionPredictionResponse,
)
def predict(
    request: RetentionPredictionRequest,
):

    request_data = request.model_dump()

    input_frame = prepare_model_frame(
        [request_data]
    )

    probability = float(
        model.predict_proba(
            input_frame
        )[0, 1]
    )

    return RetentionPredictionResponse(
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        target=TARGET,
        prediction_horizon_days=PREDICTION_HORIZON_DAYS,
        repeat_purchase_probability=probability,
    )


# =============================================================================
# Population decision endpoint
# =============================================================================

@app.post(
    "/decision",
    response_model=RetentionDecisionResponse,
)
def decision(
    request: RetentionDecisionRequest,
):

    customer_records = [
        customer.model_dump()
        for customer in request.customers
    ]

    # ---------------------------------------------------------------
    # Extract identity separately from model features.
    # ---------------------------------------------------------------

    customer_ids = [
        record["customer_id"]
        for record in customer_records
    ]

    # ---------------------------------------------------------------
    # Validate unique IDs within this decision population.
    # ---------------------------------------------------------------

    if len(customer_ids) != len(set(customer_ids)):
        raise HTTPException(
            status_code=400,
            detail=(
                "customer_id values must be unique "
                "within a decision population."
            ),
        )

    # ---------------------------------------------------------------
    # Score the population.
    # ---------------------------------------------------------------

    feature_records = []

    for record in customer_records:

        feature_records.append(
            {
                feature: record[feature]
                for feature in FEATURES
            }
        )

    input_frame = prepare_model_frame(
        feature_records
    )

    probabilities = model.predict_proba(
        input_frame
    )[:, 1]

    # ---------------------------------------------------------------
    # Build decision population.
    # ---------------------------------------------------------------

    scored_population = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "repeat_purchase_probability": probabilities,
        }
    )

    # ---------------------------------------------------------------
    # Apply population-level capacity policy.
    # ---------------------------------------------------------------

    decisions = prioritize_customers(
        scored_population,
        capacity_fraction=request.capacity_fraction,
        customer_id_column="customer_id",
    )

    # ---------------------------------------------------------------
    # Convert to API response.
    # ---------------------------------------------------------------

    customers_response = [
        RetentionDecisionCustomerResponse(
            customer_id=row["customer_id"],
            repeat_purchase_probability=float(
                row[
                    "repeat_purchase_probability"
                ]
            ),
            decision_rank=int(
                row["decision_rank"]
            ),
            priority=row["priority"],
        )
        for _, row in decisions.iterrows()
    ]

    target_population_size = int(
        decisions[
            "target_population_size"
        ].iloc[0]
    )

    return RetentionDecisionResponse(
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        target=TARGET,
        prediction_horizon_days=PREDICTION_HORIZON_DAYS,
        capacity_fraction=request.capacity_fraction,
        population_size=len(decisions),
        target_population_size=target_population_size,
        customers=customers_response,
    )