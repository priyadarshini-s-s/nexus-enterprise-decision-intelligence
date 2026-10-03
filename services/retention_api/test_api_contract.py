import pytest
from fastapi.testclient import TestClient

from services.retention_api.main import app


client = TestClient(app)


VALID_CUSTOMER = {
    "customer_id": "C001",
    "orders_to_date": 1,
    "total_spend_to_date": 100,
    "total_order_items_to_date": 1,
    "unique_products_to_date": 1,
    "unique_categories_to_date": 1,
    "unique_sellers_to_date": 1,
    "recency_days": 100,
    "average_order_value": 100,
    "average_items_per_order": 1,
    "orders_30d": 0,
    "spend_30d": 0,
    "items_30d": 0,
    "orders_60d": 0,
    "spend_60d": 0,
    "items_60d": 0,
    "orders_90d": 1,
    "spend_90d": 100,
    "items_90d": 1,
    "average_review_score_to_date": 5,
    "payment_observation_available": 1,
    "review_observation_available": 1,
    "item_observation_available": 1,
}


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model"] == "logistic_retention"
    assert body["model_version"] == "2"


def test_model_metadata():

    response = client.get(
        "/model"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "logistic_retention"
    assert body["model_version"] == "2"
    assert body["target"] == "repeat_purchase_90d"
    assert body["prediction_horizon_days"] == 90
    assert body["feature_count"] == 22
    assert len(body["features"]) == 22


def test_predict():

    response = client.post(
        "/predict",
        json={
            key: value
            for key, value in VALID_CUSTOMER.items()
            if key != "customer_id"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "logistic_retention"
    assert body["model_version"] == "2"
    assert body["target"] == "repeat_purchase_90d"

    probability = body[
        "repeat_purchase_probability"
    ]

    assert 0 <= probability <= 1


def test_predict_reproducibility():

    payload = {
        key: value
        for key, value in VALID_CUSTOMER.items()
        if key != "customer_id"
    }

    response_1 = client.post(
        "/predict",
        json=payload,
    )

    response_2 = client.post(
        "/predict",
        json=payload,
    )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    probability_1 = response_1.json()[
        "repeat_purchase_probability"
    ]

    probability_2 = response_2.json()[
        "repeat_purchase_probability"
    ]

    assert probability_1 == probability_2


def test_predict_rejects_extra_feature():

    payload = {
        key: value
        for key, value in VALID_CUSTOMER.items()
        if key != "customer_id"
    }

    payload["unexpected_feature"] = 123

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_decision_population():

    customers = []

    for i in range(10):

        customer = VALID_CUSTOMER.copy()

        customer["customer_id"] = (
            f"C{i + 1:03d}"
        )

        # Make each customer slightly different
        # so the ranking is deterministic.
        customer[
            "total_spend_to_date"
        ] = 100 + i * 25

        customer[
            "orders_to_date"
        ] = 1 + (i % 3)

        customers.append(customer)

    response = client.post(
        "/decision",
        json={
            "capacity_fraction": 0.20,
            "customers": customers,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["population_size"] == 10

    assert body[
        "target_population_size"
    ] == 2

    decisions = body["customers"]

    assert len(decisions) == 10

    prioritized = [
        customer
        for customer in decisions
        if customer["priority"]
        == "prioritize"
    ]

    assert len(prioritized) == 2

    assert [
        customer["decision_rank"]
        for customer in prioritized
    ] == [1, 2]

    probabilities = [
        customer[
            "repeat_purchase_probability"
        ]
        for customer in decisions
    ]

    assert probabilities == sorted(
        probabilities,
        reverse=True,
    )


def test_decision_rejects_invalid_capacity():

    response = client.post(
        "/decision",
        json={
            "capacity_fraction": 0,
            "customers": [
                VALID_CUSTOMER,
            ],
        },
    )

    assert response.status_code == 422


def test_decision_rejects_duplicate_customer_ids():

    customer_1 = VALID_CUSTOMER.copy()
    customer_2 = VALID_CUSTOMER.copy()

    customer_1["customer_id"] = "C001"
    customer_2["customer_id"] = "C001"

    response = client.post(
        "/decision",
        json={
            "capacity_fraction": 0.20,
            "customers": [
                customer_1,
                customer_2,
            ],
        },
    )

    assert response.status_code == 400


def test_decision_rejects_empty_population():

    response = client.post(
        "/decision",
        json={
            "capacity_fraction": 0.20,
            "customers": [],
        },
    )

    assert response.status_code == 422