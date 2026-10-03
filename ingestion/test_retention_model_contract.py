from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


# =============================================================================
# NEXUS — Retention Model Serving Contract Test
# =============================================================================
#
# Purpose:
#   Validate that the registered MLflow model can be consumed using the
#   documented 22-feature serving contract.
#
# This test does NOT retrain the model.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "retention_splits"
    / "test.parquet"
)

MODEL_URI = "models:/logistic_retention/2"

TARGET = "repeat_purchase_90d"


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


def validate_contract(dataframe):

    expected = set(FEATURES)
    actual = set(dataframe.columns)

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(sorted(missing))
        )

    if extra:
        raise ValueError(
            "Unexpected features: "
            + ", ".join(sorted(extra))
        )

    if list(dataframe.columns) != FEATURES:
        raise ValueError(
            "Feature order does not match the NEXUS serving contract."
        )

    if dataframe.isna().any().any():
        raise ValueError(
            "Serving input contains missing values."
        )

    return True


def main():

    print("=" * 80)
    print("NEXUS — Retention Model Serving Contract Test")
    print("=" * 80)

    # =========================================================================
    # 1. Load test data
    # =========================================================================

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATA_PATH}"
        )

    test = pd.read_parquet(
        TEST_DATA_PATH
    )

    valid_input = (
        test[FEATURES]
        .iloc[[0]]
        .copy()
    )
    valid_input = valid_input.astype(
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

    print("\nVALID INPUT")
    print("-" * 80)

    print(
        f"Feature count: {len(valid_input.columns)}"
    )

    print(
        f"Row count: {len(valid_input)}"
    )

    # =========================================================================
    # 2. Validate our application-level contract
    # =========================================================================

    validate_contract(
        valid_input
    )

    print(
        "Contract validation: PASS"
    )

    # =========================================================================
    # 3. Load registered MLflow model
    # =========================================================================

    print("\nMODEL")
    print("-" * 80)

    print(
        f"Model URI: {MODEL_URI}"
    )

    model = mlflow.sklearn.load_model(
    MODEL_URI
)

    print(
        "Registered model loading: PASS"
    )

    # =========================================================================
    # 4. Generate prediction
    # =========================================================================

    probability_1 = float(
    model.predict_proba(
        valid_input
    )[0, 1]
    )

    print("\nPREDICTION")
    print("-" * 80)

    print(
        f"repeat_purchase_90d probability: "
        f"{probability_1:.10f}"
    )

    # =========================================================================
    # 5. Validate probability range
    # =========================================================================

    if not 0.0 <= probability_1 <= 1.0:
        raise AssertionError(
            "Prediction is outside [0, 1]."
        )

    print(
        "Probability range: PASS"
    )

    # =========================================================================
    # 6. Reproducibility test
    # =========================================================================

    probability_2 = float(
    model.predict_proba(
        valid_input
    )[0, 1]
    )

    if probability_1 != probability_2:
        raise AssertionError(
            "Repeated prediction produced different results."
        )

    print(
        "Prediction reproducibility: PASS"
    )

    # =========================================================================
    # 7. Missing-feature test
    # =========================================================================

    missing_input = valid_input.drop(
        columns=["recency_days"]
    )

    try:

        validate_contract(
            missing_input
        )

        raise AssertionError(
            "Missing-feature input was incorrectly accepted."
        )

    except ValueError as exc:

        print(
            "\nMissing feature test: PASS"
        )

        print(
            f"Rejected as expected: {exc}"
        )

    # =========================================================================
    # 8. Extra-feature test
    # =========================================================================

    extra_input = valid_input.copy()

    extra_input["unexpected_feature"] = 123.0

    try:

        validate_contract(
            extra_input
        )

        raise AssertionError(
            "Extra-feature input was incorrectly accepted."
        )

    except ValueError as exc:

        print(
            "\nExtra feature test: PASS"
        )

        print(
            f"Rejected as expected: {exc}"
        )

    # =========================================================================
    # 9. Target-column test
    # =========================================================================

    target_input = valid_input.copy()

    target_input[TARGET] = 0

    try:

        validate_contract(
            target_input
        )

        raise AssertionError(
            "Target column was incorrectly accepted."
        )

    except ValueError as exc:

        print(
            "\nTarget-column test: PASS"
        )

        print(
            f"Rejected as expected: {exc}"
        )

    # =========================================================================
    # 10. Final result
    # =========================================================================

    print("\n" + "=" * 80)
    print("RETENTION MODEL CONTRACT TEST COMPLETE")
    print("=" * 80)

    print(
        "\nModel: logistic_retention"
    )

    print(
        "Version: 2"
    )

    print(
        "Input features: 22"
    )

    print(
        "Output: repeat_purchase_90d probability"
    )

    print(
        "\nAll serving contract tests passed."
    )


if __name__ == "__main__":
    main()