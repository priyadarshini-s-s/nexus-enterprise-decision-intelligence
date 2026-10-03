from pathlib import Path

import mlflow.sklearn
import pandas as pd


TEST_PATH = Path(
    "data/gold/olist/retention_splits/test.parquet"
)

OUTPUT_PATH = Path(
    "data/gold/olist/retention_decision_population.parquet"
)

MODEL_URI = "models:/logistic_retention/2"


MODEL_FEATURES = [
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


def main():

    print("=" * 80)
    print("NEXUS — Retention Population Scoring")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load held-out test population
    # ------------------------------------------------------------------

    print("\n[1/5] Loading Olist test population...")

    df = pd.read_parquet(TEST_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Validate identity
    # ------------------------------------------------------------------

    print("\n[2/5] Validating customer identity...")

    if "customer_unique_id" not in df.columns:
        raise ValueError(
            "customer_unique_id is missing from the test population."
        )

    if df["customer_unique_id"].isna().any():
        raise ValueError(
            "customer_unique_id contains missing values."
        )

    print(
        "Unique customers:",
        df["customer_unique_id"].nunique(),
    )

    # The temporal test set contains multiple snapshot rows per customer.
    # Decisioning must remain at customer × snapshot grain.
    duplicate_keys = df.duplicated(
        subset=[
            "customer_unique_id",
            "snapshot_timestamp",
        ]
    ).sum()

    if duplicate_keys != 0:
        raise ValueError(
            f"Duplicate customer × snapshot rows: {duplicate_keys}"
        )

    print("Customer × snapshot uniqueness: PASS")

    # ------------------------------------------------------------------
    # 3. Validate model feature contract
    # ------------------------------------------------------------------

    print("\n[3/5] Validating model feature contract...")

    missing_features = [
        col for col in MODEL_FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = df[MODEL_FEATURES].copy()

    # Match MLflow signature exactly.
    int32_features = [
        "payment_observation_available",
        "review_observation_available",
        "item_observation_available",
    ]

    for feature in int32_features:
        X[feature] = X[feature].astype("int32")

    print(f"Model features: {len(MODEL_FEATURES)}")
    print("Feature contract: PASS")

    # ------------------------------------------------------------------
    # 4. Load registered MLflow model
    # ------------------------------------------------------------------

    print("\n[4/5] Loading registered MLflow model...")

    model = mlflow.sklearn.load_model(MODEL_URI)

    print(f"Model URI: {MODEL_URI}")
    print("Registered model loading: PASS")

    # ------------------------------------------------------------------
    # 5. Generate probabilities
    # ------------------------------------------------------------------

    print("\n[5/5] Generating retention probabilities...")

    probabilities = model.predict_proba(X)[:, 1]

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(
            "Model produced probabilities outside [0, 1]."
        )

    scored = pd.DataFrame(
        {
            "customer_unique_id": df[
                "customer_unique_id"
            ].values,

            "snapshot_timestamp": df[
                "snapshot_timestamp"
            ].values,

            "repeat_purchase_probability": probabilities,

            # Keep the actual observed target for evaluation only.
            "repeat_purchase_90d": df[
                "repeat_purchase_90d"
            ].values,
        }
    )

    # Useful contextual fields for later analysis.
    if "last_purchase_timestamp" in df.columns:
        scored["last_purchase_timestamp"] = df[
            "last_purchase_timestamp"
        ].values

    # Deterministic ordering.
    scored = scored.sort_values(
        [
            "snapshot_timestamp",
            "repeat_purchase_probability",
            "customer_unique_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nScoring complete.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(scored):,}")

    print(
        "\nProbability summary:"
    )

    print(
        scored[
            "repeat_purchase_probability"
        ].describe()
    )

    print("\nTop 10 scored customers:")

    print(
        scored[
            [
                "customer_unique_id",
                "snapshot_timestamp",
                "repeat_purchase_probability",
            ]
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("RETENTION POPULATION SCORING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()