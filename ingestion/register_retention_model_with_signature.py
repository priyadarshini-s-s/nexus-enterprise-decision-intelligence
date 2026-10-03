from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature


# =============================================================================
# NEXUS — Retention Model
# Register signature-aware model version
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
)

SPLIT_DIR = (
    GOLD_DIR
    / "retention_splits"
)

MODEL_NAME = "logistic_retention"

SOURCE_MODEL_URI = (
    "runs:/e4fc80077f5e407db7a03c6bcac94d19/model"
)


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


def main():

    print("=" * 80)
    print("NEXUS - MLflow Signature-Aware Model Registration")
    print("=" * 80)

    # =========================================================================
    # 1. Load the EXACT model from Version 1's source run
    # =========================================================================

    print("\nLoading source model...")
    print(SOURCE_MODEL_URI)

    model = mlflow.sklearn.load_model(
        SOURCE_MODEL_URI
    )

    # =========================================================================
    # 2. Load test data only for signature construction
    # =========================================================================

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    X_test = test[FEATURES]

    # Use a small example for the model signature.
    #
    # We deliberately do not use the target column.
    #
    # This describes:
    #
    # customer behavioral features
    #           ↓
    #      retention model
    #           ↓
    #       prediction
    #
    input_example = X_test.iloc[[0]].copy()

    # =========================================================================
    # 3. Generate model predictions
    # =========================================================================

    predictions = model.predict(
        input_example
    )

    # =========================================================================
    # 4. Infer MLflow signature
    # =========================================================================

    signature = infer_signature(
        input_example,
        predictions,
    )

    print("\nMODEL SIGNATURE")
    print("-" * 80)

    print(signature)

    # =========================================================================
    # 5. Create a NEW MLflow run
    # =========================================================================

    mlflow.set_experiment(
        "NEXUS-Retention-Intelligence"
    )

    with mlflow.start_run(
        run_name="retention_logistic_unweighted_v1_signature"
    ) as run:

        # ---------------------------------------------------------------------
        # Record lineage
        # ---------------------------------------------------------------------

        mlflow.log_param(
            "source_run_id",
            "e4fc80077f5e407db7a03c6bcac94d19",
        )

        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )

        mlflow.log_param(
            "feature_version",
            "V1",
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURES),
        )

        mlflow.log_param(
            "target",
            "repeat_purchase_90d",
        )

        mlflow.log_param(
            "prediction_horizon_days",
            90,
        )

        mlflow.log_param(
            "purpose",
            "signature_packaging",
        )

        # ---------------------------------------------------------------------
        # Log the EXACT model with signature
        # ---------------------------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        print("\nMODEL LOGGED")
        print("-" * 80)

        print(
            f"Model URI: {model_info.model_uri}"
        )

        print(
            f"Run ID: {run.info.run_id}"
        )

        # ---------------------------------------------------------------------
        # Register as a new version
        # ---------------------------------------------------------------------

        client = MlflowClient()

        model_version = (
            client.create_model_version(
                name=MODEL_NAME,
                source=model_info.model_uri,
                run_id=run.info.run_id,
            )
        )

        # ---------------------------------------------------------------------
        # Add metadata
        # ---------------------------------------------------------------------

        client.set_model_version_tag(
            MODEL_NAME,
            model_version.version,
            "model_role",
            "primary_retention_candidate",
        )

        client.set_model_version_tag(
            MODEL_NAME,
            model_version.version,
            "feature_version",
            "V1",
        )

        client.set_model_version_tag(
            MODEL_NAME,
            model_version.version,
            "target",
            "repeat_purchase_90d",
        )

        client.set_model_version_tag(
            MODEL_NAME,
            model_version.version,
            "prediction_horizon_days",
            "90",
        )

        client.set_model_version_tag(
            MODEL_NAME,
            model_version.version,
            "packaging_change",
            "added_model_signature",
        )

        print("\nREGISTERED MODEL")
        print("-" * 80)

        print(
            f"Model: {model_version.name}"
        )

        print(
            f"Version: {model_version.version}"
        )

        print(
            f"Source run: {run.info.run_id}"
        )

    print("\n" + "=" * 80)
    print("SIGNATURE-AWARE MODEL REGISTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()