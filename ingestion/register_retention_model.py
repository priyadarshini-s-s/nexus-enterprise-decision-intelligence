from pathlib import Path

import mlflow
from mlflow import MlflowClient


# =============================================================================
# NEXUS — Register Retention Model
# =============================================================================

EXPERIMENT_NAME = "NEXUS-Retention-Intelligence"

RUN_ID = "e4fc80077f5e407db7a03c6bcac94d19"

MODEL_NAME = "logistic_retention"

MODEL_ARTIFACT_PATH = "model"


def main():

    print("=" * 80)
    print("NEXUS — MLflow Model Registration")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Connect to the local MLflow tracking store
    # -------------------------------------------------------------------------

    client = MlflowClient()

    # -------------------------------------------------------------------------
    # 2. Verify the run exists
    # -------------------------------------------------------------------------

    run = client.get_run(RUN_ID)

    print("\nSOURCE RUN")
    print("-" * 80)

    print(
        f"Experiment ID: {run.info.experiment_id}"
    )

    print(
        f"Run ID: {run.info.run_id}"
    )

    print(
        f"Run name: {run.data.tags.get('mlflow.runName')}"
    )

    print(
        f"Status: {run.info.status}"
    )

    # -------------------------------------------------------------------------
    # 3. Create registered model if it doesn't already exist
    # -------------------------------------------------------------------------

    try:

        registered_model = (
            client
            .get_registered_model(
                MODEL_NAME
            )
        )

        print(
            f"\nRegistered model already exists: "
            f"{registered_model.name}"
        )

    except Exception:

        registered_model = (
            client
            .create_registered_model(
                MODEL_NAME,
                description=(
                    "NEXUS retention propensity model. "
                    "Predicts probability of repeat purchase "
                    "within 90 days using point-in-time "
                    "Olist customer behavioral features."
                ),
            )
        )

        print(
            f"\nCreated registered model: "
            f"{registered_model.name}"
        )

    # -------------------------------------------------------------------------
    # 4. Register the model artifact from the MLflow run
    # -------------------------------------------------------------------------

    model_uri = (
        f"runs:/{RUN_ID}/{MODEL_ARTIFACT_PATH}"
    )

    print(
        f"\nModel URI:\n{model_uri}"
    )

    model_version = (
        client
        .create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=RUN_ID,
        )
    )

    print("\nMODEL VERSION")
    print("-" * 80)

    print(
        f"Model: {model_version.name}"
    )

    print(
        f"Version: {model_version.version}"
    )

    print(
        f"Run ID: {model_version.run_id}"
    )

    print(
        f"Source: {model_version.source}"
    )

    # -------------------------------------------------------------------------
    # 5. Add useful metadata
    # -------------------------------------------------------------------------

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
        "training_strategy",
        "temporal_split",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        model_version.version,
        "class_weight",
        "None",
    )

    # -------------------------------------------------------------------------
    # 6. Add registered-model metadata
    # -------------------------------------------------------------------------

    client.set_registered_model_tag(
        MODEL_NAME,
        "domain",
        "customer_retention",
    )

    client.set_registered_model_tag(
        MODEL_NAME,
        "business_use",
        "retention_propensity",
    )

    print("\n" + "=" * 80)
    print("MODEL REGISTRATION COMPLETE")
    print("=" * 80)

    print(
        f"\nRegistered model: {MODEL_NAME}"
    )

    print(
        f"Version: {model_version.version}"
    )

    print(
        "\nThe model is now versioned in MLflow."
    )

    print(
        "No deployment status has been assigned yet."
    )


if __name__ == "__main__":
    main()