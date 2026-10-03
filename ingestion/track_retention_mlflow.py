from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# =============================================================================
# NEXUS — Retention Intelligence
# MLflow experiment tracking
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

MODEL_PATH = (
    GOLD_DIR
    / "retention_models"
    / "logistic_retention_unweighted.joblib"
)


EXPERIMENT_NAME = "NEXUS-Retention-Intelligence"


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


def main():

    print("=" * 80)
    print("NEXUS — MLflow Retention Experiment")
    print("=" * 80)

    # =========================================================================
    # 1. Validate model artifact
    # =========================================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found:\n{MODEL_PATH}\n\n"
            "Run train_retention_logistic_unweighted.py first."
        )

    pipeline = joblib.load(
        MODEL_PATH
    )

    if not hasattr(
        pipeline,
        "predict_proba",
    ):
        raise TypeError(
            "Saved model does not expose predict_proba()."
        )

    # =========================================================================
    # 2. Load temporal splits
    # =========================================================================

    train = pd.read_parquet(
        SPLIT_DIR / "train.parquet"
    )

    validation = pd.read_parquet(
        SPLIT_DIR / "validation.parquet"
    )

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    # =========================================================================
    # 3. Prepare matrices
    # =========================================================================

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # =========================================================================
    # 4. Generate predictions
    # =========================================================================

    validation_probability = (
        pipeline
        .predict_proba(X_validation)[:, 1]
    )

    test_probability = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )

    # =========================================================================
    # 5. Calculate metrics
    # =========================================================================

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_probability,
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probability,
    )

    validation_brier = brier_score_loss(
        y_validation,
        validation_probability,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probability,
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probability,
    )

    test_brier = brier_score_loss(
        y_test,
        test_probability,
    )

    # =========================================================================
    # 6. MLflow experiment
    # =========================================================================

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="retention_logistic_unweighted_v1"
    ):

        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------

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
            TARGET,
        )

        mlflow.log_param(
            "prediction_horizon_days",
            90,
        )

        mlflow.log_param(
            "class_weight",
            "None",
        )

        mlflow.log_param(
            "solver",
            "lbfgs",
        )

        mlflow.log_param(
            "max_iter",
            2000,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_param(
            "train_rows",
            len(train),
        )

        mlflow.log_param(
            "validation_rows",
            len(validation),
        )

        mlflow.log_param(
            "test_rows",
            len(test),
        )

        mlflow.log_param(
            "train_positive_rate",
            float(y_train.mean()),
        )

        mlflow.log_param(
            "validation_positive_rate",
            float(y_validation.mean()),
        )

        mlflow.log_param(
            "test_positive_rate",
            float(y_test.mean()),
        )

        # ---------------------------------------------------------------------
        # Validation metrics
        # ---------------------------------------------------------------------

        mlflow.log_metric(
            "validation_roc_auc",
            validation_roc_auc,
        )

        mlflow.log_metric(
            "validation_pr_auc",
            validation_pr_auc,
        )

        mlflow.log_metric(
            "validation_brier_score",
            validation_brier,
        )

        # ---------------------------------------------------------------------
        # Test metrics
        # ---------------------------------------------------------------------

        mlflow.log_metric(
            "test_roc_auc",
            test_roc_auc,
        )

        mlflow.log_metric(
            "test_pr_auc",
            test_pr_auc,
        )

        mlflow.log_metric(
            "test_brier_score",
            test_brier,
        )

        # ---------------------------------------------------------------------
        # Feature metadata
        # ---------------------------------------------------------------------

        feature_file = (
            PROJECT_ROOT
            / "docs"
            / "retention_features_v1.txt"
        )

        feature_file.write_text(
            "\n".join(FEATURES),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(feature_file),
            artifact_path="metadata",
        )

        # ---------------------------------------------------------------------
        # Experiment documentation
        # ---------------------------------------------------------------------

        experiment_doc = (
            PROJECT_ROOT
            / "docs"
            / "retention_experiment_v1.md"
        )

        if experiment_doc.exists():

            mlflow.log_artifact(
                str(experiment_doc),
                artifact_path="documentation",
            )

        # ---------------------------------------------------------------------
        # Model artifact
        # ---------------------------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
        )

        print("\nMLFLOW RUN")
        print("-" * 80)

        print(
            f"Experiment: {EXPERIMENT_NAME}"
        )

        print(
            f"Validation ROC-AUC: "
            f"{validation_roc_auc:.6f}"
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_pr_auc:.6f}"
        )

        print(
            f"Validation Brier: "
            f"{validation_brier:.8f}"
        )

        print(
            f"Test ROC-AUC: "
            f"{test_roc_auc:.6f}"
        )

        print(
            f"Test PR-AUC: "
            f"{test_pr_auc:.6f}"
        )

        print(
            f"Test Brier: "
            f"{test_brier:.8f}"
        )

        print(
            f"\nRun ID: "
            f"{mlflow.active_run().info.run_id}"
        )

    print("\n" + "=" * 80)
    print("MLFLOW RETENTION EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()