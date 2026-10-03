from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"
SPLIT_DIR = GOLD_DIR / "retention_splits"
MODEL_DIR = GOLD_DIR / "retention_models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

TARGET = "repeat_purchase_90d"


def evaluate_model(model, X, y, name):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"ROC-AUC:   {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.6f}"
    )

    print(
        f"Precision:  {precision:.6f}"
    )

    print(
        f"Recall:     {recall:.6f}"
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print("\nClassification report:")

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
    }


def main():

    print("=" * 80)
    print("NEXUS — Logistic Regression Retention Baseline")
    print("=" * 80)

    train = pd.read_parquet(
        SPLIT_DIR / "train.parquet"
    )

    validation = pd.read_parquet(
        SPLIT_DIR / "validation.parquet"
    )

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\nFeature count:", len(FEATURES))

    print(
        "Training rows:",
        f"{len(train):,}",
    )

    print(
        "Validation rows:",
        f"{len(validation):,}",
    )

    print(
        "Test rows:",
        f"{len(test):,}",
    )

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                FEATURES,
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Baseline model
    # -------------------------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION RESULTS",
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST RESULTS",
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "logistic_retention_baseline.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print("\n" + "=" * 80)
    print("MODEL SAVED")
    print("=" * 80)

    print(
        f"Path: {model_path}"
    )

    print("\nBaseline summary:")

    print(
        f"Validation ROC-AUC: "
        f"{validation_metrics['roc_auc']:.6f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{validation_metrics['pr_auc']:.6f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['pr_auc']:.6f}"
    )


if __name__ == "__main__":
    main()