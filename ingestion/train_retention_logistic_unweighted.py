from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# NEXUS — Retention Logistic Regression — Unweighted Candidate
# =============================================================================
#
# This script creates the canonical unweighted Logistic Regression artifact
# identified during the class-weight benchmark.
#
# Training:
#   - V1 feature representation
#   - Temporal train split only
#   - StandardScaler
#   - LogisticRegression
#   - class_weight=None
#
# IMPORTANT:
#   Validation and test data are NOT used during fitting.
#
# Model purpose:
#   Calibrated retention propensity + customer ranking
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

MODEL_DIR = (
    GOLD_DIR
    / "retention_models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / "logistic_retention_unweighted.joblib"
)


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
    print("NEXUS — Retention Logistic Regression — Unweighted")
    print("=" * 80)

    # =========================================================================
    # 1. Load TRAIN only
    # =========================================================================

    train_path = (
        SPLIT_DIR
        / "train.parquet"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found:\n{train_path}"
        )

    train = pd.read_parquet(
        train_path
    )

    print("\nTRAINING DATA")
    print("-" * 80)

    print(
        f"Rows: {len(train):,}"
    )

    print(
        f"Positive observations: "
        f"{int(train[TARGET].sum()):,}"
    )

    print(
        f"Positive rate: "
        f"{train[TARGET].mean():.6%}"
    )

    # =========================================================================
    # 2. Validate feature availability
    # =========================================================================

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in train.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing training features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # =========================================================================
    # 3. Validate missing values
    # =========================================================================

    X_train = train[FEATURES]
    y_train = train[TARGET]

    missing_values = (
        X_train
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values: "
        f"{missing_values:,}"
    )

    if missing_values != 0:
        raise ValueError(
            "Training features contain missing values."
        )

    # =========================================================================
    # 4. Validate target
    # =========================================================================

    if not y_train.isin([0, 1]).all():
        raise ValueError(
            f"{TARGET} must contain only 0/1."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training target contains only one class."
        )

    # =========================================================================
    # 5. Fit scaler using TRAIN only
    # =========================================================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    # =========================================================================
    # 6. Train unweighted Logistic Regression
    # =========================================================================
    #
    # class_weight=None is intentional.
    #
    # This is the exact configuration that produced the benchmark result:
    #
    # Test ROC-AUC: 0.615269
    # Test PR-AUC:  0.015453
    #
    # The model also showed substantially better calibration than the balanced
    # alternative.
    # =========================================================================

    model = LogisticRegression(
        class_weight=None,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    print("\nTRAINING")
    print("-" * 80)

    print(
        "Algorithm: Logistic Regression"
    )

    print(
        "class_weight: None"
    )

    print(
        "solver: lbfgs"
    )

    print(
        "max_iter: 2000"
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    # =========================================================================
    # 7. Build deployment-safe Pipeline
    # =========================================================================
    #
    # We save the scaler and model together so inference uses exactly the same
    # preprocessing as training.
    #
    # A Pipeline is preferable to separately saving preprocessing logic because
    # it reduces the risk of training/serving skew.
    # =========================================================================

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                scaler,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # =========================================================================
    # 8. Save artifact
    # =========================================================================

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    # =========================================================================
    # 9. Verify saved artifact
    # =========================================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact was not created:\n{MODEL_PATH}"
        )

    loaded_pipeline = joblib.load(
        MODEL_PATH
    )

    if not hasattr(
        loaded_pipeline,
        "predict_proba",
    ):
        raise TypeError(
            "Saved artifact does not expose predict_proba()."
        )

    # =========================================================================
    # 10. Verify train predictions can be generated
    # =========================================================================

    train_probabilities = (
        loaded_pipeline
        .predict_proba(X_train)
        [:, 1]
    )

    print("\nARTIFACT VALIDATION")
    print("-" * 80)

    print(
        f"Saved artifact: {MODEL_PATH}"
    )

    print(
        f"Predictions generated: "
        f"{len(train_probabilities):,}"
    )

    print(
        f"Minimum score: "
        f"{train_probabilities.min():.8f}"
    )

    print(
        f"Maximum score: "
        f"{train_probabilities.max():.8f}"
    )

    print(
        f"Median score: "
        f"{pd.Series(train_probabilities).median():.8f}"
    )

    print("\n" + "=" * 80)
    print("UNWEIGHTED RETENTION MODEL ARTIFACT COMPLETE")
    print("=" * 80)

    print(
        f"Output: {MODEL_PATH}"
    )

    print(
        "\nTraining used TRAIN data only."
    )

    print(
        "Validation and TEST data were not used for fitting."
    )


if __name__ == "__main__":
    main()