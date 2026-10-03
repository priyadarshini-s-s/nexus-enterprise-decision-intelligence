from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

INPUT_PATH = (
    GOLD_DIR / "retention_training_table_v2_reduced.parquet"
)

SPLIT_DIR = (
    GOLD_DIR / "retention_splits"
)

MODEL_DIR = (
    GOLD_DIR / "retention_models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR / "logistic_retention_v2_reduced.joblib"
)


TARGET = "repeat_purchase_90d"


FEATURES = [
    "orders_to_date",
    "total_spend_to_date",
    "unique_products_to_date",
    "unique_categories_to_date",
    "unique_sellers_to_date",
    "recency_days",
    "average_review_score_to_date",

    "orders_90d",
    "spend_90d",
    "orders_90d_share",
    "spend_90d_velocity",
    "spend_per_order_item",
]


def evaluate(
    model,
    scaler,
    data,
    split_name,
):

    X = data[FEATURES]
    y = data[TARGET]

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    print(
        f"\n{split_name.upper()}"
    )

    print(
        f"Rows: {len(data):,}"
    )

    print(
        f"Positive rate: {y.mean():.6%}"
    )

    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )

    print(
        f"PR-AUC: {pr_auc:.6f}"
    )

    # -------------------------------------------------------------------------
    # Ranking / lift analysis
    # -------------------------------------------------------------------------

    evaluation = pd.DataFrame(
        {
            "actual": y.to_numpy(),
            "score": probabilities,
        }
    )

    evaluation = evaluation.sort_values(
        "score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    baseline_rate = y.mean()

    print("\nRanking performance:")

    for percentage in [
        1,
        2,
        5,
        10,
        20,
    ]:

        n = max(
            1,
            int(
                len(evaluation)
                * percentage
                / 100
            ),
        )

        top = evaluation.iloc[:n]

        captured = top["actual"].sum()

        precision = (
            captured / len(top)
        )

        recall = (
            captured / y.sum()
            if y.sum() > 0
            else 0
        )

        lift = (
            precision / baseline_rate
            if baseline_rate > 0
            else 0
        )

        print(
            f"Top {percentage:>2}% | "
            f"Targeted={len(top):,} | "
            f"Captured={int(captured):,} | "
            f"Precision={precision:.4%} | "
            f"Recall={recall:.4%} | "
            f"Lift={lift:.3f}x"
        )

    return {
        "split": split_name,
        "rows": len(data),
        "positive_rate": y.mean(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():

    print("=" * 80)
    print("NEXUS — Retention Logistic Regression V2 Reduced")
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

    # IMPORTANT:
    # The original V1 splits contain the same temporal observations.
    # We join the reduced V2 feature representation to those split keys.

    v2 = pd.read_parquet(
        INPUT_PATH
    )

    key_columns = [
        "customer_unique_id",
        "snapshot_timestamp",
    ]

    train = train[key_columns + [TARGET]].merge(
        v2,
        on=key_columns + [TARGET],
        how="inner",
        validate="one_to_one",
    )

    validation = validation[key_columns + [TARGET]].merge(
        v2,
        on=key_columns + [TARGET],
        how="inner",
        validate="one_to_one",
    )

    test = test[key_columns + [TARGET]].merge(
        v2,
        on=key_columns + [TARGET],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"\nTrain rows: {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    # -------------------------------------------------------------------------
    # Prepare matrices
    # -------------------------------------------------------------------------

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # -------------------------------------------------------------------------
    # SAME preprocessing/model family as V1
    # -------------------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    validation_metrics = evaluate(
        model,
        scaler,
        validation,
        "validation",
    )

    test_metrics = evaluate(
        model,
        scaler,
        test,
        "test",
    )

    # -------------------------------------------------------------------------
    # Save model artifact
    # -------------------------------------------------------------------------

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "target": TARGET,
        "version": "v2_reduced",
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print("\n" + "=" * 80)
    print("V2 MODEL TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        "\nValidation ROC-AUC: "
        f"{validation_metrics['roc_auc']:.6f}"
    )

    print(
        "Validation PR-AUC: "
        f"{validation_metrics['pr_auc']:.6f}"
    )

    print(
        "\nTest ROC-AUC: "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        "Test PR-AUC: "
        f"{test_metrics['pr_auc']:.6f}"
    )


if __name__ == "__main__":
    main()