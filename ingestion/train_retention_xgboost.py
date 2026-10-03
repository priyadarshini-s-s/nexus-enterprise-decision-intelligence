from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


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


def ranking_metrics(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    evaluation = pd.DataFrame(
        {
            "actual": y.to_numpy(),
            "score": probabilities,
        }
    ).sort_values(
        "score",
        ascending=False,
    )

    total_positive = evaluation["actual"].sum()

    baseline_rate = (
        total_positive / len(evaluation)
    )

    rows = []

    for pct in [1, 2, 5, 10, 20]:

        n = max(
            1,
            int(len(evaluation) * pct / 100),
        )

        selected = evaluation.iloc[:n]

        captured = selected["actual"].sum()

        precision = captured / n

        recall = (
            captured / total_positive
            if total_positive > 0
            else 0
        )

        lift = (
            precision / baseline_rate
            if baseline_rate > 0
            else 0
        )

        rows.append(
            {
                "top_pct": pct,
                "customers_targeted": n,
                "repeat_purchasers_captured": int(
                    captured
                ),
                "precision_pct": precision * 100,
                "recall_pct": recall * 100,
                "lift_vs_random": lift,
            }
        )

    return (
        roc_auc,
        pr_auc,
        pd.DataFrame(rows),
    )


def evaluate(
    model,
    X,
    y,
    name,
):

    roc_auc, pr_auc, ranking = ranking_metrics(
        model,
        X,
        y,
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.6f}"
    )

    print("\nTop-K ranking performance")

    print(
        ranking.to_string(
            index=False,
            formatters={
                "precision_pct": "{:.4f}".format,
                "recall_pct": "{:.4f}".format,
                "lift_vs_random": "{:.3f}".format,
            },
        )
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ranking": ranking,
    }


def main():

    print("=" * 80)
    print("NEXUS — XGBoost Retention Benchmark")
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

    # -------------------------------------------------------------------------
    # Class imbalance
    # -------------------------------------------------------------------------

    positive_count = y_train.sum()

    negative_count = len(y_train) - positive_count

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nTraining positives: {positive_count:,}"
    )

    print(
        f"Training negatives: {negative_count:,}"
    )

    print(
        f"scale_pos_weight: {scale_pos_weight:.4f}"
    )

    # -------------------------------------------------------------------------
    # Controlled benchmark configuration
    # -------------------------------------------------------------------------

    model = xgb.XGBClassifier(
        objective="binary:logistic",

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.0,

        reg_lambda=1.0,

        scale_pos_weight=scale_pos_weight,

        eval_metric="aucpr",

        tree_method="hist",

        random_state=42,

        n_jobs=-1,
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
        "VALIDATION — XGBOOST",
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
        "TEST — XGBOOST",
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "xgboost_retention_benchmark.joblib"
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

    print("\nBenchmark summary")

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