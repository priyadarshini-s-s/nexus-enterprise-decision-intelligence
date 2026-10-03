from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"
SPLIT_DIR = GOLD_DIR / "retention_splits"
MODEL_DIR = GOLD_DIR / "retention_models"


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


def evaluate_ranking(model, df, name):

    X = df[FEATURES]
    y = df[TARGET]

    probabilities = model.predict_proba(X)[:, 1]

    evaluation = pd.DataFrame(
        {
            "actual": y.to_numpy(),
            "score": probabilities,
        }
    )

    evaluation = evaluation.sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)

    total_customers = len(evaluation)
    total_positives = evaluation["actual"].sum()

    baseline_rate = (
        total_positives / total_customers
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Customers: {total_customers:,}"
    )

    print(
        f"Actual repeat purchasers: "
        f"{int(total_positives):,}"
    )

    print(
        f"Baseline repeat rate: "
        f"{baseline_rate * 100:.4f}%"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y, probabilities):.6f}"
    )

    print(
        f"PR-AUC: "
        f"{average_precision_score(y, probabilities):.6f}"
    )

    print("\nTop-K ranking performance")

    rows = []

    for percentage in [
        1,
        2,
        5,
        10,
        20,
    ]:

        n = max(
            1,
            int(len(evaluation) * percentage / 100),
        )

        selected = evaluation.iloc[:n]

        captured = selected["actual"].sum()

        precision = (
            captured / n
        )

        recall = (
            captured / total_positives
            if total_positives > 0
            else 0
        )

        lift = (
            precision / baseline_rate
            if baseline_rate > 0
            else 0
        )

        rows.append(
            {
                "top_pct": percentage,
                "customers_targeted": n,
                "repeat_purchasers_captured": int(
                    captured
                ),
                "precision_pct": precision * 100,
                "recall_pct": recall * 100,
                "lift_vs_random": lift,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "precision_pct": "{:.4f}".format,
                "recall_pct": "{:.4f}".format,
                "lift_vs_random": "{:.3f}".format,
            },
        )
    )

    return result


def main():

    print("=" * 80)
    print("NEXUS — Retention Ranking Evaluation")
    print("=" * 80)

    model_path = (
        MODEL_DIR
        / "logistic_retention_baseline.joblib"
    )

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    model = __import__(
        "joblib"
    ).load(model_path)

    evaluate_ranking(
        model,
        test,
        "TEST — LOGISTIC REGRESSION",
    )


if __name__ == "__main__":
    main()