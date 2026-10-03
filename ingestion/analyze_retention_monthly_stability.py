from pathlib import Path

import pandas as pd
import joblib

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

SPLIT_DIR = GOLD_DIR / "retention_splits"

MODEL_PATH = (
    GOLD_DIR
    / "retention_models"
    / "logistic_retention_unweighted.joblib"
)

OUTPUT_PATH = (
    GOLD_DIR
    / "retention_monthly_stability.parquet"
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


def main():

    print("=" * 80)
    print("NEXUS — Retention Monthly Stability Analysis")
    print("=" * 80)

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    test["snapshot_timestamp"] = pd.to_datetime(
        test["snapshot_timestamp"]
    )

    artifact = joblib.load(
        MODEL_PATH
    )

    if hasattr(
        artifact,
        "predict_proba",
    ):
        pipeline = artifact
    elif isinstance(
        artifact,
        dict,
    ):
        pipeline = artifact["model"]
    else:
        raise TypeError(
            f"Unsupported model artifact: {type(artifact)}"
        )

    test["score"] = pipeline.predict_proba(
        test[FEATURES]
    )[:, 1]

    test["month"] = (
        test["snapshot_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    results = []

    for month, group in test.groupby(
        "month",
        sort=True,
    ):

        y = group[TARGET]

        # AUC requires both classes.
        if y.nunique() < 2:
            print(
                f"\n{month}: skipped — only one class"
            )
            continue

        scores = group["score"]

        roc_auc = roc_auc_score(
            y,
            scores,
        )

        pr_auc = average_precision_score(
            y,
            scores,
        )

        brier = brier_score_loss(
            y,
            scores,
        )

        fraction_positive, mean_predicted = (
            calibration_curve(
                y,
                scores,
                n_bins=5,
                strategy="quantile",
            )
        )

        calibration_gap = (
            abs(
                mean_predicted
                - fraction_positive
            ).mean()
        )

        results.append(
            {
                "month": month,
                "rows": len(group),
                "positive_count": int(y.sum()),
                "positive_rate": y.mean(),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "mean_absolute_calibration_gap":
                    calibration_gap,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    print("\nMONTHLY RESULTS")
    print("-" * 80)

    print(
        result_df.to_string(
            index=False,
            formatters={
                "positive_rate":
                    "{:.6%}".format,
                "roc_auc":
                    "{:.6f}".format,
                "pr_auc":
                    "{:.6f}".format,
                "brier_score":
                    "{:.8f}".format,
                "mean_absolute_calibration_gap":
                    "{:.6%}".format,
            },
        )
    )

    result_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("MONTHLY STABILITY ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()