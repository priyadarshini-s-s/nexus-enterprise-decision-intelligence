from pathlib import Path

import joblib
import pandas as pd
import numpy as np


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


MODEL_PATH = (
    MODEL_DIR
    / "logistic_retention_baseline.joblib"
)


def main():

    print("=" * 80)
    print("NEXUS — Logistic Regression Feature Audit")
    print("=" * 80)

    model = joblib.load(
        MODEL_PATH
    )

    train = pd.read_parquet(
        SPLIT_DIR / "train.parquet"
    )

    # -------------------------------------------------------------------------
    # Extract fitted classifier
    # -------------------------------------------------------------------------

    classifier = (
        model.named_steps["classifier"]
    )

    coefficients = (
        classifier.coef_[0]
    )

    coefficient_table = pd.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": coefficients,
        }
    )

    coefficient_table["abs_coefficient"] = (
        coefficient_table["coefficient"]
        .abs()
    )

    coefficient_table["direction"] = np.where(
        coefficient_table["coefficient"] > 0,
        "positive",
        "negative",
    )

    coefficient_table = (
        coefficient_table
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Odds ratio
    # -------------------------------------------------------------------------

    coefficient_table["odds_ratio"] = np.exp(
        coefficient_table["coefficient"]
    )

    print("\n" + "-" * 80)
    print("STANDARDIZED LOGISTIC COEFFICIENTS")
    print("-" * 80)

    print(
        coefficient_table[
            [
                "feature",
                "coefficient",
                "odds_ratio",
                "direction",
            ]
        ].to_string(
            index=False,
            formatters={
                "coefficient": "{:.6f}".format,
                "odds_ratio": "{:.4f}".format,
            },
        )
    )

    # -------------------------------------------------------------------------
    # Correlation analysis
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("FEATURE CORRELATION AUDIT")
    print("-" * 80)

    numeric_data = train[
        FEATURES
    ].select_dtypes(
        include="number"
    )

    correlation = numeric_data.corr()

    pairs = []

    for i in range(len(correlation.columns)):

        for j in range(i + 1, len(correlation.columns)):

            feature_a = correlation.columns[i]
            feature_b = correlation.columns[j]

            value = correlation.iloc[i, j]

            if abs(value) >= 0.80:

                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": value,
                    }
                )

    correlation_pairs = pd.DataFrame(
        pairs
    )

    if correlation_pairs.empty:

        print(
            "No feature pairs with "
            "|correlation| >= 0.80."
        )

    else:

        correlation_pairs = (
            correlation_pairs
            .sort_values(
                "correlation",
                key=lambda s: s.abs(),
                ascending=False,
            )
        )

        print(
            correlation_pairs.to_string(
                index=False,
                formatters={
                    "correlation": "{:.4f}".format,
                },
            )
        )

    # -------------------------------------------------------------------------
    # Feature summary
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("FEATURE DISTRIBUTION SUMMARY")
    print("-" * 80)

    summary = train[
        FEATURES
    ].describe().T

    summary["missing"] = (
        train[FEATURES].isna().sum()
    )

    print(
        summary[
            [
                "mean",
                "std",
                "min",
                "50%",
                "max",
                "missing",
            ]
        ].to_string(
            formatters={
                "mean": "{:.4f}".format,
                "std": "{:.4f}".format,
                "min": "{:.4f}".format,
                "50%": "{:.4f}".format,
                "max": "{:.4f}".format,
            }
        )
    )

    # -------------------------------------------------------------------------
    # Save coefficient report
    # -------------------------------------------------------------------------

    output_path = (
        GOLD_DIR
        / "retention_logistic_feature_coefficients.parquet"
    )

    coefficient_table.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FEATURE AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"Coefficient report: {output_path}"
    )


if __name__ == "__main__":
    main()