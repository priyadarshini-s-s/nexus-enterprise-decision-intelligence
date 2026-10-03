from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

INPUT_PATH = (
    GOLD_DIR / "retention_training_table_v2.parquet"
)


FEATURES = [
    "orders_to_date",
    "total_spend_to_date",
    "unique_products_to_date",
    "unique_categories_to_date",
    "unique_sellers_to_date",
    "recency_days",
    "average_review_score_to_date",

    "orders_30d_share",
    "orders_60d_share",
    "orders_90d_share",

    "spend_30d_share",
    "spend_60d_share",
    "spend_90d_share",

    "orders_30d_velocity",
    "orders_60d_velocity",
    "orders_90d_velocity",

    "spend_30d_velocity",
    "spend_60d_velocity",
    "spend_90d_velocity",

    "spend_per_order_item",
    "recent_activity_90d",
]


def main():

    print("=" * 80)
    print("NEXUS — Retention Feature V2 Audit")
    print("=" * 80)

    df = pd.read_parquet(
        INPUT_PATH
    )

    correlation = df[
        FEATURES
    ].corr()

    pairs = []

    for i in range(len(FEATURES)):

        for j in range(i + 1, len(FEATURES)):

            value = correlation.iloc[i, j]

            if abs(value) >= 0.80:

                pairs.append(
                    {
                        "feature_a": FEATURES[i],
                        "feature_b": FEATURES[j],
                        "correlation": value,
                    }
                )

    print("\nHighly correlated V2 feature pairs")
    print(
        "(absolute correlation >= 0.80)"
    )

    if not pairs:

        print(
            "\nNone found."
        )

    else:

        result = (
            pd.DataFrame(pairs)
            .sort_values(
                "correlation",
                key=lambda x: x.abs(),
                ascending=False,
            )
        )

        print(
            result.to_string(
                index=False,
                formatters={
                    "correlation": "{:.4f}".format
                },
            )
        )

    print("\nV2 descriptive statistics")

    print(
        df[FEATURES]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "50%",
                "max",
            ]
        ]
        .to_string()
    )


if __name__ == "__main__":
    main()