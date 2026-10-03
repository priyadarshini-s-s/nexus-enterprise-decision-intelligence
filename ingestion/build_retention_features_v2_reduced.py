from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

INPUT_PATH = (
    GOLD_DIR / "retention_training_table_v2.parquet"
)

OUTPUT_PATH = (
    GOLD_DIR / "retention_training_table_v2_reduced.parquet"
)


TARGET = "repeat_purchase_90d"


# Deliberately compact feature representation.
#
# The full V2 audit showed strong redundancy among:
# - short-window share features
# - share vs velocity features
# - recent_activity_90d vs orders_90d_share
#
# Therefore we retain one representative feature from those groups.

V2_REDUCED_FEATURES = [
    # Historical customer behavior
    "orders_to_date",
    "total_spend_to_date",
    "unique_products_to_date",
    "unique_categories_to_date",
    "unique_sellers_to_date",
    "recency_days",
    "average_review_score_to_date",

    # Recent behavior
    "orders_90d",
    "spend_90d",

    # Recent activity concentration
    "orders_90d_share",

    # Recent monetary velocity
    "spend_90d_velocity",

    # Economic behavior
    "spend_per_order_item",
]


def main():

    print("=" * 80)
    print("NEXUS — Retention Feature Engineering V2 Reduced")
    print("=" * 80)

    df = pd.read_parquet(INPUT_PATH)

    print(
        f"\nInput rows: {len(df):,}"
    )

    missing_columns = [
        column
        for column in V2_REDUCED_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_columns)
        )

    # -------------------------------------------------------------------------
    # Preserve the original identifiers, temporal information and target.
    # -------------------------------------------------------------------------

    metadata_columns = [
        "customer_unique_id",
        "snapshot_timestamp",
        TARGET,
    ]

    missing_metadata = [
        column
        for column in metadata_columns
        if column not in df.columns
    ]

    if missing_metadata:
        raise ValueError(
            "Missing required metadata columns: "
            + ", ".join(missing_metadata)
        )

    output_columns = (
        metadata_columns
        + V2_REDUCED_FEATURES
    )

    reduced = df[output_columns].copy()

    # -------------------------------------------------------------------------
    # Validate grain
    # -------------------------------------------------------------------------

    duplicate_keys = reduced.duplicated(
        subset=[
            "customer_unique_id",
            "snapshot_timestamp",
        ]
    ).sum()

    print(
        f"Duplicate customer × snapshot rows: "
        f"{duplicate_keys:,}"
    )

    assert duplicate_keys == 0

    # -------------------------------------------------------------------------
    # Validate target
    # -------------------------------------------------------------------------

    assert reduced[TARGET].isin([0, 1]).all()

    print(
        f"Positive targets: "
        f"{int(reduced[TARGET].sum()):,}"
    )

    print(
        f"Positive rate: "
        f"{reduced[TARGET].mean():.6%}"
    )

    # -------------------------------------------------------------------------
    # Validate feature values
    # -------------------------------------------------------------------------

    missing_values = (
        reduced[V2_REDUCED_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values: "
        f"{missing_values:,}"
    )

    assert missing_values == 0

    negative_values = (
        reduced[V2_REDUCED_FEATURES]
        .select_dtypes(include="number")
        .lt(0)
        .sum()
        .sum()
    )

    print(
        f"Negative feature values: "
        f"{negative_values:,}"
    )

    assert negative_values == 0

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    reduced.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("V2 REDUCED FEATURE SET COMPLETE")
    print("=" * 80)

    print(
        f"Rows: {len(reduced):,}"
    )

    print(
        f"Feature count: "
        f"{len(V2_REDUCED_FEATURES)}"
    )

    print("\nFeatures:")

    for index, feature in enumerate(
        V2_REDUCED_FEATURES,
        start=1,
    ):
        print(
            f"  {index:02d}. {feature}"
        )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )

    print(
        "\nPASS — V2 reduced preserves the "
        "validated customer × snapshot grain and target."
    )


if __name__ == "__main__":
    main()