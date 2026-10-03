from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

INPUT_PATH = (
    GOLD_DIR / "retention_training_table.parquet"
)

OUTPUT_PATH = (
    GOLD_DIR / "retention_training_table_v2.parquet"
)


TARGET = "repeat_purchase_90d"


def safe_ratio(
    numerator,
    denominator,
):
    """
    Ratio with explicit zero-denominator handling.

    A zero denominator means the behavior represented by the
    numerator has no historical denominator. We encode this as 0
    for modeling and retain an explicit flag where useful.
    """

    return np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


def main():

    print("=" * 80)
    print("NEXUS — Retention Feature Engineering V2")
    print("=" * 80)

    df = pd.read_parquet(
        INPUT_PATH
    )

    df["snapshot_timestamp"] = pd.to_datetime(
        df["snapshot_timestamp"]
    )

    print(
        f"\nInput rows: {len(df):,}"
    )

    # =========================================================================
    # 1. Historical activity ratios
    # =========================================================================

    df["orders_30d_share"] = safe_ratio(
        df["orders_30d"],
        df["orders_to_date"],
    )

    df["orders_60d_share"] = safe_ratio(
        df["orders_60d"],
        df["orders_to_date"],
    )

    df["orders_90d_share"] = safe_ratio(
        df["orders_90d"],
        df["orders_to_date"],
    )

    # =========================================================================
    # 2. Recent spend concentration
    # =========================================================================

    df["spend_30d_share"] = safe_ratio(
        df["spend_30d"],
        df["total_spend_to_date"],
    )

    df["spend_60d_share"] = safe_ratio(
        df["spend_60d"],
        df["total_spend_to_date"],
    )

    df["spend_90d_share"] = safe_ratio(
        df["spend_90d"],
        df["total_spend_to_date"],
    )

    # =========================================================================
    # 3. Activity velocity
    # =========================================================================

    df["orders_30d_velocity"] = (
        df["orders_30d"] / 30.0
    )

    df["orders_60d_velocity"] = (
        df["orders_60d"] / 60.0
    )

    df["orders_90d_velocity"] = (
        df["orders_90d"] / 90.0
    )

    # =========================================================================
    # 4. Spend velocity
    # =========================================================================

    df["spend_30d_velocity"] = (
        df["spend_30d"] / 30.0
    )

    df["spend_60d_velocity"] = (
        df["spend_60d"] / 60.0
    )

    df["spend_90d_velocity"] = (
        df["spend_90d"] / 90.0
    )

    # =========================================================================
    # 5. Economic behavior
    # =========================================================================

    df["spend_per_order_item"] = safe_ratio(
        df["total_spend_to_date"],
        df["total_order_items_to_date"],
    )

    # =========================================================================
    # 6. Recent activity indicator
    # =========================================================================

    df["recent_activity_90d"] = (
        df["orders_90d"] > 0
    ).astype("int8")

    # =========================================================================
    # 7. Historical customer span
    # =========================================================================
    #
    # The existing training table does not contain the customer's first
    # purchase timestamp. We therefore deliberately DO NOT manufacture
    # "customer age" from full-history data.
    #
    # Instead, this V2 keeps the feature set leakage-safe and leaves
    # customer maturity for a later point-in-time feature-engineering phase.
    # =========================================================================

    # =========================================================================
    # 8. Define V2 model features
    # =========================================================================

    v2_features = [
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

    # =========================================================================
    # 9. Validation
    # =========================================================================

    print("\n" + "-" * 80)
    print("V2 FEATURE VALIDATION")
    print("-" * 80)

    print(
        f"V2 feature count: {len(v2_features)}"
    )

    missing_columns = [
        column
        for column in v2_features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing V2 features: "
            + ", ".join(missing_columns)
        )

    missing_values = (
        df[v2_features]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing V2 feature values: "
        f"{missing_values:,}"
    )

    assert missing_values == 0

    # No derived feature may be negative.
    negative_counts = (
        (df[v2_features] < 0)
        .sum()
        .sum()
    )

    print(
        f"Negative V2 feature values: "
        f"{negative_counts:,}"
    )

    assert negative_counts == 0

    # Target must remain unchanged.
    assert df[TARGET].isin([0, 1]).all()

    # Grain must remain unchanged.
    duplicate_keys = df.duplicated(
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

    # =========================================================================
    # 10. Save
    # =========================================================================

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("V2 FEATURE ENGINEERING COMPLETE")
    print("=" * 80)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Features: {len(v2_features)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "\nPASS — V2 features created without changing "
        "the validated target or temporal grain."
    )


if __name__ == "__main__":
    main()