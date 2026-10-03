from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "olist"
GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

TRAINING_PATH = GOLD_DIR / "retention_training_table.parquet"
ORDERS_PATH = SILVER_DIR / "orders.parquet"
CUSTOMERS_PATH = SILVER_DIR / "customers.parquet"


HORIZON_DAYS = 90


def main():

    print("=" * 80)
    print("NEXUS — Retention Training Table Audit")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    training = pd.read_parquet(TRAINING_PATH)
    orders = pd.read_parquet(ORDERS_PATH)
    customers = pd.read_parquet(CUSTOMERS_PATH)

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    # Attach persistent customer identity.
    orders = orders.merge(
        customers[
            ["customer_id", "customer_unique_id"]
        ].drop_duplicates(),
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    print("\nLoaded:")
    print(f"Training rows: {len(training):,}")
    print(f"Order rows: {len(orders):,}")

    # -------------------------------------------------------------------------
    # A. Grain audit
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("A. GRAIN AUDIT")
    print("-" * 80)

    duplicate_keys = training.duplicated(
        subset=[
            "customer_unique_id",
            "snapshot_timestamp",
        ]
    ).sum()

    print(
        f"Duplicate customer × snapshot rows: {duplicate_keys}"
    )

    assert duplicate_keys == 0

    # -------------------------------------------------------------------------
    # B. Target reconstruction
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("B. TARGET RECONSTRUCTION")
    print("-" * 80)

    audit = training[
        [
            "customer_unique_id",
            "snapshot_timestamp",
            "repeat_purchase_90d",
        ]
    ].copy()

    audit["snapshot_timestamp"] = pd.to_datetime(
        audit["snapshot_timestamp"]
    )

    reconstructed_targets = []

    for snapshot in audit[
        "snapshot_timestamp"
    ].drop_duplicates().sort_values():

        future_end = (
            snapshot
            + pd.Timedelta(days=HORIZON_DAYS)
        )

        future_orders = orders[
            (orders["order_purchase_timestamp"] > snapshot)
            & (
                orders["order_purchase_timestamp"]
                <= future_end
            )
        ]

        future_customers = set(
            future_orders[
                "customer_unique_id"
            ].dropna().unique()
        )

        snapshot_mask = (
            audit["snapshot_timestamp"]
            == snapshot
        )

        audit.loc[
            snapshot_mask,
            "reconstructed_target",
        ] = (
            audit.loc[
                snapshot_mask,
                "customer_unique_id",
            ]
            .isin(future_customers)
            .astype("int8")
        )

    audit["reconstructed_target"] = (
        audit["reconstructed_target"]
        .astype("int8")
    )

    mismatches = (
        audit["repeat_purchase_90d"]
        != audit["reconstructed_target"]
    ).sum()

    print(
        f"Target mismatches: {mismatches:,}"
    )

    assert mismatches == 0

    print("PASS — target independently reconstructed.")

    # -------------------------------------------------------------------------
    # C. Temporal feature sanity
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("C. TEMPORAL SANITY")
    print("-" * 80)

    print(
        "Recency minimum:",
        training["recency_days"].min(),
    )

    print(
        "Recency maximum:",
        training["recency_days"].max(),
    )

    negative_recency = (
        training["recency_days"] < 0
    ).sum()

    print(
        f"Negative recency rows: {negative_recency:,}"
    )

    assert negative_recency == 0

    # -------------------------------------------------------------------------
    # D. Target distribution by snapshot
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("D. TARGET DISTRIBUTION BY SNAPSHOT")
    print("-" * 80)

    snapshot_stats = (
        training.groupby(
            "snapshot_timestamp"
        )
        .agg(
            customers=(
                "customer_unique_id",
                "count",
            ),
            positive_targets=(
                "repeat_purchase_90d",
                "sum",
            ),
            positive_rate=(
                "repeat_purchase_90d",
                "mean",
            ),
        )
        .reset_index()
    )

    snapshot_stats["positive_rate_pct"] = (
        snapshot_stats["positive_rate"] * 100
    )

    print(
        snapshot_stats[
            [
                "snapshot_timestamp",
                "customers",
                "positive_targets",
                "positive_rate_pct",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # E. Target class balance
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("E. CLASS BALANCE")
    print("-" * 80)

    positives = int(
        training["repeat_purchase_90d"].sum()
    )

    total = len(training)

    negatives = total - positives

    print(f"Positive: {positives:,}")
    print(f"Negative: {negatives:,}")
    print(
        f"Positive rate: {positives / total * 100:.4f}%"
    )

    print(
        f"Negative : Positive ratio: "
        f"{negatives / positives:.2f}:1"
    )

    # -------------------------------------------------------------------------
    # F. Feature missingness
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("F. FEATURE MISSINGNESS")
    print("-" * 80)

    missing = (
        training.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_pct = (
        missing / len(training) * 100
    )

    missing_report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": missing_pct,
        }
    )

    print(
        missing_report[
            missing_report["missing_count"] > 0
        ].to_string()
    )

    # -------------------------------------------------------------------------
    # G. Basic numeric sanity
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("G. NUMERIC SANITY")
    print("-" * 80)

    numeric_features = [
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
        "orders_60d",
        "orders_90d",
        "spend_30d",
        "spend_60d",
        "spend_90d",
    ]

    for column in numeric_features:

        if column not in training.columns:
            continue

        negative_count = (
            training[column] < 0
        ).sum()

        print(
            f"{column:<35} "
            f"negative={negative_count:,}"
        )

        assert negative_count == 0

    # -------------------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RETENTION TRAINING TABLE AUDIT COMPLETE")
    print("=" * 80)

    print(
        "\nPASS — target, temporal constraints, grain, "
        "and numerical sanity checks passed."
    )


if __name__ == "__main__":
    main()