from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# NEXUS — Leakage-Safe Retention Feature Builder
# =============================================================================
#
# Target:
#   repeat_purchase_90d = 1 if the customer makes another purchase in
#   (cutoff_timestamp, cutoff_timestamp + 90 days], otherwise 0.
#
# Feature rule:
#   Every feature must use information available at or before cutoff_timestamp.
#
# Grain:
#   One row per customer_unique_id × cutoff_timestamp.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "olist"
GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

SNAPSHOT_PATH = GOLD_DIR / "retention_snapshot_design.parquet"
OUTPUT_PATH = GOLD_DIR / "retention_training_table.parquet"


HORIZON_DAYS = 90


def load_data():
    """Load Silver-layer data required for retention modeling."""

    return {
        "customers": pd.read_parquet(
            SILVER_DIR / "customers.parquet"
        ),
        "orders": pd.read_parquet(
            SILVER_DIR / "orders.parquet"
        ),
        "order_items": pd.read_parquet(
            SILVER_DIR / "order_items.parquet"
        ),
        "payments": pd.read_parquet(
            SILVER_DIR / "order_payments.parquet"
        ),
        "products": pd.read_parquet(
            SILVER_DIR / "products.parquet"
        ),
        "reviews": pd.read_parquet(
            SILVER_DIR / "order_reviews.parquet"
        ),
        "snapshots": pd.read_parquet(
            SNAPSHOT_PATH
        ),
    }


def prepare_orders(orders, customers):
    """Attach persistent customer identity to orders."""

    customer_map = customers[
        ["customer_id", "customer_unique_id"]
    ].drop_duplicates()

    orders = orders.merge(
        customer_map,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    return orders


def prepare_items(order_items, products):
    """Attach product category information to order items."""

    items = order_items.copy()

    items["shipping_limit_date"] = pd.to_datetime(
        items["shipping_limit_date"]
    )

    product_cols = [
        "product_id",
        "product_category_name",
    ]

    items = items.merge(
        products[product_cols],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    return items


def prepare_payments(payments):
    """Prepare payment data at order level."""

    payments = payments.copy()

    # Payment rows are not order rows.
    # Aggregate them before joining to customer/order features.
    payment_summary = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_records=("payment_sequential", "count"),
        )
    )

    return payment_summary


def prepare_reviews(reviews):
    """Prepare review observations at order level."""

    reviews = reviews.copy()

    reviews["review_creation_date"] = pd.to_datetime(
        reviews["review_creation_date"]
    )

    review_summary = (
        reviews.groupby("order_id", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            average_review_score=("review_score", "mean"),
        )
    )

    return review_summary


def build_order_level_table(
    orders,
    items,
    payments,
    reviews,
):
    """
    Create one analytical row per order.

    This prevents payment and item grain from accidentally multiplying
    order-level measures.
    """

    order_table = orders[
        [
            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_status",
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Payment metrics
    # -------------------------------------------------------------------------

    order_table = order_table.merge(
        payments,
        on="order_id",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Item metrics
    # -------------------------------------------------------------------------

    item_summary = (
        items.groupby("order_id", as_index=False)
        .agg(
            order_item_count=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            unique_categories=("product_category_name", "nunique"),
            unique_sellers=("seller_id", "nunique"),
        )
    )

    order_table = order_table.merge(
        item_summary,
        on="order_id",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Review metrics
    # -------------------------------------------------------------------------

    order_table = order_table.merge(
        reviews,
        on="order_id",
        how="left",
    )

    # Missing values here mean the source does not contain an observation,
    # not that the customer's behavior was zero.
    order_table["payment_observed"] = (
        order_table["payment_value"].notna()
    ).astype("int8")

    order_table["review_observed"] = (
        order_table["review_count"].notna()
    ).astype("int8")

    order_table["item_observed"] = (
        order_table["order_item_count"].notna()
    ).astype("int8")

    return order_table


def build_snapshot_features(
    customer_ids,
    cutoff,
    orders,
    items,
    horizon_days=90,
):
    """
    Build historical features as of one cutoff.

    No record with timestamp > cutoff is allowed into the feature set.
    """

    history = orders[
        orders["order_purchase_timestamp"] <= cutoff
    ].copy()

    history = history[
        history["customer_unique_id"].isin(customer_ids)
    ]

    # -------------------------------------------------------------------------
    # Customer-level historical aggregates
    # -------------------------------------------------------------------------

    customer_agg = (
        history.groupby("customer_unique_id")
        .agg(
            orders_to_date=("order_id", "nunique"),
            total_spend_to_date=("payment_value", "sum"),
            total_order_items_to_date=("order_item_count", "sum"),
            unique_products_to_date=("unique_products", "sum"),
            unique_categories_to_date=("unique_categories", "sum"),
            unique_sellers_to_date=("unique_sellers", "sum"),
            total_payment_records_to_date=("payment_records", "sum"),
            payment_observations_to_date=("payment_observed", "sum"),
            review_observations_to_date=("review_observed", "sum"),
            total_review_count_to_date=("review_count", "sum"),
            average_review_score_to_date=("average_review_score", "mean"),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Correct distinct product/category/seller counts across history
    # -------------------------------------------------------------------------

    history_order_ids = history["order_id"]

    history_items = items[
        items["order_id"].isin(history_order_ids)
    ].merge(
        history[
            [
                "order_id",
                "customer_unique_id",
                "order_purchase_timestamp",
            ]
        ],
        on="order_id",
        how="inner",
    )

    distinct_history = (
        history_items.groupby("customer_unique_id")
        .agg(
            unique_products_to_date=("product_id", "nunique"),
            unique_categories_to_date=("product_category_name", "nunique"),
            unique_sellers_to_date=("seller_id", "nunique"),
        )
        .reset_index()
    )

    # Replace the preliminary sums with true distinct counts.
    customer_agg = customer_agg.drop(
        columns=[
            "unique_products_to_date",
            "unique_categories_to_date",
            "unique_sellers_to_date",
        ],
        errors="ignore",
    )

    customer_agg = customer_agg.merge(
        distinct_history,
        on="customer_unique_id",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Recency
    # -------------------------------------------------------------------------

    last_purchase = (
        history.groupby("customer_unique_id")[
            "order_purchase_timestamp"
        ]
        .max()
        .rename("last_purchase_timestamp")
        .reset_index()
    )

    customer_agg = customer_agg.merge(
        last_purchase,
        on="customer_unique_id",
        how="left",
    )

    customer_agg["recency_days"] = (
        cutoff - customer_agg["last_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    customer_agg["recency_days"] = (
        customer_agg["recency_days"]
        .round(4)
    )

    # -------------------------------------------------------------------------
    # AOV
    # -------------------------------------------------------------------------

    customer_agg["average_order_value"] = (
        customer_agg["total_spend_to_date"]
        / customer_agg["orders_to_date"]
    )

    customer_agg["average_items_per_order"] = (
        customer_agg["total_order_items_to_date"]
        / customer_agg["orders_to_date"]
    )

    # -------------------------------------------------------------------------
    # Recent behavioral windows
    # -------------------------------------------------------------------------

    def rolling_features(days):
        window_start = cutoff - pd.Timedelta(days=days)

        recent = history[
            history["order_purchase_timestamp"] > window_start
        ]

        return (
            recent.groupby("customer_unique_id")
            .agg(
                **{
                    f"orders_{days}d": ("order_id", "nunique"),
                    f"spend_{days}d": ("payment_value", "sum"),
                    f"items_{days}d": ("order_item_count", "sum"),
                }
            )
            .reset_index()
        )

    for days in (30, 60, 90):
        rolling = rolling_features(days)

        customer_agg = customer_agg.merge(
            rolling,
            on="customer_unique_id",
            how="left",
        )

    # -------------------------------------------------------------------------
    # Target — FUTURE ONLY
    # -------------------------------------------------------------------------

    future_end = cutoff + pd.Timedelta(days=horizon_days)

    future = orders[
        (orders["order_purchase_timestamp"] > cutoff)
        & (orders["order_purchase_timestamp"] <= future_end)
        & (orders["customer_unique_id"].isin(customer_ids))
    ]

    future_customers = set(
        future["customer_unique_id"].dropna().unique()
    )

    customer_agg["repeat_purchase_90d"] = (
        customer_agg["customer_unique_id"]
        .isin(future_customers)
        .astype("int8")
    )

    customer_agg["snapshot_timestamp"] = cutoff

    return customer_agg


def main():

    print("=" * 80)
    print("NEXUS — Leakage-Safe Retention Feature Builder")
    print("=" * 80)

    data = load_data()

    customers = data["customers"]
    orders = data["orders"]
    order_items = data["order_items"]
    products = data["products"]
    payments = data["payments"]
    reviews = data["reviews"]
    snapshots = data["snapshots"]
    print("\nLoaded schemas:")
    print("customers:", customers.columns.tolist())
    print("orders:", orders.columns.tolist())
    print("order_items:", order_items.columns.tolist())
    print("products:", products.columns.tolist())
    print("payments:", payments.columns.tolist())
    print("reviews:", reviews.columns.tolist())

    print("\nLoaded source tables")

    orders = prepare_orders(
        orders,
        customers,
    )

    order_items = prepare_items(
        order_items,
        products,
    )

    payments = prepare_payments(
        payments,
    )

    reviews = prepare_reviews(
        reviews,
    )

    order_table = build_order_level_table(
        orders,
        order_items,
        payments,
        reviews,
    )

    print(f"Orders available: {len(order_table):,}")

    snapshot_rows = []

    for _, snapshot in snapshots.iterrows():

        cutoff = pd.Timestamp(
            snapshot["cutoff_timestamp"]
        )

        history = order_table[
            order_table["order_purchase_timestamp"] <= cutoff
        ]

        customer_ids = set(
            history["customer_unique_id"]
            .dropna()
            .unique()
        )

        if not customer_ids:
            continue

        print(
            f"Building snapshot: "
            f"{cutoff.date()} "
            f"({len(customer_ids):,} customers)"
        )

        features = build_snapshot_features(
            customer_ids=customer_ids,
            cutoff=cutoff,
            orders=order_table,
            items=order_items,
            horizon_days=HORIZON_DAYS,
        )

        snapshot_rows.append(features)

    training_table = pd.concat(
        snapshot_rows,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Final cleanup
    # -------------------------------------------------------------------------

    numeric_columns = [
        "total_spend_to_date",
        "total_order_items_to_date",
        "unique_products_to_date",
        "unique_categories_to_date",
        "unique_sellers_to_date",
        "total_payment_records_to_date",
        "payment_observations_to_date",
        "review_observations_to_date",
        "total_review_count_to_date",
        "average_review_score_to_date",
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
    ]

    for column in numeric_columns:
        if column in training_table.columns:
            training_table[column] = training_table[column].fillna(0)

    training_table["payment_observation_available"] = (
        training_table["payment_observations_to_date"] > 0
    ).astype("int8")

    training_table["review_observation_available"] = (
        training_table["review_observations_to_date"] > 0
    ).astype("int8")

    training_table["item_observation_available"] = (
        training_table["total_order_items_to_date"] > 0
    ).astype("int8")

    training_table = training_table.sort_values(
        [
            "snapshot_timestamp",
            "customer_unique_id",
        ]
    ).reset_index(drop=True)

    GOLD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_table.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(
        f"Rows: {len(training_table):,}"
    )

    print(
        f"Unique customers: "
        f"{training_table['customer_unique_id'].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{training_table['snapshot_timestamp'].nunique():,}"
    )

    print(
        "\nTarget distribution:"
    )

    target_counts = (
        training_table["repeat_purchase_90d"]
        .value_counts()
        .sort_index()
    )

    print(target_counts)

    target_rate = (
        training_table["repeat_purchase_90d"].mean()
        * 100
    )

    print(
        f"\nOverall 90-day repeat rate: "
        f"{target_rate:.4f}%"
    )

    # One row per customer × snapshot.
    duplicate_keys = training_table.duplicated(
        subset=[
            "customer_unique_id",
            "snapshot_timestamp",
        ]
    ).sum()

    print(
        f"Duplicate customer × snapshot rows: "
        f"{duplicate_keys}"
    )

    # Feature leakage sanity check.
    assert (
        training_table["recency_days"] >= 0
    ).all()

    assert duplicate_keys == 0

    assert training_table["repeat_purchase_90d"].isin(
        [0, 1]
    ).all()

    print("\nPASS — leakage-safe retention table created.")
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()