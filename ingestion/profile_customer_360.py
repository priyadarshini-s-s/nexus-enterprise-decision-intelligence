from pathlib import Path
import pandas as pd


SILVER_DIR = Path("data/silver/olist")
GOLD_DIR = Path("data/gold/olist")


def load_silver(name):
    return pd.read_parquet(SILVER_DIR / name)


def load_gold(name):
    return pd.read_parquet(GOLD_DIR / name)


def main():

    print("=" * 80)
    print("NEXUS — Customer 360 Gold Profiling")
    print("=" * 80)

    customers = load_silver("customers.parquet")
    orders = load_silver("orders.parquet")
    payments = load_silver("order_payments.parquet")
    items = load_silver("order_items.parquet")
    reviews = load_silver("order_reviews.parquet")

    customer_360 = load_gold("customer_360.parquet")

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    print("\n1. STRUCTURE")
    print("-" * 80)

    print(f"Rows:    {len(customer_360):,}")
    print(f"Columns: {len(customer_360.columns)}")

    print("\nColumns:")

    for column in customer_360.columns:
        print(f"  - {column}")

    # ------------------------------------------------------------------
    # Identity reconciliation
    # ------------------------------------------------------------------

    print("\n2. CUSTOMER IDENTITY")
    print("-" * 80)

    expected_customers = customers[
        "customer_unique_id"
    ].nunique()

    actual_customers = customer_360[
        "customer_unique_id"
    ].nunique()

    print(
        f"Silver unique customers : {expected_customers:,}"
    )

    print(
        f"Gold unique customers   : {actual_customers:,}"
    )

    print(
        "Status:",
        "PASS" if expected_customers == actual_customers else "FAIL"
    )

    # ------------------------------------------------------------------
    # Payment reconciliation
    # ------------------------------------------------------------------

    print("\n3. PAYMENT RECONCILIATION")
    print("-" * 80)

    silver_payment_total = payments[
        "payment_value"
    ].sum()

    gold_payment_total = customer_360[
        "total_payment_value"
    ].sum()

    difference = (
        gold_payment_total
        - silver_payment_total
    )

    print(
        f"Silver payment total : "
        f"{silver_payment_total:,.2f}"
    )

    print(
        f"Gold payment total   : "
        f"{gold_payment_total:,.2f}"
    )

    print(
        f"Difference           : "
        f"{difference:,.2f}"
    )

    tolerance = 0.01

    print(
        "Status:",
        "PASS" if abs(difference) <= tolerance else "FAIL"
    )

    # ------------------------------------------------------------------
    # Order reconciliation
    # ------------------------------------------------------------------

    print("\n4. ORDER RECONCILIATION")
    print("-" * 80)

    order_customer = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    expected_order_counts = (
        order_customer
        .groupby("customer_unique_id")["order_id"]
        .nunique()
    )

    gold_order_counts = customer_360.set_index(
        "customer_unique_id"
    )["total_orders"]

    aligned = expected_order_counts.reindex(
        gold_order_counts.index
    )

    order_differences = (
        aligned - gold_order_counts
    ).abs()

    print(
        f"Customers checked : "
        f"{len(order_differences):,}"
    )

    print(
        f"Mismatched customers : "
        f"{(order_differences > 0).sum():,}"
    )

    print(
        "Status:",
        "PASS"
        if (order_differences > 0).sum() == 0
        else "FAIL"
    )

    # ------------------------------------------------------------------
    # Distribution profiling
    # ------------------------------------------------------------------

    print("\n5. CUSTOMER BEHAVIOR DISTRIBUTIONS")
    print("-" * 80)

    metrics = [
        "total_orders",
        "total_payment_value",
        "average_order_value",
        "total_items",
        "unique_products",
        "unique_categories",
        "unique_sellers",
        "review_count",
        "average_review_score",
        "customer_lifetime_days",
        "order_frequency",
    ]

    print(
        customer_360[metrics]
        .describe()
        .round(3)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Repeat customers
    # ------------------------------------------------------------------

    print("\n6. REPEAT CUSTOMER ANALYSIS")
    print("-" * 80)

    repeat_count = customer_360[
        "is_repeat_customer"
    ].sum()

    total_customers = len(customer_360)

    repeat_rate = (
        repeat_count / total_customers * 100
    )

    print(
        f"Repeat customers : {repeat_count:,}"
    )

    print(
        f"One-time customers: "
        f"{total_customers - repeat_count:,}"
    )

    print(
        f"Repeat customer rate: "
        f"{repeat_rate:.2f}%"
    )

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    print("\n7. GOLD MISSINGNESS")
    print("-" * 80)

    missing = customer_360.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) == 0:
        print("No missing values.")
    else:
        for column, count in missing.items():
            percentage = (
                count / len(customer_360) * 100
            )

            print(
                f"{column:<30}"
                f"{count:>8,}"
                f" ({percentage:>6.2f}%)"
            )

    # ------------------------------------------------------------------
    # Review coverage
    # ------------------------------------------------------------------

    print("\n8. REVIEW COVERAGE")
    print("-" * 80)

    reviewed_customers = (
        customer_360["review_count"] > 0
    ).sum()

    print(
        f"Customers with reviews: "
        f"{reviewed_customers:,}"
    )

    print(
        f"Customers without reviews: "
        f"{total_customers - reviewed_customers:,}"
    )

    # ------------------------------------------------------------------
    # Extreme values
    # ------------------------------------------------------------------

    print("\n9. TOP 10 CUSTOMERS BY PAYMENT VALUE")
    print("-" * 80)

    top_customers = customer_360[
        [
            "customer_unique_id",
            "total_orders",
            "total_payment_value",
            "average_order_value",
        ]
    ].head(10)

    print(
        top_customers.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Negative values
    # ------------------------------------------------------------------

    print("\n10. NUMERIC SANITY CHECKS")
    print("-" * 80)

    numeric_columns = [
        "total_orders",
        "total_payment_value",
        "average_order_value",
        "total_items",
        "unique_products",
        "unique_categories",
        "unique_sellers",
        "review_count",
    ]

    for column in numeric_columns:

        negative_count = (
            customer_360[column] < 0
        ).sum()

        print(
            f"{column:<30}"
            f"negative values: "
            f"{negative_count:,}"
        )

    print("\n" + "=" * 80)
    print("CUSTOMER 360 PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()