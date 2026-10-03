from pathlib import Path

import pandas as pd


SILVER_DIR = Path("data/silver/olist")
GOLD_DIR = Path("data/gold/olist")


def main():
    print("=" * 80)
    print("NEXUS — Retention Temporal Coverage Analysis")
    print("=" * 80)

    orders = pd.read_parquet(
        SILVER_DIR / "orders.parquet"
    )

    customers = pd.read_parquet(
        SILVER_DIR / "customers.parquet"
    )

    # --------------------------------------------------------------
    # Build persistent-customer purchase history
    # --------------------------------------------------------------

    customer_orders = orders.merge(
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

    customer_orders = customer_orders[
        customer_orders["order_purchase_timestamp"].notna()
    ].copy()

    dataset_end = (
        customer_orders[
            "order_purchase_timestamp"
        ].max()
    )

    dataset_start = (
        customer_orders[
            "order_purchase_timestamp"
        ].min()
    )

    print("\nDataset purchase window")
    print("-" * 80)
    print(f"Start: {dataset_start}")
    print(f"End:   {dataset_end}")
    print(
        "Days:",
        (dataset_end - dataset_start).days,
    )

    # --------------------------------------------------------------
    # Customer last observed purchase
    # --------------------------------------------------------------

    customer_history = (
        customer_orders
        .groupby("customer_unique_id")
        .agg(
            first_purchase=(
                "order_purchase_timestamp",
                "min",
            ),
            last_purchase=(
                "order_purchase_timestamp",
                "max",
            ),
            total_orders=(
                "order_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    customer_history["days_available_after_last_purchase"] = (
        dataset_end
        - customer_history["last_purchase"]
    ).dt.days

    # --------------------------------------------------------------
    # Coverage thresholds
    # --------------------------------------------------------------

    horizons = [
        30,
        60,
        90,
        120,
        180,
        270,
        365,
    ]

    print("\nCustomers with sufficient future observation")
    print("-" * 80)

    coverage_rows = []

    for horizon in horizons:
        eligible = (
            customer_history[
                "days_available_after_last_purchase"
            ] >= horizon
        )

        count = eligible.sum()

        pct = count / len(customer_history) * 100

        coverage_rows.append(
            {
                "horizon_days": horizon,
                "eligible_customers": count,
                "eligible_pct": pct,
            }
        )

        print(
            f"{horizon:>3} days → "
            f"{count:>6,} customers "
            f"({pct:>6.2f}%)"
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    # --------------------------------------------------------------
    # Purchase timing distribution
    # --------------------------------------------------------------

    print("\nDays available after last purchase")
    print("-" * 80)

    print(
        customer_history[
            "days_available_after_last_purchase"
        ]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------------
    # Recent-customer counts
    # --------------------------------------------------------------

    print("\nCustomers by last-purchase period")
    print("-" * 80)

    customer_history["last_purchase_month"] = (
        customer_history["last_purchase"]
        .dt.to_period("M")
    )

    monthly = (
        customer_history
        .groupby("last_purchase_month")
        .size()
        .reset_index(
            name="customers"
        )
    )

    print(
        monthly.tail(15).to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Save analysis artifact
    # --------------------------------------------------------------

    output_path = (
        GOLD_DIR /
        "retention_temporal_coverage.parquet"
    )

    coverage.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("TEMPORAL COVERAGE ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Customers analyzed: "
        f"{len(customer_history):,}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()