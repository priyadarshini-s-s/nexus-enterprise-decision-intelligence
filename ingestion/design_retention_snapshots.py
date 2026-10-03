from pathlib import Path

import pandas as pd


SILVER_DIR = Path("data/silver/olist")
GOLD_DIR = Path("data/gold/olist")


PREDICTION_HORIZON_DAYS = 90
MIN_HISTORY_DAYS = 90


def main():
    print("=" * 80)
    print("NEXUS — Retention Snapshot Design")
    print("=" * 80)

    orders = pd.read_parquet(
        SILVER_DIR / "orders.parquet"
    )

    customers = pd.read_parquet(
        SILVER_DIR / "customers.parquet"
    )

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

    dataset_start = (
        customer_orders[
            "order_purchase_timestamp"
        ].min()
    )

    dataset_end = (
        customer_orders[
            "order_purchase_timestamp"
        ].max()
    )

    # We need at least MIN_HISTORY_DAYS before a cutoff
    # and PREDICTION_HORIZON_DAYS after it.
    earliest_cutoff = (
        dataset_start
        + pd.Timedelta(
            days=MIN_HISTORY_DAYS
        )
    )

    latest_cutoff = (
        dataset_end
        - pd.Timedelta(
            days=PREDICTION_HORIZON_DAYS
        )
    )

    print("\nDataset window")
    print("-" * 80)
    print(f"Start: {dataset_start}")
    print(f"End:   {dataset_end}")

    print("\nEligible cutoff window")
    print("-" * 80)
    print(f"Earliest cutoff: {earliest_cutoff}")
    print(f"Latest cutoff:   {latest_cutoff}")

    # Month-end cutoffs.
    month_ends = pd.date_range(
        start=earliest_cutoff.normalize(),
        end=latest_cutoff.normalize(),
        freq="ME",
    )

    snapshot_rows = []

    for cutoff in month_ends:
        cutoff = pd.Timestamp(cutoff)

        future_end = (
            cutoff
            + pd.Timedelta(
                days=PREDICTION_HORIZON_DAYS
            )
        )

        eligible = (
            (customer_orders["order_purchase_timestamp"] <= cutoff)
        )

        future = (
            (customer_orders["order_purchase_timestamp"] > cutoff)
            &
            (customer_orders["order_purchase_timestamp"] <= future_end)
        )

        customers_with_history = (
            customer_orders.loc[
                eligible,
                "customer_unique_id",
            ].nunique()
        )

                # Customers who already existed by the cutoff.
        history_customers = set(
            customer_orders.loc[
                eligible,
                "customer_unique_id",
            ].unique()
        )

        # Customers purchasing during the future window.
        future_customers = set(
            customer_orders.loc[
                future,
                "customer_unique_id",
            ].unique()
        )

        # Retention target population:
        # future purchasers must ALSO have existed before/equal to T.
        retained_customers = (
            history_customers
            & future_customers
        )

        customers_with_history = len(
            history_customers
        )

        customers_with_future_purchase = len(
            retained_customers
        )

        retention_rate = (
            customers_with_future_purchase
            / customers_with_history
            * 100
            if customers_with_history > 0
            else 0
        )

        snapshot_rows.append(
            {
                "cutoff_timestamp": cutoff,
                "future_window_end": future_end,
                "customers_with_history": (
                    customers_with_history
                ),
                "customers_with_future_purchase": (
                    customers_with_future_purchase
                ),
                "repeat_purchase_90d_rate": (
                    retention_rate
                ),
            }
        )

    snapshots = pd.DataFrame(
        snapshot_rows
    )

    print("\nCandidate monthly snapshots")
    print("-" * 80)

    print(
        snapshots.to_string(
            index=False
        )
    )

    output_path = (
        GOLD_DIR /
        "retention_snapshot_design.parquet"
    )

    snapshots.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RETENTION SNAPSHOT DESIGN COMPLETE")
    print("=" * 80)

    print(
        f"Candidate snapshots: "
        f"{len(snapshots)}"
    )

    print(
        f"Prediction horizon: "
        f"{PREDICTION_HORIZON_DAYS} days"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()