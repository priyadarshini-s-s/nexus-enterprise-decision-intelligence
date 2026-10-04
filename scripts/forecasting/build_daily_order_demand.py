from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "silver" / "olist" / "orders.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold" / "olist"
OUTPUT_PATH = OUTPUT_DIR / "daily_order_demand.parquet"


def main():
    print("NEXUS — Build Daily Order Demand")
    print("=" * 50)

    orders = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "order_id",
        "order_purchase_timestamp",
    }

    missing = required_columns - set(orders.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if orders["order_id"].duplicated().any():
        duplicate_count = orders["order_id"].duplicated().sum()
        raise ValueError(
            f"Expected unique order_id in orders table, "
            f"found {duplicate_count} duplicates."
        )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    invalid_timestamps = orders["order_purchase_timestamp"].isna().sum()

    if invalid_timestamps:
        raise ValueError(
            f"Found {invalid_timestamps} invalid purchase timestamps."
        )

    orders["date"] = (
        orders["order_purchase_timestamp"]
        .dt.normalize()
    )

    daily = (
        orders.groupby("date", as_index=False)
        .agg(
            order_count=("order_id", "nunique")
        )
        .sort_values("date")
    )

    if daily.empty:
        raise ValueError("Daily demand table is empty.")

    expected_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D",
    )

    daily = (
        daily.set_index("date")
        .reindex(expected_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    daily["order_count"] = daily["order_count"].astype("int64")

    if daily["order_count"].lt(0).any():
        raise ValueError("Negative order counts detected.")

    if daily["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Input rows: {len(orders):,}")
    print(f"Output days: {len(daily):,}")
    print(f"Start date: {daily['date'].min()}")
    print(f"End date: {daily['date'].max()}")
    print(f"Total orders: {daily['order_count'].sum():,}")
    print(
        f"Zero-demand days: "
        f"{(daily['order_count'] == 0).sum():,}"
    )
    print(
        f"Mean daily orders: "
        f"{daily['order_count'].mean():.2f}"
    )
    print(
        f"Median daily orders: "
        f"{daily['order_count'].median():.2f}"
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()