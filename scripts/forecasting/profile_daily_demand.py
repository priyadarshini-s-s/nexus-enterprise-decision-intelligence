from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "daily_order_demand.parquet"
)


def main():
    print("NEXUS — Daily Demand Profile")
    print("=" * 50)

    daily = pd.read_parquet(INPUT_PATH)

    required = {"date", "order_count"}

    missing = required - set(daily.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values("date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    expected_dates = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D",
    )

    if len(daily) != len(expected_dates):
        raise ValueError(
            "Daily series is not continuous."
        )

    if not daily["date"].equals(
        pd.Series(expected_dates, name="date")
    ):
        raise ValueError(
            "Date index does not match expected daily calendar."
        )

    if daily["order_count"].lt(0).any():
        raise ValueError(
            "Negative order counts detected."
        )

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    print("\nBASIC STATISTICS")
    print("-" * 50)

    print(f"Days: {len(daily):,}")
    print(f"Total orders: {daily['order_count'].sum():,}")
    print(f"Mean: {daily['order_count'].mean():.2f}")
    print(f"Median: {daily['order_count'].median():.2f}")
    print(f"Std: {daily['order_count'].std():.2f}")
    print(f"Minimum: {daily['order_count'].min():,}")
    print(f"Maximum: {daily['order_count'].max():,}")

    # ---------------------------------------------------------
    # Percentiles
    # ---------------------------------------------------------

    print("\nPERCENTILES")
    print("-" * 50)

    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    for p in percentiles:
        value = daily["order_count"].quantile(p)
        print(f"p{p * 100:>5.1f}: {value:.2f}")

    # ---------------------------------------------------------
    # Zero-demand analysis
    # ---------------------------------------------------------

    zero_days = daily[daily["order_count"] == 0]

    print("\nZERO-DEMAND DAYS")
    print("-" * 50)

    print(f"Zero-demand days: {len(zero_days):,}")
    print(
        f"Percentage of calendar: "
        f"{len(zero_days) / len(daily) * 100:.2f}%"
    )

    if not zero_days.empty:
        print(f"First zero-demand date: {zero_days['date'].min()}")
        print(f"Last zero-demand date: {zero_days['date'].max()}")

    # ---------------------------------------------------------
    # Day-of-week analysis
    # ---------------------------------------------------------

    daily["day_of_week"] = daily["date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_summary = (
        daily.groupby("day_of_week")["order_count"]
        .agg(["count", "mean", "median", "std"])
        .reindex(weekday_order)
    )

    print("\nDAY-OF-WEEK DEMAND")
    print("-" * 50)

    print(weekday_summary.round(2).to_string())

    # ---------------------------------------------------------
    # Monthly analysis
    # ---------------------------------------------------------

    daily["year_month"] = daily["date"].dt.to_period("M")

    monthly = (
        daily.groupby("year_month")["order_count"]
        .agg(["count", "sum", "mean", "median"])
    )

    print("\nMONTHLY DEMAND — FIRST 10 MONTHS")
    print("-" * 50)

    print(monthly.head(10).round(2).to_string())

    print("\nMONTHLY DEMAND — LAST 10 MONTHS")
    print("-" * 50)

    print(monthly.tail(10).round(2).to_string())

    # ---------------------------------------------------------
    # Highest-demand days
    # ---------------------------------------------------------

    print("\nTOP 10 DEMAND DAYS")
    print("-" * 50)

    top_days = (
        daily[["date", "order_count"]]
        .sort_values(
            "order_count",
            ascending=False,
        )
        .head(10)
    )

    print(top_days.to_string(index=False))

    # ---------------------------------------------------------
    # Lowest non-zero demand days
    # ---------------------------------------------------------

    print("\nLOWEST NON-ZERO DEMAND DAYS")
    print("-" * 50)

    low_days = (
        daily.loc[
            daily["order_count"] > 0,
            ["date", "order_count"],
        ]
        .sort_values(
            "order_count",
            ascending=True,
        )
        .head(10)
    )

    print(low_days.to_string(index=False))


if __name__ == "__main__":
    main()