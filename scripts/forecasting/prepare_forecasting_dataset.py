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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "daily_order_demand_modeling.parquet"
)

# Initial modeling window selected after source-coverage profiling.
MODEL_START = pd.Timestamp("2017-01-01")
MODEL_END = pd.Timestamp("2018-08-31")


def main():
    print("NEXUS — Prepare Forecasting Modeling Dataset")
    print("=" * 55)

    daily = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "date",
        "order_count",
    }

    missing = required_columns - set(daily.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    daily["date"] = pd.to_datetime(daily["date"])

    daily = (
        daily
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Validate source series
    # ---------------------------------------------------------

    if daily["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected in source demand series."
        )

    expected_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D",
    )

    if len(daily) != len(expected_dates):
        raise ValueError(
            "Source demand series is not a continuous daily calendar."
        )

    if not daily["date"].equals(
        pd.Series(expected_dates, name="date")
    ):
        raise ValueError(
            "Source dates do not match the expected daily calendar."
        )

    if daily["order_count"].isna().any():
        raise ValueError(
            "Missing order counts detected."
        )

    if daily["order_count"].lt(0).any():
        raise ValueError(
            "Negative order counts detected."
        )

    # ---------------------------------------------------------
    # Report full observed range
    # ---------------------------------------------------------

    print("\nSOURCE OBSERVATION WINDOW")
    print("-" * 55)

    print(f"Start: {daily['date'].min().date()}")
    print(f"End:   {daily['date'].max().date()}")
    print(f"Days:  {len(daily):,}")

    # ---------------------------------------------------------
    # Apply modeling window
    # ---------------------------------------------------------

    modeling = (
    daily.loc[
        daily["date"].between(
            MODEL_START,
            MODEL_END,
            inclusive="both",
        )
    ].copy()
    .reset_index(drop=True)
     )

    if modeling.empty:
        raise ValueError(
            "Forecasting modeling window is empty."
        )

    # ---------------------------------------------------------
    # Validate modeling calendar
    # ---------------------------------------------------------

    expected_model_dates = pd.date_range(
        start=MODEL_START,
        end=MODEL_END,
        freq="D",
    )

    if len(modeling) != len(expected_model_dates):
        raise ValueError(
            "Modeling window does not contain every calendar day."
        )

    if not modeling["date"].reset_index(drop=True).equals(
    pd.Series(
        expected_model_dates,
        name="date",
    )
    ):
        raise ValueError(
            "Modeling dates do not match expected calendar."
        )

    # ---------------------------------------------------------
    # Coverage comparison
    # ---------------------------------------------------------

    excluded = daily.loc[
        ~daily["date"].between(
            MODEL_START,
            MODEL_END,
            inclusive="both",
        )
    ].copy()

    print("\nMODELING WINDOW")
    print("-" * 55)

    print(f"Start: {MODEL_START.date()}")
    print(f"End:   {MODEL_END.date()}")
    print(f"Days:  {len(modeling):,}")

    print("\nMODELING DATA")
    print("-" * 55)

    print(
        f"Total orders: "
        f"{modeling['order_count'].sum():,}"
    )

    print(
        f"Mean daily orders: "
        f"{modeling['order_count'].mean():.2f}"
    )

    print(
        f"Median daily orders: "
        f"{modeling['order_count'].median():.2f}"
    )

    print(
        f"Zero-demand days: "
        f"{(modeling['order_count'] == 0).sum():,}"
    )

    print("\nEXCLUDED DATA")
    print("-" * 55)

    print(f"Excluded days: {len(excluded):,}")

    print(
        f"Excluded orders: "
        f"{excluded['order_count'].sum():,}"
    )

    print(
        f"Excluded zero-demand days: "
        f"{(excluded['order_count'] == 0).sum():,}"
    )

    # ---------------------------------------------------------
    # Preserve only analytical columns
    # ---------------------------------------------------------

    modeling = modeling[
        [
            "date",
            "order_count",
        ]
    ].copy()

    modeling["order_count"] = (
        modeling["order_count"]
        .astype("int64")
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if modeling["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected after preparation."
        )

    if modeling["order_count"].lt(0).any():
        raise ValueError(
            "Negative order counts detected after preparation."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nOUTPUT")
    print("-" * 55)
    print(f"Saved: {OUTPUT_PATH}")

    print("\nForecasting modeling dataset ready.")


if __name__ == "__main__":
    main()