from pathlib import Path

import pandas as pd


# ============================================================
# NEXUS — Split Forecasting Dataset
# ============================================================

INPUT_PATH = Path(
    "data/gold/olist/daily_order_demand_modeling.parquet"
)

OUTPUT_DIR = Path(
    "data/gold/olist/forecasting_splits"
)

TRAIN_START = pd.Timestamp("2017-01-01")
TRAIN_END = pd.Timestamp("2018-03-31")

VALIDATION_START = pd.Timestamp("2018-04-01")
VALIDATION_END = pd.Timestamp("2018-06-30")

TEST_START = pd.Timestamp("2018-07-01")
TEST_END = pd.Timestamp("2018-08-31")


def validate_split(
    df: pd.DataFrame,
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:

    if df.empty:
        raise ValueError(f"{name} split is empty.")

    if df["date"].min() != start:
        raise ValueError(
            f"{name} start mismatch: "
            f"{df['date'].min().date()} != {start.date()}"
        )

    if df["date"].max() != end:
        raise ValueError(
            f"{name} end mismatch: "
            f"{df['date'].max().date()} != {end.date()}"
        )

    expected_dates = pd.date_range(
        start=start,
        end=end,
        freq="D",
    )

    actual_dates = (
        df["date"]
        .reset_index(drop=True)
    )

    expected_dates = pd.Series(
        expected_dates,
        name="date",
    )

    if not actual_dates.equals(expected_dates):
        raise ValueError(
            f"{name} does not contain a continuous daily calendar."
        )

    if df["order_count"].isna().any():
        raise ValueError(
            f"{name} contains missing order counts."
        )

    if (df["order_count"] < 0).any():
        raise ValueError(
            f"{name} contains negative demand."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate dates."
        )


def main():

    print("NEXUS — Split Forecasting Dataset")
    print("=" * 55)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    daily = pd.read_parquet(INPUT_PATH)

    daily["date"] = pd.to_datetime(
        daily["date"]
    ).dt.normalize()

    daily = (
        daily
        .sort_values("date")
        .reset_index(drop=True)
    )

    print("\nSOURCE")
    print("-" * 55)
    print(f"Rows:       {len(daily):,}")
    print(f"Start:      {daily['date'].min().date()}")
    print(f"End:        {daily['date'].max().date()}")

    # --------------------------------------------------------
    # Create temporal splits
    # --------------------------------------------------------

    train = daily[
        daily["date"].between(
            TRAIN_START,
            TRAIN_END,
            inclusive="both",
        )
    ].copy()

    validation = daily[
        daily["date"].between(
            VALIDATION_START,
            VALIDATION_END,
            inclusive="both",
        )
    ].copy()

    test = daily[
        daily["date"].between(
            TEST_START,
            TEST_END,
            inclusive="both",
        )
    ].copy()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_split(
        train,
        "TRAIN",
        TRAIN_START,
        TRAIN_END,
    )

    validate_split(
        validation,
        "VALIDATION",
        VALIDATION_START,
        VALIDATION_END,
    )

    validate_split(
        test,
        "TEST",
        TEST_START,
        TEST_END,
    )

    # --------------------------------------------------------
    # Validate no temporal overlap
    # --------------------------------------------------------

    if train["date"].max() >= validation["date"].min():
        raise ValueError(
            "TRAIN and VALIDATION overlap."
        )

    if validation["date"].max() >= test["date"].min():
        raise ValueError(
            "VALIDATION and TEST overlap."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.parquet"
    validation_path = OUTPUT_DIR / "validation.parquet"
    test_path = OUTPUT_DIR / "test.parquet"

    train.to_parquet(
        train_path,
        index=False,
    )

    validation.to_parquet(
        validation_path,
        index=False,
    )

    test.to_parquet(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nTRAIN")
    print("-" * 55)
    print(f"Dates:       {train['date'].min().date()} → "
          f"{train['date'].max().date()}")
    print(f"Days:        {len(train):,}")
    print(f"Total orders:{train['order_count'].sum():,.0f}")
    print(f"Mean demand: {train['order_count'].mean():.2f}")

    print("\nVALIDATION")
    print("-" * 55)
    print(f"Dates:       {validation['date'].min().date()} → "
          f"{validation['date'].max().date()}")
    print(f"Days:        {len(validation):,}")
    print(f"Total orders:{validation['order_count'].sum():,.0f}")
    print(f"Mean demand: {validation['order_count'].mean():.2f}")

    print("\nTEST")
    print("-" * 55)
    print(f"Dates:       {test['date'].min().date()} → "
          f"{test['date'].max().date()}")
    print(f"Days:        {len(test):,}")
    print(f"Total orders:{test['order_count'].sum():,.0f}")
    print(f"Mean demand: {test['order_count'].mean():.2f}")

    print("\nOUTPUT")
    print("-" * 55)
    print(f"Saved: {train_path}")
    print(f"Saved: {validation_path}")
    print(f"Saved: {test_path}")

    print("\nTemporal forecasting splits ready.")


if __name__ == "__main__":
    main()