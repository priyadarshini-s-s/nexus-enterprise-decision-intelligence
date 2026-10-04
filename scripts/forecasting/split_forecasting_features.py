from pathlib import Path

import pandas as pd


# ============================================================
# NEXUS — Split Forecasting Feature Dataset
# ============================================================

INPUT_PATH = Path(
    "data/gold/olist/daily_order_demand_features.parquet"
)

OUTPUT_DIR = Path(
    "data/gold/olist/forecasting_feature_splits"
)

TRAIN_START = pd.Timestamp("2017-01-29")
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
        raise ValueError(
            f"{name} split is empty."
        )

    if df["date"].min() != start:
        raise ValueError(
            f"{name} start mismatch: "
            f"{df['date'].min().date()} "
            f"!= {start.date()}"
        )

    if df["date"].max() != end:
        raise ValueError(
            f"{name} end mismatch: "
            f"{df['date'].max().date()} "
            f"!= {end.date()}"
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

    if not actual_dates.equals(
        expected_dates
    ):
        raise ValueError(
            f"{name} does not contain "
            f"a continuous daily calendar."
        )

    if df["order_count"].isna().any():
        raise ValueError(
            f"{name} contains missing targets."
        )

    if (
        df["order_count"] < 0
    ).any():
        raise ValueError(
            f"{name} contains negative demand."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate dates."
        )


def main():

    print(
        "NEXUS — Split Forecasting Feature Dataset"
    )
    print("=" * 60)

    df = pd.read_parquet(
        INPUT_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate feature table
    # --------------------------------------------------------

    if df.isna().any().any():
        raise ValueError(
            "Feature table contains missing values."
        )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    train = df[
        df["date"].between(
            TRAIN_START,
            TRAIN_END,
            inclusive="both",
        )
    ].copy()

    validation = df[
        df["date"].between(
            VALIDATION_START,
            VALIDATION_END,
            inclusive="both",
        )
    ].copy()

    test = df[
        df["date"].between(
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
    # Validate temporal separation
    # --------------------------------------------------------

    if (
        train["date"].max()
        >= validation["date"].min()
    ):
        raise ValueError(
            "TRAIN and VALIDATION overlap."
        )

    if (
        validation["date"].max()
        >= test["date"].min()
    ):
        raise ValueError(
            "VALIDATION and TEST overlap."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        OUTPUT_DIR / "train.parquet"
    )

    validation_path = (
        OUTPUT_DIR / "validation.parquet"
    )

    test_path = (
        OUTPUT_DIR / "test.parquet"
    )

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

    for name, split in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        print(f"\n{name}")
        print("-" * 60)

        print(
            f"Dates: "
            f"{split['date'].min().date()} → "
            f"{split['date'].max().date()}"
        )

        print(
            f"Days: "
            f"{len(split):,}"
        )

        print(
            f"Total orders: "
            f"{split['order_count'].sum():,.0f}"
        )

        print(
            f"Mean demand: "
            f"{split['order_count'].mean():.2f}"
        )

    print("\nOUTPUT")
    print("-" * 60)

    print(f"Saved: {train_path}")
    print(f"Saved: {validation_path}")
    print(f"Saved: {test_path}")

    print(
        "\nForecasting feature splits ready."
    )


if __name__ == "__main__":
    main()