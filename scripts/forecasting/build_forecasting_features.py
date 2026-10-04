from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NEXUS — Build Forecasting Features
# ============================================================

INPUT_PATH = Path(
    "data/gold/olist/daily_order_demand_modeling.parquet"
)

OUTPUT_PATH = Path(
    "data/gold/olist/daily_order_demand_features.parquet"
)


def main():

    print("NEXUS — Build Forecasting Features")
    print("=" * 60)

    df = pd.read_parquet(INPUT_PATH)

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate source
    # --------------------------------------------------------

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found."
        )

    if df["order_count"].isna().any():
        raise ValueError(
            "Missing order_count values found."
        )

    if (df["order_count"] < 0).any():
        raise ValueError(
            "Negative demand found."
        )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["date"].dt.day
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Trend feature
    # --------------------------------------------------------

    df["time_index"] = np.arange(
        len(df)
    )

    # --------------------------------------------------------
    # Lag features
    #
    # IMPORTANT:
    # Every lag uses only historical observations.
    # --------------------------------------------------------

    for lag in [1, 2, 3, 7, 14, 21, 28]:

        df[f"lag_{lag}"] = (
            df["order_count"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling features
    #
    # Shift FIRST, then roll.
    # This prevents today's target from
    # entering today's features.
    # --------------------------------------------------------

    historical_demand = (
        df["order_count"]
        .shift(1)
    )

    for window in [7, 14, 28]:

        df[f"rolling_mean_{window}"] = (
            historical_demand
            .rolling(window)
            .mean()
        )

        df[f"rolling_std_{window}"] = (
            historical_demand
            .rolling(window)
            .std()
        )

        df[f"rolling_min_{window}"] = (
            historical_demand
            .rolling(window)
            .min()
        )

        df[f"rolling_max_{window}"] = (
            historical_demand
            .rolling(window)
            .max()
        )

    # --------------------------------------------------------
    # Validate feature leakage
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "date",
            "order_count",
        ]
    ]

    if not feature_columns:
        raise ValueError(
            "No features were created."
        )

    # --------------------------------------------------------
    # Drop rows without enough history
    # --------------------------------------------------------

    before = len(df)

    df = (
        df
        .dropna(
            subset=[
                "lag_28",
                "rolling_mean_28",
                "rolling_std_28",
            ]
        )
        .reset_index(drop=True)
    )

    removed = before - len(df)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Feature table contains missing values."
        )

    if df["order_count"].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nSOURCE")
    print("-" * 60)
    print(
        f"Rows before feature history requirement: "
        f"{before:,}"
    )

    print(
        f"Rows removed: {removed:,}"
    )

    print(
        f"Rows after feature engineering: "
        f"{len(df):,}"
    )

    print(
        f"Feature count: {len(feature_columns):,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print("\nFEATURES")
    print("-" * 60)

    for feature in feature_columns:
        print(feature)

    print("\nOUTPUT")
    print("-" * 60)
    print(f"Saved: {OUTPUT_PATH}")

    print(
        "\nForecasting feature engineering complete."
    )


if __name__ == "__main__":
    main()