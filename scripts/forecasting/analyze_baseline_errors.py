from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_DIR = Path(
    "data/gold/olist/forecasting_splits"
)

OUTPUT_PATH = Path(
    "data/gold/olist/forecasting_baseline_error_analysis.parquet"
)


def calculate_errors(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    result["naive_error"] = (
        result["actual"] - result["naive"]
    )

    result["seasonal_naive_error"] = (
        result["actual"] - result["seasonal_naive_7"]
    )

    result["moving_average_error"] = (
        result["actual"] - result["moving_average_7"]
    )

    result["naive_abs_error"] = (
        result["naive_error"].abs()
    )

    result["seasonal_naive_abs_error"] = (
        result["seasonal_naive_error"].abs()
    )

    result["moving_average_abs_error"] = (
        result["moving_average_error"].abs()
    )

    result["weekday"] = result["date"].dt.day_name()

    result["month"] = (
        result["date"]
        .dt.to_period("M")
        .astype(str)
    )

    return result


def summarize(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    return (
        df.groupby(group_column)
        .agg(
            days=("date", "count"),
            actual_mean=("actual", "mean"),
            naive_mae=("naive_abs_error", "mean"),
            seasonal_naive_mae=(
                "seasonal_naive_abs_error",
                "mean",
            ),
            moving_average_mae=(
                "moving_average_abs_error",
                "mean",
            ),
        )
        .reset_index()
    )


def main():

    print(
        "NEXUS — Forecasting Baseline Error Analysis"
    )
    print("=" * 65)

    validation = pd.read_parquet(
        SPLIT_DIR / "validation.parquet"
    )

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    full = pd.concat(
        [validation, test],
        ignore_index=True,
    )

    full["date"] = pd.to_datetime(
        full["date"]
    )

    # --------------------------------------------------------
    # Reconstruct baseline predictions
    # --------------------------------------------------------

    all_data = (
        pd.concat(
            [
                pd.read_parquet(
                    SPLIT_DIR / "train.parquet"
                ),
                validation,
                test,
            ],
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    series = (
        all_data
        .set_index("date")["order_count"]
        .sort_index()
    )

    dates = pd.DatetimeIndex(
        full["date"]
    )

    full["actual"] = (
        series.loc[dates].to_numpy()
    )

    full["naive"] = (
        series.shift(1)
        .loc[dates]
        .to_numpy()
    )

    full["seasonal_naive_7"] = (
        series.shift(7)
        .loc[dates]
        .to_numpy()
    )

    full["moving_average_7"] = (
        series
        .shift(1)
        .rolling(7)
        .mean()
        .loc[dates]
        .to_numpy()
    )

    full = calculate_errors(full)

    # --------------------------------------------------------
    # Validation vs Test
    # --------------------------------------------------------

    print("\nOVERALL ERROR")
    print("-" * 65)

    for split_name in ["validation", "test"]:

        subset = full[
            full["date"].isin(
                validation["date"]
                if split_name == "validation"
                else test["date"]
            )
        ]

        print(f"\n{split_name.upper()}")

        print(
            f"Naive MAE:            "
            f"{subset['naive_abs_error'].mean():.2f}"
        )

        print(
            f"Seasonal Naive MAE:   "
            f"{subset['seasonal_naive_abs_error'].mean():.2f}"
        )

        print(
            f"Moving Average MAE:   "
            f"{subset['moving_average_abs_error'].mean():.2f}"
        )

    # --------------------------------------------------------
    # Monthly analysis
    # --------------------------------------------------------

    monthly = summarize(
        full,
        "month",
    )

    print("\nMONTHLY ERROR")
    print("-" * 65)

    print(
        monthly.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Weekday analysis
    # --------------------------------------------------------

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday = summarize(
        full,
        "weekday",
    )

    weekday["weekday"] = pd.Categorical(
        weekday["weekday"],
        categories=weekday_order,
        ordered=True,
    )

    weekday = weekday.sort_values(
        "weekday"
    )

    print("\nWEEKDAY ERROR")
    print("-" * 65)

    print(
        weekday.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Error direction
    # --------------------------------------------------------

    print("\nBIAS")
    print("-" * 65)

    for model, error_column in [
        ("Naive", "naive_error"),
        (
            "Seasonal Naive",
            "seasonal_naive_error",
        ),
        (
            "Moving Average",
            "moving_average_error",
        ),
    ]:

        mean_error = full[
            error_column
        ].mean()

        print(
            f"{model:20s}: "
            f"{mean_error:+.2f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    full.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nOUTPUT")
    print("-" * 65)
    print(f"Saved: {OUTPUT_PATH}")

    print(
        "\nBaseline error analysis complete."
    )


if __name__ == "__main__":
    main()