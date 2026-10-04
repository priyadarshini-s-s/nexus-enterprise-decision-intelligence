from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NEXUS — Forecasting Baseline Benchmark
# ============================================================

SPLIT_DIR = Path(
    "data/gold/olist/forecasting_splits"
)

OUTPUT_PATH = Path(
    "data/gold/olist/forecasting_baseline_results.parquet"
)


def mae(actual, predicted):
    return np.mean(
        np.abs(actual - predicted)
    )


def rmse(actual, predicted):
    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


def evaluate(
    actual: pd.Series,
    predicted: pd.Series,
):
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
    }


def prepare_full_series():

    train = pd.read_parquet(
        SPLIT_DIR / "train.parquet"
    )

    validation = pd.read_parquet(
        SPLIT_DIR / "validation.parquet"
    )

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    full = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    full["date"] = pd.to_datetime(
        full["date"]
    )

    full = (
        full
        .sort_values("date")
        .reset_index(drop=True)
    )

    return train, validation, test, full


def make_predictions(
    full: pd.DataFrame,
    evaluation_dates: pd.Series,
):

    series = (
        full
        .set_index("date")["order_count"]
        .sort_index()
    )

    dates = pd.DatetimeIndex(
        evaluation_dates
    )

    actual = series.loc[dates]

    # --------------------------------------------------------
    # 1. Naive
    # Yesterday's demand
    # --------------------------------------------------------

    naive = series.shift(1).loc[dates]

    # --------------------------------------------------------
    # 2. Seasonal Naive
    # Same weekday from previous week
    # --------------------------------------------------------

    seasonal_naive = series.shift(7).loc[dates]

    # --------------------------------------------------------
    # 3. 7-Day Moving Average
    # Mean of previous 7 observed days
    # --------------------------------------------------------

    moving_average = (
        series
        .shift(1)
        .rolling(window=7)
        .mean()
        .loc[dates]
    )

    predictions = pd.DataFrame(
        {
            "date": dates,
            "actual": actual.values,
            "naive": naive.values,
            "seasonal_naive_7": seasonal_naive.values,
            "moving_average_7": moving_average.values,
        }
    )

    return predictions


def evaluate_split(
    predictions: pd.DataFrame,
    split_name: str,
):

    results = []

    actual = predictions["actual"]

    models = {
        "naive": predictions["naive"],
        "seasonal_naive_7": predictions[
            "seasonal_naive_7"
        ],
        "moving_average_7": predictions[
            "moving_average_7"
        ],
    }

    for model_name, predicted in models.items():

        valid = (
            actual.notna()
            & predicted.notna()
        )

        metrics = evaluate(
            actual[valid].to_numpy(),
            predicted[valid].to_numpy(),
        )

        results.append(
            {
                "split": split_name,
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "evaluation_days": int(
                    valid.sum()
                ),
            }
        )

    return results


def main():

    print(
        "NEXUS — Forecasting Baseline Benchmark"
    )
    print("=" * 60)

    train, validation, test, full = (
        prepare_full_series()
    )

    print("\nDATA")
    print("-" * 60)
    print(
        f"Train:      {train['date'].min().date()} → "
        f"{train['date'].max().date()}"
    )
    print(
        f"Validation: {validation['date'].min().date()} → "
        f"{validation['date'].max().date()}"
    )
    print(
        f"Test:       {test['date'].min().date()} → "
        f"{test['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_predictions = make_predictions(
        full,
        validation["date"],
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_predictions = make_predictions(
        full,
        test["date"],
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    validation_results = evaluate_split(
        validation_predictions,
        "validation",
    )

    test_results = evaluate_split(
        test_predictions,
        "test",
    )

    results = pd.DataFrame(
        validation_results
        + test_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nVALIDATION RESULTS")
    print("-" * 60)

    print(
        results[
            results["split"] == "validation"
        ]
        .sort_values("MAE")
        .to_string(index=False)
    )

    print("\nTEST RESULTS")
    print("-" * 60)

    print(
        results[
            results["split"] == "test"
        ]
        .sort_values("MAE")
        .to_string(index=False)
    )

    print("\nBEST VALIDATION MODEL")
    print("-" * 60)

    best_validation = (
        results[
            results["split"] == "validation"
        ]
        .sort_values("MAE")
        .iloc[0]
    )

    print(
        f"Model: {best_validation['model']}"
    )

    print(
        f"MAE:   {best_validation['MAE']:.2f}"
    )

    print(
        f"RMSE:  {best_validation['RMSE']:.2f}"
    )

    print("\nOUTPUT")
    print("-" * 60)
    print(f"Saved: {OUTPUT_PATH}")

    print(
        "\nForecasting baseline benchmark complete."
    )


if __name__ == "__main__":
    main()