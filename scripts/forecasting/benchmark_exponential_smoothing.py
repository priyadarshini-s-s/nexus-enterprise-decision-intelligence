from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# NEXUS — Exponential Smoothing Forecast Benchmark
# ============================================================

SPLIT_DIR = Path(
    "data/gold/olist/forecasting_splits"
)

OUTPUT_PATH = Path(
    "data/gold/olist/"
    "exponential_smoothing_results.parquet"
)


def mae(actual, predicted):
    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def rmse(actual, predicted):
    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def load_split(name):
    df = pd.read_parquet(
        SPLIT_DIR / f"{name}.parquet"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def evaluate_model(
    train_series,
    validation_series,
    trend,
    seasonal,
    damped_trend,
    model_name,
):

    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=7,
        damped_trend=damped_trend,
        initialization_method="estimated",
    )

    fitted = model.fit(
        optimized=True
    )

    validation_pred = fitted.forecast(
        len(validation_series)
    )

    validation_pred = np.maximum(
        np.asarray(validation_pred),
        0,
    )

    actual = validation_series.to_numpy()

    return {
        "model": model_name,
        "trend": trend,
        "seasonal": seasonal,
        "damped_trend": damped_trend,
        "MAE": mae(
            actual,
            validation_pred,
        ),
        "RMSE": rmse(
            actual,
            validation_pred,
        ),
    }


def main():

    print(
        "NEXUS — Exponential Smoothing Forecast Benchmark"
    )
    print("=" * 70)

    train = load_split("train")
    validation = load_split("validation")

    train_series = (
        train
        .set_index("date")["order_count"]
        .asfreq("D")
    )

    validation_series = (
        validation
        .set_index("date")["order_count"]
        .asfreq("D")
    )

    print("\nDATA")
    print("-" * 70)

    print(
        f"Train: "
        f"{train['date'].min().date()} → "
        f"{train['date'].max().date()}"
    )

    print(
        f"Validation: "
        f"{validation['date'].min().date()} → "
        f"{validation['date'].max().date()}"
    )

    experiments = [
        {
            "name": "additive_trend_seasonal",
            "trend": "add",
            "seasonal": "add",
            "damped_trend": False,
        },
        {
            "name": "damped_additive_trend_seasonal",
            "trend": "add",
            "seasonal": "add",
            "damped_trend": True,
        },
        {
            "name": "seasonal_only",
            "trend": None,
            "seasonal": "add",
            "damped_trend": False,
        },
        {
            "name": "trend_only",
            "trend": "add",
            "seasonal": None,
            "damped_trend": False,
        },
    ]

    results = []

    for config in experiments:

        print(
            f"\nRunning: {config['name']}"
        )

        result = evaluate_model(
            train_series=train_series,
            validation_series=validation_series,
            trend=config["trend"],
            seasonal=config["seasonal"],
            damped_trend=config["damped_trend"],
            model_name=config["name"],
        )

        results.append(result)

        print(
            f"Validation MAE: "
            f"{result['MAE']:.4f}"
        )

        print(
            f"Validation RMSE: "
            f"{result['RMSE']:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nRESULTS")
    print("-" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nBEST VALIDATION MODEL")
    print("-" * 70)

    best = results_df.iloc[0]

    print(
        f"Model: {best['model']}"
    )

    print(
        f"Validation MAE: "
        f"{best['MAE']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{best['RMSE']:.4f}"
    )

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nExponential smoothing benchmark complete."
    )


if __name__ == "__main__":
    main()