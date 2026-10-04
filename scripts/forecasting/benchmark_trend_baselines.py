from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_DIR = Path("data/gold/olist/forecasting_splits")
OUTPUT_PATH = Path(
    "data/gold/olist/trend_baseline_results.parquet"
)


def mae(actual, predicted):
    return float(
        np.mean(np.abs(actual - predicted))
    )


def rmse(actual, predicted):
    return float(
        np.sqrt(np.mean((actual - predicted) ** 2))
    )


def load_split(name):
    df = pd.read_parquet(
        SPLIT_DIR / f"{name}.parquet"
    )

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def drift_forecast(train_values, horizon):
    """
    Linear drift from first observed value
    to latest observed value.
    """
    n = len(train_values)

    if n < 2:
        return np.repeat(
            train_values[-1],
            horizon,
        )

    drift = (
        train_values[-1] - train_values[0]
    ) / (n - 1)

    steps = np.arange(
        1,
        horizon + 1,
    )

    forecast = (
        train_values[-1]
        + steps * drift
    )

    return np.maximum(
        forecast,
        0,
    )


def damped_drift_forecast(
    train_values,
    horizon,
    damping=0.85,
):
    """
    Damped trend:
    recent trend is retained but its
    contribution decays over the horizon.
    """
    n = len(train_values)

    if n < 2:
        return np.repeat(
            train_values[-1],
            horizon,
        )

    drift = (
        train_values[-1] - train_values[0]
    ) / (n - 1)

    steps = np.arange(
        1,
        horizon + 1,
    )

    trend_multiplier = (
        1 - damping ** steps
    ) / (1 - damping)

    forecast = (
        train_values[-1]
        + drift * trend_multiplier
    )

    return np.maximum(
        forecast,
        0,
    )


def seasonal_naive_with_drift(
    train_values,
    horizon,
    season_length=7,
    drift_weight=0.25,
):
    """
    Seasonal naive forecast with a small
    trend correction.

    Base:
        y[t-7]

    Trend correction:
        recent level - seasonal level
    """

    history = list(train_values)

    if len(history) < season_length:
        return np.repeat(
            history[-1],
            horizon,
        )

    recent_window = np.mean(
        history[-season_length:]
    )

    previous_season_window = np.mean(
        history[
            -2 * season_length:
            -season_length
        ]
    )

    trend_signal = (
        recent_window
        - previous_season_window
    )

    predictions = []

    for _ in range(horizon):

        seasonal_value = history[-season_length]

        prediction = (
            seasonal_value
            + drift_weight * trend_signal
        )

        prediction = max(
            prediction,
            0,
        )

        predictions.append(
            prediction
        )

        history.append(prediction)

    return np.asarray(predictions)


def weekday_adjusted_naive(
    train_df,
    validation_df,
):
    """
    Start from the latest observed level
    and apply a weekday correction estimated
    from training data.

    Correction is multiplicative relative
    to the overall training mean.
    """

    train = train_df.copy()
    validation = validation_df.copy()

    train["dow"] = train["date"].dt.dayofweek
    validation["dow"] = validation["date"].dt.dayofweek

    overall_mean = train["order_count"].mean()

    weekday_means = (
        train
        .groupby("dow")["order_count"]
        .mean()
    )

    weekday_factors = (
        weekday_means / overall_mean
    )

    latest = train["order_count"].iloc[-1]

    predictions = []

    for dow in validation["dow"]:

        factor = weekday_factors.get(
            dow,
            1.0,
        )

        prediction = (
            latest * factor
        )

        predictions.append(
            max(prediction, 0)
        )

    return np.asarray(predictions)


def evaluate(
    name,
    actual,
    predicted,
):
    return {
        "model": name,
        "MAE": mae(
            actual,
            predicted,
        ),
        "RMSE": rmse(
            actual,
            predicted,
        ),
    }


def main():

    print(
        "NEXUS — Trend-Aware Forecast Baseline Benchmark"
    )
    print("=" * 70)

    train = load_split("train")
    validation = load_split("validation")

    train_values = (
        train["order_count"]
        .to_numpy(dtype=float)
    )

    actual = (
        validation["order_count"]
        .to_numpy(dtype=float)
    )

    horizon = len(validation)

    results = []

    # --------------------------------------------------------
    # 1. Drift
    # --------------------------------------------------------

    prediction = drift_forecast(
        train_values,
        horizon,
    )

    results.append(
        evaluate(
            "drift",
            actual,
            prediction,
        )
    )

    # --------------------------------------------------------
    # 2. Damped drift
    # --------------------------------------------------------

    for damping in [
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
    ]:

        prediction = damped_drift_forecast(
            train_values,
            horizon,
            damping=damping,
        )

        results.append(
            evaluate(
                f"damped_drift_{damping}",
                actual,
                prediction,
            )
        )

    # --------------------------------------------------------
    # 3. Seasonal naive + trend
    # --------------------------------------------------------

    for weight in [
        0.10,
        0.25,
        0.50,
        0.75,
    ]:

        prediction = seasonal_naive_with_drift(
            train_values,
            horizon,
            season_length=7,
            drift_weight=weight,
        )

        results.append(
            evaluate(
                f"seasonal_naive_drift_{weight}",
                actual,
                prediction,
            )
        )

    # --------------------------------------------------------
    # 4. Weekday-adjusted naive
    # --------------------------------------------------------

    prediction = weekday_adjusted_naive(
        train,
        validation,
    )

    results.append(
        evaluate(
            "weekday_adjusted_naive",
            actual,
            prediction,
        )
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

    print("\nBEST MODEL")
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


if __name__ == "__main__":
    main()