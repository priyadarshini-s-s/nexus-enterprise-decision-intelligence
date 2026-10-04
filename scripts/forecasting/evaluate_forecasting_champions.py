from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


SPLIT_DIR = Path("data/gold/olist/forecasting_splits")
FEATURE_SPLIT_DIR = Path(
    "data/gold/olist/forecasting_feature_splits"
)
OUTPUT_DIR = Path("data/gold/olist")

RESULTS_PATH = (
    OUTPUT_DIR /
    "final_forecasting_champion_results.parquet"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR /
    "final_forecasting_champion_predictions.parquet"
)


FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "time_index",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_min_7",
    "rolling_max_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_min_14",
    "rolling_max_14",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_min_28",
    "rolling_max_28",
]


def mae(actual, predicted):
    return float(
        np.mean(np.abs(actual - predicted))
    )


def rmse(actual, predicted):
    return float(
        np.sqrt(
            np.mean((actual - predicted) ** 2)
        )
    )


def load_series_split(name):
    df = pd.read_parquet(
        SPLIT_DIR / f"{name}.parquet"
    )

    df["date"] = pd.to_datetime(df["date"])

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_feature_split(name):
    df = pd.read_parquet(
        FEATURE_SPLIT_DIR / f"{name}.parquet"
    )

    df["date"] = pd.to_datetime(df["date"])

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


def one_step_naive_forecast(
    history,
    future_actual,
):
    """
    One-step-ahead Naive forecast.

    For each future day:
        prediction[t] = actual[t-1]

    This is the same evaluation convention
    used by the original baseline benchmark.
    """

    combined = np.concatenate(
        [
            history,
            future_actual,
        ]
    )

    start = len(history)

    predictions = combined[
        start - 1:
        start - 1 + len(future_actual)
    ]

    return predictions.astype(float)


def one_step_seasonal_naive_forecast(
    history,
    future_actual,
    season_length=7,
):
    """
    One-step-ahead seasonal Naive forecast.

    prediction[t] = actual[t-7]
    """

    combined = np.concatenate(
        [
            history,
            future_actual,
        ]
    )

    start = len(history)

    predictions = []

    for i in range(len(future_actual)):

        current_index = start + i
        source_index = (
            current_index - season_length
        )

        predictions.append(
            combined[source_index]
        )

    return np.asarray(
        predictions,
        dtype=float,
    )


def train_xgboost(train_df):

    X_train = train_df[FEATURES]
    y_train = train_df["order_count"]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=131,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


def main():

    print(
        "NEXUS — Corrected Final Forecasting Champion Evaluation"
    )
    print("=" * 70)

    # ========================================================
    # Load chronological data
    # ========================================================

    train = load_series_split("train")
    validation = load_series_split("validation")
    test = load_series_split("test")

    train_values = (
        train["order_count"]
        .to_numpy(dtype=float)
    )

    validation_values = (
        validation["order_count"]
        .to_numpy(dtype=float)
    )

    test_actual = (
        test["order_count"]
        .to_numpy(dtype=float)
    )

    train_validation_values = np.concatenate(
        [
            train_values,
            validation_values,
        ]
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

    print(
        f"Test: "
        f"{test['date'].min().date()} → "
        f"{test['date'].max().date()}"
    )

    # ========================================================
    # Naive
    # ========================================================

    naive_pred = one_step_naive_forecast(
        train_validation_values,
        test_actual,
    )

    # ========================================================
    # Seasonal Naive
    # ========================================================

    seasonal_pred = (
        one_step_seasonal_naive_forecast(
            train_validation_values,
            test_actual,
            season_length=7,
        )
    )

    # ========================================================
    # XGBoost
    # ========================================================

    train_features = load_feature_split(
        "train"
    )

    validation_features = load_feature_split(
        "validation"
    )

    test_features = load_feature_split(
        "test"
    )

    combined_features = pd.concat(
        [
            train_features,
            validation_features,
        ],
        ignore_index=True,
    )

    print("\nXGBOOST")
    print("-" * 70)

    print(
        f"Training rows: "
        f"{len(combined_features)}"
    )

    print(
        f"Training period: "
        f"{combined_features['date'].min().date()} → "
        f"{combined_features['date'].max().date()}"
    )

    model = train_xgboost(
        combined_features
    )

    xgb_pred = model.predict(
        test_features[FEATURES]
    )

    xgb_pred = np.maximum(
        xgb_pred,
        0,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    predictions = {
        "naive": naive_pred,
        "seasonal_naive": seasonal_pred,
        "xgboost": xgb_pred,
    }

    results = []
    prediction_rows = []

    for model_name, prediction in predictions.items():

        model_mae = mae(
            test_actual,
            prediction,
        )

        model_rmse = rmse(
            test_actual,
            prediction,
        )

        results.append(
            {
                "model": model_name,
                "test_mae": model_mae,
                "test_rmse": model_rmse,
                "test_days": len(test),
                "evaluation_method": (
                    "one_step_ahead"
                ),
            }
        )

        for date, actual, pred in zip(
            test["date"],
            test_actual,
            prediction,
        ):

            prediction_rows.append(
                {
                    "date": date,
                    "model": model_name,
                    "actual_order_count": actual,
                    "predicted_order_count": pred,
                    "absolute_error": abs(
                        actual - pred
                    ),
                }
            )

    results_df = (
        pd.DataFrame(results)
        .sort_values("test_mae")
        .reset_index(drop=True)
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_parquet(
        RESULTS_PATH,
        index=False,
    )

    predictions_df.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # Results
    # ========================================================

    print("\nFINAL TEST RESULTS")
    print("-" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    champion = results_df.iloc[0]

    print("\nCHAMPION")
    print("-" * 70)

    print(
        f"Model: {champion['model']}"
    )

    print(
        f"Test MAE: "
        f"{champion['test_mae']:.4f}"
    )

    print(
        f"Test RMSE: "
        f"{champion['test_rmse']:.4f}"
    )

    # ========================================================
    # Compare against Naive
    # ========================================================

    naive_mae = float(
        results_df.loc[
            results_df["model"] == "naive",
            "test_mae",
        ].iloc[0]
    )

    print("\nMODEL COMPARISON")
    print("-" * 70)

    for _, row in results_df.iterrows():

        if row["model"] == "naive":
            continue

        difference = (
            (
                row["test_mae"]
                - naive_mae
            )
            / naive_mae
            * 100
        )

        print(
            f"{row['model']}: "
            f"{difference:+.2f}% vs Naive"
        )

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"Results: {RESULTS_PATH}"
    )

    print(
        f"Predictions: {PREDICTIONS_PATH}"
    )

    print(
        "\nCorrected final forecasting evaluation complete."
    )


if __name__ == "__main__":
    main()