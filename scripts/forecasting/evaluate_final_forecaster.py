from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# NEXUS — Final Demand Forecaster Evaluation
# ============================================================

SPLIT_DIR = Path(
    "data/gold/olist/forecasting_feature_splits"
)

BASELINE_DIR = Path(
    "data/gold/olist/forecasting_splits"
)

OUTPUT_DIR = Path(
    "data/gold/olist/forecasting_models"
)

RESULTS_PATH = Path(
    "data/gold/olist/"
    "final_forecasting_test_results.parquet"
)

PREDICTIONS_PATH = Path(
    "data/gold/olist/"
    "final_forecasting_test_predictions.parquet"
)

MODEL_PATH = (
    OUTPUT_DIR / "xgboost_demand_final.json"
)

TARGET = "order_count"

BEST_ITERATION = 130

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


def load_features(name):

    path = (
        SPLIT_DIR
        / f"{name}.parquet"
    )

    df = pd.read_parquet(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def main():

    print(
        "NEXUS — Final Demand Forecaster Evaluation"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load feature splits
    # --------------------------------------------------------

    train = load_features("train")
    validation = load_features(
        "validation"
    )
    test = load_features("test")

    # --------------------------------------------------------
    # Combine TRAIN + VALIDATION
    #
    # Validation has already served its purpose:
    # model selection.
    #
    # TEST remains completely untouched.
    # --------------------------------------------------------

    training_data = (
        pd.concat(
            [train, validation],
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    X_train = training_data[
        FEATURES
    ]

    y_train = training_data[
        TARGET
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        TARGET
    ]

    # --------------------------------------------------------
    # Validate feature consistency
    # --------------------------------------------------------

    if list(X_train.columns) != FEATURES:
        raise ValueError(
            "Training feature order does not match contract."
        )

    if list(X_test.columns) != FEATURES:
        raise ValueError(
            "Test feature order does not match contract."
        )

    # --------------------------------------------------------
    # Final candidate
    # --------------------------------------------------------

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=BEST_ITERATION + 1,
        max_depth=5,
        min_child_weight=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\nFINAL MODEL")
    print("-" * 70)

    print(
        f"Training rows: "
        f"{len(training_data):,}"
    )

    print(
        f"Training period: "
        f"{training_data['date'].min().date()} → "
        f"{training_data['date'].max().date()}"
    )

    print(
        f"Test period: "
        f"{test['date'].min().date()} → "
        f"{test['date'].max().date()}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Boosting rounds: "
        f"{BEST_ITERATION + 1}"
    )

    # --------------------------------------------------------
    # Train final candidate
    # --------------------------------------------------------

    print("\nTRAINING FINAL CANDIDATE")
    print("-" * 70)

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    actual = y_test.to_numpy()

    # --------------------------------------------------------
    # Final XGBoost metrics
    # --------------------------------------------------------

    xgb_mae = mae(
        actual,
        predictions,
    )

    xgb_rmse = rmse(
        actual,
        predictions,
    )

    # --------------------------------------------------------
    # Baseline predictions
    #
    # Reconstruct them using the complete chronological
    # modeling series, exactly as in the original benchmark.
    # --------------------------------------------------------

    modeling = pd.read_parquet(
        "data/gold/olist/"
        "daily_order_demand_modeling.parquet"
    )

    modeling["date"] = pd.to_datetime(
        modeling["date"]
    )

    modeling = (
        modeling
        .sort_values("date")
        .reset_index(drop=True)
    )

    series = (
        modeling
        .set_index("date")[
            TARGET
        ]
        .sort_index()
    )

    test_dates = pd.DatetimeIndex(
        test["date"]
    )

    naive_predictions = (
        series
        .shift(1)
        .loc[test_dates]
        .to_numpy()
    )

    seasonal_predictions = (
        series
        .shift(7)
        .loc[test_dates]
        .to_numpy()
    )

    moving_average_predictions = (
        series
        .shift(1)
        .rolling(7)
        .mean()
        .loc[test_dates]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Baseline metrics
    # --------------------------------------------------------

    baseline_results = [
        {
            "model": "naive",
            "MAE": mae(
                actual,
                naive_predictions,
            ),
            "RMSE": rmse(
                actual,
                naive_predictions,
            ),
        },
        {
            "model": "seasonal_naive_7",
            "MAE": mae(
                actual,
                seasonal_predictions,
            ),
            "RMSE": rmse(
                actual,
                seasonal_predictions,
            ),
        },
        {
            "model": "moving_average_7",
            "MAE": mae(
                actual,
                moving_average_predictions,
            ),
            "RMSE": rmse(
                actual,
                moving_average_predictions,
            ),
        },
        {
            "model": "xgboost_final",
            "MAE": xgb_mae,
            "RMSE": xgb_rmse,
        },
    ]

    results = pd.DataFrame(
        baseline_results
    )

    # --------------------------------------------------------
    # Relative performance vs naive
    # --------------------------------------------------------

    naive_mae = float(
        results.loc[
            results["model"] == "naive",
            "MAE",
        ].iloc[0]
    )

    results["MAE_vs_naive_pct"] = (
        (
            results["MAE"]
            - naive_mae
        )
        / naive_mae
        * 100
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_output = pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "actual": actual,
            "xgboost_prediction": predictions,
            "naive_prediction": (
                naive_predictions
            ),
            "seasonal_naive_prediction": (
                seasonal_predictions
            ),
            "moving_average_prediction": (
                moving_average_predictions
            ),
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_PATH
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        RESULTS_PATH,
        index=False,
    )

    prediction_output.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nFINAL TEST RESULTS")
    print("-" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nXGBOOST FINAL")
    print("-" * 70)

    print(
        f"MAE:  {xgb_mae:.4f}"
    )

    print(
        f"RMSE: {xgb_rmse:.4f}"
    )

    print("\nMODEL")
    print("-" * 70)

    print(
        f"Saved: {MODEL_PATH}"
    )

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"Results:     {RESULTS_PATH}"
    )

    print(
        f"Predictions: {PREDICTIONS_PATH}"
    )

    print(
        "\nFinal forecasting evaluation complete."
    )


if __name__ == "__main__":
    main()