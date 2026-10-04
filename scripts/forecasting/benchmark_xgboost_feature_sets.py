from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# NEXUS — XGBoost Feature Ablation Benchmark
# ============================================================

SPLIT_DIR = Path(
    "data/gold/olist/forecasting_feature_splits"
)

OUTPUT_PATH = Path(
    "data/gold/olist/"
    "xgboost_feature_ablation_results.parquet"
)


TARGET = "order_count"


# ------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------

LAG_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
]


CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "time_index",
]


ROLLING_MEAN_FEATURES = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
]


ALL_FEATURES = [
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


EXPERIMENTS = {
    "A_lags_only": LAG_FEATURES,

    "B_lags_calendar": (
        LAG_FEATURES
        + CALENDAR_FEATURES
    ),

    "C_lags_calendar_rolling_means": (
        LAG_FEATURES
        + CALENDAR_FEATURES
        + ROLLING_MEAN_FEATURES
    ),

    "D_full_v1": ALL_FEATURES,
}


def load_split(name: str) -> pd.DataFrame:

    path = (
        SPLIT_DIR
        / f"{name}.parquet"
    )

    df = pd.read_parquet(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def mae(
    actual,
    predicted,
) -> float:

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def rmse(
    actual,
    predicted,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def train_and_evaluate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    experiment_name: str,
):

    X_train = train[
        feature_columns
    ]

    y_train = train[TARGET]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        TARGET
    ]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    predictions = model.predict(
        X_validation
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    validation_mae = mae(
        y_validation.to_numpy(),
        predictions,
    )

    validation_rmse = rmse(
        y_validation.to_numpy(),
        predictions,
    )

    # --------------------------------------------------------
    # Training error
    #
    # This is useful for detecting obvious overfitting.
    # --------------------------------------------------------

    train_predictions = model.predict(
        X_train
    )

    train_predictions = np.maximum(
        train_predictions,
        0,
    )

    train_mae = mae(
        y_train.to_numpy(),
        train_predictions,
    )

    train_rmse = rmse(
        y_train.to_numpy(),
        train_predictions,
    )

    return {
        "experiment": experiment_name,
        "feature_count": len(
            feature_columns
        ),
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
    }


def main():

    print(
        "NEXUS — XGBoost Feature Ablation Benchmark"
    )
    print("=" * 70)

    train = load_split("train")
    validation = load_split(
        "validation"
    )

    print("\nDATA")
    print("-" * 70)

    print(
        f"Train rows:      {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    results = []

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for experiment_name, features in (
        EXPERIMENTS.items()
    ):

        print(
            f"\nRunning: {experiment_name}"
        )

        print(
            f"Features: {len(features)}"
        )

        result = train_and_evaluate(
            train=train,
            validation=validation,
            feature_columns=features,
            experiment_name=experiment_name,
        )

        results.append(result)

        print(
            f"Train MAE:      "
            f"{result['train_mae']:.4f}"
        )

        print(
            f"Validation MAE: "
            f"{result['validation_mae']:.4f}"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "validation_mae"
    ).reset_index(drop=True)

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
        f"Experiment: "
        f"{best['experiment']}"
    )

    print(
        f"Features:   "
        f"{int(best['feature_count'])}"
    )

    print(
        f"Train MAE:  "
        f"{best['train_mae']:.4f}"
    )

    print(
        f"Validation MAE: "
        f"{best['validation_mae']:.4f}"
    )

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nFeature ablation benchmark complete."
    )


if __name__ == "__main__":
    main()