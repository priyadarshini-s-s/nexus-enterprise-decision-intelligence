from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# NEXUS — XGBoost Early Stopping Benchmark
# ============================================================

SPLIT_DIR = Path(
    "data/gold/olist/forecasting_feature_splits"
)

OUTPUT_PATH = Path(
    "data/gold/olist/"
    "xgboost_early_stopping_results.parquet"
)

TARGET = "order_count"

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


def load_split(name):

    df = pd.read_parquet(
        SPLIT_DIR / f"{name}.parquet"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def main():

    print(
        "NEXUS — XGBoost Early Stopping Benchmark"
    )
    print("=" * 70)

    train = load_split("train")
    validation = load_split("validation")

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    experiments = [
        {
            "name": "early_stop_20",
            "early_stopping_rounds": 20,
        },
        {
            "name": "early_stop_40",
            "early_stopping_rounds": 40,
        },
        {
            "name": "early_stop_60",
            "early_stopping_rounds": 60,
        },
        {
            "name": "early_stop_80",
            "early_stopping_rounds": 80,
        },
    ]

    results = []

    for config in experiments:

        print(
            f"\nRunning: {config['name']}"
        )

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
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
            eval_metric="mae",
            early_stopping_rounds=(
                config["early_stopping_rounds"]
            ),
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

        train_pred = np.maximum(
            model.predict(X_train),
            0,
        )

        validation_pred = np.maximum(
            model.predict(X_validation),
            0,
        )

        result = {
            "experiment": config["name"],
            "early_stopping_rounds": (
                config[
                    "early_stopping_rounds"
                ]
            ),
            "best_iteration": (
                model.best_iteration
            ),
            "best_score": (
                model.best_score
            ),
            "train_mae": mae(
                y_train,
                train_pred,
            ),
            "validation_mae": mae(
                y_validation,
                validation_pred,
            ),
            "train_rmse": rmse(
                y_train,
                train_pred,
            ),
            "validation_rmse": rmse(
                y_validation,
                validation_pred,
            ),
        }

        results.append(result)

        print(
            f"Best iteration: "
            f"{result['best_iteration']}"
        )

        print(
            f"Train MAE:      "
            f"{result['train_mae']:.4f}"
        )

        print(
            f"Validation MAE: "
            f"{result['validation_mae']:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("validation_mae")
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

    print("\nBEST CONFIGURATION")
    print("-" * 70)

    best = results_df.iloc[0]

    print(
        f"Experiment: "
        f"{best['experiment']}"
    )

    print(
        f"Best iteration: "
        f"{int(best['best_iteration'])}"
    )

    print(
        f"Validation MAE: "
        f"{best['validation_mae']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{best['validation_rmse']:.4f}"
    )

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "\nEarly stopping benchmark complete."
    )


if __name__ == "__main__":
    main()