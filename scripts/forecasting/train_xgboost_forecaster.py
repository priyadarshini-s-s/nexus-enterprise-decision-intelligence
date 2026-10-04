from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# NEXUS — XGBoost Demand Forecasting V1
# ============================================================

SPLIT_DIR = Path(
    "data/gold/olist/forecasting_feature_splits"
)

OUTPUT_DIR = Path(
    "data/gold/olist/forecasting_models"
)

MODEL_PATH = (
    OUTPUT_DIR / "xgboost_demand_v1.json"
)

RESULTS_PATH = (
    Path(
        "data/gold/olist/"
        "xgboost_forecasting_v1_results.parquet"
    )
)


TARGET = "order_count"

DROP_COLUMNS = [
    "date",
    TARGET,
]


def load_split(name: str) -> pd.DataFrame:

    path = SPLIT_DIR / f"{name}.parquet"

    df = pd.read_parquet(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(
        columns=DROP_COLUMNS
    )

    y = df[TARGET]

    return X, y


def mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def main():

    print(
        "NEXUS — XGBoost Demand Forecasting V1"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    print("\nDATA")
    print("-" * 65)

    print(
        f"Train:      {len(train):,} rows"
    )

    print(
        f"Validation: {len(validation):,} rows"
    )

    print(
        f"Test:       {len(test):,} rows"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X_train, y_train = prepare_features(
        train
    )

    X_validation, y_validation = (
        prepare_features(validation)
    )

    X_test, y_test = prepare_features(
        test
    )

    # --------------------------------------------------------
    # Feature consistency
    # --------------------------------------------------------

    if list(X_train.columns) != list(
        X_validation.columns
    ):
        raise ValueError(
            "Train and validation features differ."
        )

    if list(X_train.columns) != list(
        X_test.columns
    ):
        raise ValueError(
            "Train and test features differ."
        )

    print(
        f"Features: {X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    print("\nTRAINING")
    print("-" * 65)

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

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    # Demand cannot be negative
    validation_pred = np.maximum(
        validation_pred,
        0,
    )

    test_pred = np.maximum(
        test_pred,
        0,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    validation_mae = mae(
        y_validation.to_numpy(),
        validation_pred,
    )

    validation_rmse = rmse(
        y_validation.to_numpy(),
        validation_pred,
    )

    test_mae = mae(
        y_test.to_numpy(),
        test_pred,
    )

    test_rmse = rmse(
        y_test.to_numpy(),
        test_pred,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "split": "validation",
                "model": "xgboost_v1",
                "MAE": validation_mae,
                "RMSE": validation_rmse,
                "evaluation_days": len(
                    y_validation
                ),
            },
            {
                "split": "test",
                "model": "xgboost_v1",
                "MAE": test_mae,
                "RMSE": test_rmse,
                "evaluation_days": len(
                    y_test
                ),
            },
        ]
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nRESULTS")
    print("-" * 65)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nTOP FEATURE IMPORTANCE")
    print("-" * 65)

    print(
        importance.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nMODEL")
    print("-" * 65)

    print(
        f"Saved: {MODEL_PATH}"
    )

    print("\nRESULTS")
    print("-" * 65)

    print(
        f"Saved: {RESULTS_PATH}"
    )

    print(
        "\nXGBoost forecasting V1 complete."
    )


if __name__ == "__main__":
    main()