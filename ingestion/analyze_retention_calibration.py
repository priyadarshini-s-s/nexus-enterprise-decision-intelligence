from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
)

SPLIT_DIR = (
    GOLD_DIR
    / "retention_splits"
)

MODEL_PATH = (
    GOLD_DIR
    / "retention_models"
    / "logistic_retention_baseline.joblib"
)

OUTPUT_PATH = (
    GOLD_DIR
    / "retention_calibration.parquet"
)


FEATURES = [
    "orders_to_date",
    "total_spend_to_date",
    "total_order_items_to_date",
    "unique_products_to_date",
    "unique_categories_to_date",
    "unique_sellers_to_date",
    "recency_days",
    "average_order_value",
    "average_items_per_order",
    "orders_30d",
    "spend_30d",
    "items_30d",
    "orders_60d",
    "spend_60d",
    "items_60d",
    "orders_90d",
    "spend_90d",
    "items_90d",
    "average_review_score_to_date",
    "payment_observation_available",
    "review_observation_available",
    "item_observation_available",
]

TARGET = "repeat_purchase_90d"


def main():

    print("=" * 80)
    print("NEXUS — Retention Probability Calibration Analysis")
    print("=" * 80)

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    artifact = joblib.load(
        MODEL_PATH
    )

    if hasattr(
        artifact,
        "predict_proba",
    ):
        pipeline = artifact

    elif isinstance(
        artifact,
        dict,
    ):
        pipeline = artifact["model"]

    else:
        raise TypeError(
            f"Unsupported artifact type: {type(artifact)}"
        )

    X = test[FEATURES]
    y = test[TARGET]

    probabilities = pipeline.predict_proba(
        X
    )[:, 1]

    # -------------------------------------------------------------------------
    # Brier score
    # -------------------------------------------------------------------------

    brier = brier_score_loss(
        y,
        probabilities,
    )

    print(
        f"\nBrier score: {brier:.8f}"
    )

    # -------------------------------------------------------------------------
    # Calibration curve
    # -------------------------------------------------------------------------

    fraction_positive, mean_predicted = calibration_curve(
        y,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    calibration = pd.DataFrame(
        {
            "bin": np.arange(
                1,
                len(fraction_positive) + 1,
            ),
            "mean_predicted_probability":
                mean_predicted,
            "observed_positive_rate":
                fraction_positive,
        }
    )

    calibration["absolute_gap"] = (
        calibration[
            "mean_predicted_probability"
        ]
        - calibration[
            "observed_positive_rate"
        ]
    ).abs()

    print("\nCalibration bins")
    print("-" * 80)

    print(
        calibration.to_string(
            index=False,
            formatters={
                "mean_predicted_probability":
                    "{:.6%}".format,
                "observed_positive_rate":
                    "{:.6%}".format,
                "absolute_gap":
                    "{:.6%}".format,
            },
        )
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    calibration.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()