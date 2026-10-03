from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# NEXUS — Retention Class-Weight Benchmark
# =============================================================================
#
# Experiment:
#
#   Model A: Logistic Regression + class_weight="balanced"
#   Model B: Logistic Regression without class weighting
#
# Everything else remains identical:
#   - Same V1 features
#   - Same temporal train/validation/test split
#   - Same StandardScaler
#   - Same solver
#   - Same max_iter
#   - Same random_state
#
# Questions:
#   1. Does class weighting improve ranking?
#   2. Does class weighting improve rare-event PR-AUC?
#   3. Does removing class weighting improve probability calibration?
#   4. What is the trade-off?
# =============================================================================


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

OUTPUT_PATH = (
    GOLD_DIR
    / "retention_class_weight_benchmark.parquet"
)


TARGET = "repeat_purchase_90d"


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


# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate_model(
    model_name,
    model,
    scaler,
    validation,
    test,
):

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    results = []

    for split_name, data in [
        ("validation", validation),
        ("test", test),
    ]:

        X = data[FEATURES]
        y = data[TARGET]

        X_scaled = scaler.transform(X)

        probabilities = model.predict_proba(
            X_scaled
        )[:, 1]

        # ---------------------------------------------------------------------
        # Ranking metrics
        # ---------------------------------------------------------------------

        roc_auc = roc_auc_score(
            y,
            probabilities,
        )

        pr_auc = average_precision_score(
            y,
            probabilities,
        )

        # ---------------------------------------------------------------------
        # Probability calibration
        # ---------------------------------------------------------------------

        brier = brier_score_loss(
            y,
            probabilities,
        )

        # ---------------------------------------------------------------------
        # Ranking / lift
        # ---------------------------------------------------------------------

        ranking = pd.DataFrame(
            {
                "actual": y.to_numpy(),
                "score": probabilities,
            }
        )

        ranking = ranking.sort_values(
            "score",
            ascending=False,
        ).reset_index(
            drop=True
        )

        baseline_rate = y.mean()

        capacity_metrics = {}

        for capacity in [
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
        ]:

            n = max(
                1,
                int(
                    len(ranking)
                    * capacity
                ),
            )

            selected = ranking.iloc[
                :n
            ]

            captured = (
                selected["actual"]
                .sum()
            )

            precision = (
                captured / n
            )

            recall = (
                captured / y.sum()
                if y.sum() > 0
                else 0.0
            )

            lift = (
                precision / baseline_rate
                if baseline_rate > 0
                else 0.0
            )

            capacity_metrics[
                capacity
            ] = {
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"Rows: {len(data):,}"
        )

        print(
            f"Positive rate: "
            f"{baseline_rate:.6%}"
        )

        print(
            f"ROC-AUC: "
            f"{roc_auc:.6f}"
        )

        print(
            f"PR-AUC: "
            f"{pr_auc:.6f}"
        )

        print(
            f"Brier score: "
            f"{brier:.8f}"
        )

        print("\nRanking:")

        for capacity, metrics in (
            capacity_metrics.items()
        ):

            print(
                f"Top {capacity:>2.0%} | "
                f"Precision={metrics['precision']:.4%} | "
                f"Recall={metrics['recall']:.4%} | "
                f"Lift={metrics['lift']:.3f}x"
            )

        # ---------------------------------------------------------------------
        # Calibration bins
        # ---------------------------------------------------------------------

        fraction_positive, mean_predicted = (
            calibration_curve(
                y,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
        )

        calibration_gap = np.mean(
            np.abs(
                mean_predicted
                - fraction_positive
            )
        )

        print(
            f"\nMean absolute calibration gap: "
            f"{calibration_gap:.6%}"
        )

        results.append(
            {
                "model": model_name,
                "split": split_name,
                "rows": len(data),
                "positive_rate": baseline_rate,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "mean_absolute_calibration_gap":
                    calibration_gap,

                "top_1_precision":
                    capacity_metrics[0.01]["precision"],
                "top_1_recall":
                    capacity_metrics[0.01]["recall"],
                "top_1_lift":
                    capacity_metrics[0.01]["lift"],

                "top_2_precision":
                    capacity_metrics[0.02]["precision"],
                "top_2_recall":
                    capacity_metrics[0.02]["recall"],
                "top_2_lift":
                    capacity_metrics[0.02]["lift"],

                "top_5_precision":
                    capacity_metrics[0.05]["precision"],
                "top_5_recall":
                    capacity_metrics[0.05]["recall"],
                "top_5_lift":
                    capacity_metrics[0.05]["lift"],

                "top_10_precision":
                    capacity_metrics[0.10]["precision"],
                "top_10_recall":
                    capacity_metrics[0.10]["recall"],
                "top_10_lift":
                    capacity_metrics[0.10]["lift"],

                "top_20_precision":
                    capacity_metrics[0.20]["precision"],
                "top_20_recall":
                    capacity_metrics[0.20]["recall"],
                "top_20_lift":
                    capacity_metrics[0.20]["lift"],
            }
        )

    return results


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 80)
    print("NEXUS — Retention Class-Weight Benchmark")
    print("=" * 80)

    # =========================================================================
    # 1. Load temporal splits
    # =========================================================================

    train = pd.read_parquet(
        SPLIT_DIR / "train.parquet"
    )

    validation = pd.read_parquet(
        SPLIT_DIR / "validation.parquet"
    )

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    print("\nDATA")
    print("-" * 80)

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    # =========================================================================
    # 2. Validate features
    # =========================================================================

    for name, data in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        missing = [
            feature
            for feature in FEATURES
            if feature not in data.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing features: "
                + ", ".join(missing)
            )

        if not data[TARGET].isin([0, 1]).all():
            raise ValueError(
                f"{name} contains invalid target values."
            )

    # =========================================================================
    # 3. Prepare matrices
    # =========================================================================

    X_train = train[FEATURES]
    y_train = train[TARGET]

    # =========================================================================
    # 4. Same preprocessing for both models
    # =========================================================================

    scaler_balanced = StandardScaler()

    X_train_balanced = (
        scaler_balanced.fit_transform(
            X_train
        )
    )

    scaler_unweighted = StandardScaler()

    X_train_unweighted = (
        scaler_unweighted.fit_transform(
            X_train
        )
    )

    # =========================================================================
    # 5. Model A — Balanced
    # =========================================================================

    print("\nTraining Model A...")
    print(
        "Logistic Regression + class_weight='balanced'"
    )

    balanced_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    balanced_model.fit(
        X_train_balanced,
        y_train,
    )

    # =========================================================================
    # 6. Model B — Unweighted
    # =========================================================================

    print("\nTraining Model B...")
    print(
        "Logistic Regression without class weighting"
    )

    unweighted_model = LogisticRegression(
        class_weight=None,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    unweighted_model.fit(
        X_train_unweighted,
        y_train,
    )

    # =========================================================================
    # 7. Evaluate both models
    # =========================================================================

    results = []

    results.extend(
        evaluate_model(
            "balanced_logistic",
            balanced_model,
            scaler_balanced,
            validation,
            test,
        )
    )

    results.extend(
        evaluate_model(
            "unweighted_logistic",
            unweighted_model,
            scaler_unweighted,
            validation,
            test,
        )
    )

    # =========================================================================
    # 8. Save results
    # =========================================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # =========================================================================
    # 9. Comparison
    # =========================================================================

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    comparison_columns = [
        "model",
        "split",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "mean_absolute_calibration_gap",
        "top_1_lift",
        "top_5_lift",
        "top_10_lift",
        "top_20_lift",
    ]

    print(
        result_df[
            comparison_columns
        ].to_string(
            index=False,
            formatters={
                "roc_auc": "{:.6f}".format,
                "pr_auc": "{:.6f}".format,
                "brier_score": "{:.8f}".format,
                "mean_absolute_calibration_gap":
                    "{:.6%}".format,
                "top_1_lift":
                    "{:.3f}x".format,
                "top_5_lift":
                    "{:.3f}x".format,
                "top_10_lift":
                    "{:.3f}x".format,
                "top_20_lift":
                    "{:.3f}x".format,
            },
        )
    )

    # =========================================================================
    # 10. Final note
    # =========================================================================

    print("\n" + "=" * 80)
    print("CLASS-WEIGHT BENCHMARK COMPLETE")
    print("=" * 80)

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Balanced and unweighted Logistic Regression are compared "
        "under identical temporal evaluation conditions."
    )

    print(
        "Ranking metrics evaluate customer ordering."
    )

    print(
        "Brier score and calibration gap evaluate probability quality."
    )

    print(
        "Do not select a model based on one metric alone."
    )

    print(
        "The final decision should consider the intended NEXUS use case:"
    )

    print(
        "ranking customers for limited intervention capacity versus "
        "producing calibrated probabilities."
    )


if __name__ == "__main__":
    main()