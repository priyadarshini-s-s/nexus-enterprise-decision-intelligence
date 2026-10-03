from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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


def main():

    print("=" * 80)
    print("NEXUS — Retention Model Ranking Agreement")
    print("=" * 80)

    # =========================================================================
    # Load temporal data
    # =========================================================================

    train = pd.read_parquet(
        SPLIT_DIR / "train.parquet"
    )

    test = pd.read_parquet(
        SPLIT_DIR / "test.parquet"
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]

    # =========================================================================
    # Same preprocessing
    # =========================================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # =========================================================================
    # Balanced model
    # =========================================================================

    balanced = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    balanced.fit(
        X_train_scaled,
        y_train,
    )

    balanced_scores = balanced.predict_proba(
        X_test_scaled
    )[:, 1]

    # =========================================================================
    # Unweighted model
    # =========================================================================

    unweighted = LogisticRegression(
        class_weight=None,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    unweighted.fit(
        X_train_scaled,
        y_train,
    )

    unweighted_scores = unweighted.predict_proba(
        X_test_scaled
    )[:, 1]

    # =========================================================================
    # Ranking comparison
    # =========================================================================

    comparison = test[
        [
            "customer_unique_id",
            "snapshot_timestamp",
        ]
    ].copy()

    comparison["balanced_score"] = (
        balanced_scores
    )

    comparison["unweighted_score"] = (
        unweighted_scores
    )

    comparison["balanced_rank"] = (
        comparison[
            "balanced_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    comparison["unweighted_rank"] = (
        comparison[
            "unweighted_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    # =========================================================================
    # Spearman rank correlation
    # =========================================================================

    rank_correlation, p_value = spearmanr(
        comparison["balanced_rank"],
        comparison["unweighted_rank"],
    )

    print("\nOVERALL RANK AGREEMENT")
    print("-" * 80)

    print(
        f"Spearman rank correlation: "
        f"{rank_correlation:.6f}"
    )

    print(
        f"p-value: "
        f"{p_value:.6e}"
    )

    # =========================================================================
    # Top-K overlap
    # =========================================================================

    print("\nTOP-K CUSTOMER OVERLAP")
    print("-" * 80)

    overlap_results = []

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
                len(comparison)
                * capacity
            ),
        )

        balanced_top = set(
            comparison
            .nsmallest(
                n,
                "balanced_rank",
            )[
                "customer_unique_id"
            ]
        )

        unweighted_top = set(
            comparison
            .nsmallest(
                n,
                "unweighted_rank",
            )[
                "customer_unique_id"
            ]
        )

        intersection = (
            balanced_top
            & unweighted_top
        )

        union = (
            balanced_top
            | unweighted_top
        )

        overlap_count = len(
            intersection
        )

        overlap_rate = (
            overlap_count / n
        )

        jaccard = (
            overlap_count
            / len(union)
            if union
            else 0
        )

        overlap_results.append(
            {
                "capacity": capacity,
                "n": n,
                "overlap_count": overlap_count,
                "overlap_rate": overlap_rate,
                "jaccard_similarity": jaccard,
            }
        )

        print(
            f"Top {capacity:>2.0%} | "
            f"N={n:,} | "
            f"Overlap={overlap_count:,} | "
            f"Overlap rate={overlap_rate:.4%} | "
            f"Jaccard={jaccard:.4%}"
        )

    # =========================================================================
    # Score distribution comparison
    # =========================================================================

    print("\nSCORE DISTRIBUTION")
    print("-" * 80)

    distribution = pd.DataFrame(
        {
            "balanced_score": balanced_scores,
            "unweighted_score": unweighted_scores,
        }
    )

    print(
        distribution.describe(
            percentiles=[
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    # =========================================================================
    # Save comparison
    # =========================================================================

    output_path = (
        GOLD_DIR
        / "retention_model_ranking_comparison.parquet"
    )

    comparison.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RANKING COMPARISON COMPLETE")
    print("=" * 80)

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()