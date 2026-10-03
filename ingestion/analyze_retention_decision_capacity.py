from pathlib import Path

import joblib
import pandas as pd


# =============================================================================
# NEXUS — Retention Decision Capacity Analysis
# =============================================================================
#
# Purpose:
#   Evaluate how well the V1 retention model ranks customers when the business
#   can only act on a fixed percentage of the customer population.
#
# Important:
#   This is a PREDICTIVE ranking analysis.
#   It does NOT estimate causal campaign impact.
#
# Current model:
#   Logistic Regression V1
#
# Current evaluation population:
#   Temporal TEST split
#
# Decision question:
#   "If the business can only target the top X% of customers, how many
#    future repeat purchasers are found, and what is the observed lift?"
# =============================================================================


# =============================================================================
# Paths
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

MODEL_PATH = (
    GOLD_DIR
    / "retention_models"
    / "logistic_retention_baseline.joblib"
)

OUTPUT_PATH = (
    GOLD_DIR
    / "retention_decision_capacity.parquet"
)


# =============================================================================
# V1 feature definition
# =============================================================================
#
# This must remain identical to the feature representation used when the V1
# Logistic Regression model was trained.
# =============================================================================

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

KEY_COLUMNS = [
    "customer_unique_id",
    "snapshot_timestamp",
]


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 80)
    print("NEXUS — Retention Decision Capacity Analysis")
    print("=" * 80)

    # =========================================================================
    # 1. Load temporal TEST data
    # =========================================================================

    test_path = (
        SPLIT_DIR
        / "test.parquet"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found:\n{test_path}"
        )

    test = pd.read_parquet(
        test_path
    )

    print("\nTEST DATA")
    print("-" * 80)

    print(
        f"Test observations: {len(test):,}"
    )

    # =========================================================================
    # 2. Validate required columns
    # =========================================================================

    required_columns = (
        KEY_COLUMNS
        + FEATURES
        + [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    if missing_columns:
        raise ValueError(
            "The test dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # =========================================================================
    # 3. Basic target validation
    # =========================================================================

    if not test[TARGET].isin([0, 1]).all():
        raise ValueError(
            f"{TARGET} must contain only 0/1 values."
        )

    actual_repeat_purchasers = int(
        test[TARGET].sum()
    )

    baseline_rate = (
        test[TARGET].mean()
    )

    print(
        f"Actual repeat purchasers: "
        f"{actual_repeat_purchasers:,}"
    )

    print(
        f"Baseline repeat-purchase rate: "
        f"{baseline_rate:.6%}"
    )

    # =========================================================================
    # 4. Validate test-set grain
    # =========================================================================

    duplicate_keys = test.duplicated(
        subset=KEY_COLUMNS
    ).sum()

    print(
        f"Duplicate customer × snapshot rows: "
        f"{duplicate_keys:,}"
    )

    if duplicate_keys != 0:
        raise ValueError(
            "Duplicate customer × snapshot observations detected."
        )

    # =========================================================================
    # 5. Load V1 model
    # =========================================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Retention model not found:\n{MODEL_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    print("\nMODEL ARTIFACT")
    print("-" * 80)

    print(
        f"Loaded: {MODEL_PATH}"
    )

    # =========================================================================
    # 6. Handle the actual saved artifact
    # =========================================================================
    #
    # Your V1 artifact is a scikit-learn Pipeline.
    #
    # Therefore:
    #
    #     X
    #     ↓
    #     Pipeline
    #     ↓
    #     predict_proba()
    #
    # We intentionally do NOT manually apply StandardScaler here because the
    # Pipeline already contains the preprocessing step.
    # =========================================================================

    if hasattr(
        artifact,
        "predict_proba",
    ):

        pipeline = artifact

        print(
            "Artifact type: scikit-learn Pipeline/model"
        )

    elif isinstance(
        artifact,
        dict,
    ):

        # Backward-compatible handling in case a dictionary artifact is
        # encountered later.

        if "model" not in artifact:
            raise ValueError(
                "Dictionary model artifact does not contain a 'model' key."
            )

        pipeline = artifact["model"]

        print(
            "Artifact type: dictionary containing model"
        )

    else:

        raise TypeError(
            "Unsupported model artifact type: "
            f"{type(artifact)}"
        )

    if not hasattr(
        pipeline,
        "predict_proba",
    ):
        raise TypeError(
            "Loaded model does not support predict_proba()."
        )

    # =========================================================================
    # 7. Prepare features
    # =========================================================================

    X = test[
        FEATURES
    ]

    y = test[
        TARGET
    ]

    # Check for missing feature values.

    missing_feature_values = (
        X.isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values: "
        f"{missing_feature_values:,}"
    )

    if missing_feature_values != 0:
        raise ValueError(
            "Model features contain missing values."
        )

    # =========================================================================
    # 8. Generate probability scores
    # =========================================================================

    print("\nGenerating retention scores...")

    probabilities = pipeline.predict_proba(
        X
    )[:, 1]

    print(
        f"Generated scores: "
        f"{len(probabilities):,}"
    )

    # =========================================================================
    # 9. Build ranking table
    # =========================================================================

    ranked = test[
        KEY_COLUMNS
        + [TARGET]
    ].copy()

    ranked["retention_score"] = (
        probabilities
    )

    # Highest predicted probability first.

    ranked = ranked.sort_values(
        "retention_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # Rank starts at 1.

    ranked["rank"] = (
        ranked.index + 1
    )

    ranked["rank_percentile"] = (
        ranked["rank"]
        / len(ranked)
    )

    # =========================================================================
    # 10. Capacity analysis
    # =========================================================================
    #
    # We evaluate several operational capacities:
    #
    # Top 1%
    # Top 2%
    # Top 5%
    # Top 10%
    # Top 20%
    # Top 30%
    # Top 50%
    #
    # This avoids selecting an arbitrary probability threshold.
    # =========================================================================

    print("\n" + "-" * 80)
    print("CAPACITY / RANKING ANALYSIS")
    print("-" * 80)

    capacities = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]

    results = []

    for capacity in capacities:

        number_to_target = max(
            1,
            int(
                len(ranked)
                * capacity
            ),
        )

        selected = ranked.iloc[
            :number_to_target
        ]

        captured = int(
            selected[TARGET].sum()
        )

        precision = (
            captured
            / number_to_target
        )

        recall = (
            captured
            / actual_repeat_purchasers
            if actual_repeat_purchasers > 0
            else 0.0
        )

        lift = (
            precision
            / baseline_rate
            if baseline_rate > 0
            else 0.0
        )

        results.append(
            {
                "capacity": capacity,
                "customers_targeted": number_to_target,
                "repeat_purchasers_captured": captured,
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }
        )

        print(
            f"Top {capacity:>5.0%} | "
            f"Targeted={number_to_target:>7,} | "
            f"Captured={captured:>4,} | "
            f"Precision={precision:>8.4%} | "
            f"Recall={recall:>8.4%} | "
            f"Lift={lift:>6.3f}x"
        )

    # =========================================================================
    # 11. Convert results to DataFrame
    # =========================================================================

    result_df = pd.DataFrame(
        results
    )

    # =========================================================================
    # 12. Save decision-capacity table
    # =========================================================================

    result_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # =========================================================================
    # 13. Sanity check against the previously measured V1 results
    # =========================================================================
    #
    # These are the previously established V1 test results from our model
    # evaluation. They should reproduce closely.
    #
    # We use approximate tolerance rather than exact equality because ranking
    # implementation details can affect boundary observations.
    # =========================================================================

    expected_lifts = {
        0.01: 7.000,
        0.02: 5.059,
        0.05: 3.238,
        0.10: 2.496,
        0.20: 1.792,
    }

    print("\n" + "-" * 80)
    print("V1 CONSISTENCY CHECK")
    print("-" * 80)

    for capacity, expected in expected_lifts.items():

        row = result_df[
            result_df["capacity"] == capacity
        ]

        if row.empty:
            print(
                f"Top {capacity:.0%}: "
                f"NOT FOUND"
            )
            continue

        actual = float(
            row["lift"].iloc[0]
        )

        difference = (
            actual - expected
        )

        print(
            f"Top {capacity:.0%} | "
            f"Expected lift ≈ {expected:.3f}x | "
            f"Actual lift = {actual:.3f}x | "
            f"Difference = {difference:+.3f}x"
        )

    # =========================================================================
    # 14. Final summary
    # =========================================================================

    print("\n" + "=" * 80)
    print("DECISION CAPACITY ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "The model is being evaluated as a ranking system."
    )

    print(
        "Lift measures how much higher the observed repeat-purchase "
        "rate is within a model-selected group compared with the "
        "overall test population."
    )

    print(
        "\nImportant:"
    )

    print(
        "These results measure predictive association."
    )

    print(
        "They do NOT prove that contacting, discounting, or otherwise "
        "intervening on a customer causes an additional purchase."
    )

    print(
        "\nCausal intervention analysis will be handled separately "
        "using experimental/causal data in the NEXUS experimentation "
        "module."
    )


if __name__ == "__main__":
    main()