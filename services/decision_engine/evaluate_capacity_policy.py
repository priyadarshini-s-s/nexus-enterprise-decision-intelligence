from pathlib import Path

import pandas as pd

from services.decision_engine.population_policy import (
    prioritize_customers,
)


INPUT_PATH = Path(
    "data/gold/olist/retention_decision_population.parquet"
)

EVALUATION_OUTPUT_PATH = Path(
    "data/gold/olist/retention_capacity_policy_evaluation.parquet"
)

DECISION_OUTPUT_PATH = Path(
    "data/gold/olist/retention_decisions.parquet"
)

CAPACITIES = [0.01, 0.05, 0.10, 0.20]


def evaluate_capacity_at_snapshot(
    snapshot: pd.DataFrame,
    capacity: float,
):
    population = snapshot[
        [
            "customer_unique_id",
            "repeat_purchase_probability",
        ]
    ].copy()

    decisions = prioritize_customers(
        population,
        capacity_fraction=capacity,
        customer_id_column="customer_unique_id",
    )

    actual = snapshot[
        [
            "customer_unique_id",
            "repeat_purchase_90d",
        ]
    ].copy()

    evaluation = decisions.merge(
        actual,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    targeted = (
        evaluation["priority"] == "prioritize"
    )

    actual_positive = (
        evaluation["repeat_purchase_90d"] == 1
    )

    targeted_count = int(targeted.sum())

    captured_positive = int(
        (
            targeted
            & actual_positive
        ).sum()
    )

    total_positive = int(
        actual_positive.sum()
    )

    precision = (
        captured_positive / targeted_count
        if targeted_count > 0
        else 0.0
    )

    recall = (
        captured_positive / total_positive
        if total_positive > 0
        else 0.0
    )

    baseline_rate = (
        total_positive / len(evaluation)
        if len(evaluation) > 0
        else 0.0
    )

    lift = (
        precision / baseline_rate
        if baseline_rate > 0
        else 0.0
    )

    evaluation["snapshot_timestamp"] = (
        snapshot["snapshot_timestamp"].iloc[0]
    )

    return evaluation, {
        "snapshot_timestamp": snapshot[
            "snapshot_timestamp"
        ].iloc[0],
        "capacity_fraction": capacity,
        "population_size": len(evaluation),
        "targeted_count": targeted_count,
        "total_positive": total_positive,
        "captured_positive": captured_positive,
        "baseline_positive_rate": baseline_rate,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def main():

    print("=" * 80)
    print("NEXUS — Decision Engine Capacity Evaluation")
    print("=" * 80)

    print("\nLoading scored population...")

    df = pd.read_parquet(INPUT_PATH)

    required = {
        "customer_unique_id",
        "snapshot_timestamp",
        "repeat_purchase_probability",
        "repeat_purchase_90d",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(f"Rows: {len(df):,}")

    snapshots = sorted(
        df["snapshot_timestamp"].unique()
    )

    print(f"Snapshots: {len(snapshots)}")

    evaluation_results = []
    decision_results = []

    for snapshot_timestamp in snapshots:

        snapshot = df[
            df["snapshot_timestamp"]
            == snapshot_timestamp
        ].copy()

        duplicate_customers = snapshot.duplicated(
            subset=["customer_unique_id"]
        ).sum()

        if duplicate_customers != 0:
            raise ValueError(
                f"Snapshot {snapshot_timestamp} contains "
                f"{duplicate_customers} duplicate customers."
            )

        for capacity in CAPACITIES:

            decisions, metrics = (
                evaluate_capacity_at_snapshot(
                    snapshot,
                    capacity,
                )
            )

            evaluation_results.append(metrics)

            decision_results.append(
                decisions[
                    [
                        "customer_unique_id",
                        "snapshot_timestamp",
                        "repeat_purchase_probability",
                        "decision_rank",
                        "priority",
                        "capacity_fraction",
                        "target_population_size",
                        "repeat_purchase_90d",
                    ]
                ]
            )

    evaluation_df = pd.DataFrame(
        evaluation_results
    )

    decisions_df = pd.concat(
        decision_results,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Validate decision artifact
    # ------------------------------------------------------------------

    expected_decision_rows = (
        len(df) * len(CAPACITIES)
    )

    if len(decisions_df) != expected_decision_rows:
        raise ValueError(
            "Decision artifact row-count mismatch: "
            f"{len(decisions_df)} != "
            f"{expected_decision_rows}"
        )

    duplicate_decisions = decisions_df.duplicated(
        subset=[
            "customer_unique_id",
            "snapshot_timestamp",
            "capacity_fraction",
        ]
    ).sum()

    if duplicate_decisions != 0:
        raise ValueError(
            "Duplicate customer × snapshot × capacity "
            f"decisions: {duplicate_decisions}"
        )

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------

    EVALUATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df.to_parquet(
        EVALUATION_OUTPUT_PATH,
        index=False,
    )

    decisions_df.to_parquet(
        DECISION_OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    print("\nSnapshot-level evaluation:")

    print(
        evaluation_df[
            [
                "snapshot_timestamp",
                "capacity_fraction",
                "population_size",
                "targeted_count",
                "captured_positive",
                "baseline_positive_rate",
                "precision",
                "recall",
                "lift",
            ]
        ].to_string(index=False)
    )

    summary = (
        evaluation_df
        .groupby("capacity_fraction")
        .agg(
            snapshots=(
                "snapshot_timestamp",
                "count",
            ),
            mean_population_size=(
                "population_size",
                "mean",
            ),
            mean_targeted_count=(
                "targeted_count",
                "mean",
            ),
            total_captured_positive=(
                "captured_positive",
                "sum",
            ),
            mean_baseline_positive_rate=(
                "baseline_positive_rate",
                "mean",
            ),
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_lift=(
                "lift",
                "mean",
            ),
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("CAPACITY SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(index=False)
    )

    print("\nDecision artifact:")
    print(DECISION_OUTPUT_PATH)

    print(
        f"Decision rows: {len(decisions_df):,}"
    )

    print("\nEvaluation artifact:")
    print(EVALUATION_OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("DECISION ENGINE CAPACITY EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()