from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "olist"

INPUT_PATH = GOLD_DIR / "retention_training_table.parquet"
OUTPUT_DIR = GOLD_DIR / "retention_splits"


TRAIN_END = pd.Timestamp("2017-12-31")
VALIDATION_END = pd.Timestamp("2018-03-31")


def summarize_split(name, df):

    positives = int(
        df["repeat_purchase_90d"].sum()
    )

    total = len(df)

    positive_rate = (
        positives / total * 100
        if total
        else 0
    )

    print(
        f"\n{name}"
    )

    print("-" * 70)

    print(
        f"Snapshots: "
        f"{df['snapshot_timestamp'].min().date()} "
        f"→ "
        f"{df['snapshot_timestamp'].max().date()}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Unique customers: "
        f"{df['customer_unique_id'].nunique():,}"
    )

    print(
        f"Positive targets: {positives:,}"
    )

    print(
        f"Positive rate: {positive_rate:.4f}%"
    )


def main():

    print("=" * 80)
    print("NEXUS — Temporal Retention Train / Validation / Test Split")
    print("=" * 80)

    df = pd.read_parquet(
        INPUT_PATH
    )

    df["snapshot_timestamp"] = pd.to_datetime(
        df["snapshot_timestamp"]
    )

    # -------------------------------------------------------------------------
    # Temporal split
    # -------------------------------------------------------------------------

    train = df[
        df["snapshot_timestamp"]
        <= TRAIN_END
    ].copy()

    validation = df[
        (
            df["snapshot_timestamp"]
            > TRAIN_END
        )
        &
        (
            df["snapshot_timestamp"]
            <= VALIDATION_END
        )
    ].copy()

    test = df[
        df["snapshot_timestamp"]
        > VALIDATION_END
    ].copy()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    assert (
        train["snapshot_timestamp"].max()
        < validation["snapshot_timestamp"].min()
    )

    assert (
        validation["snapshot_timestamp"].max()
        < test["snapshot_timestamp"].min()
    )

    assert (
        len(train)
        + len(validation)
        + len(test)
        == len(df)
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        OUTPUT_DIR / "train.parquet",
        index=False,
    )

    validation.to_parquet(
        OUTPUT_DIR / "validation.parquet",
        index=False,
    )

    test.to_parquet(
        OUTPUT_DIR / "test.parquet",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    summarize_split(
        "TRAIN",
        train,
    )

    summarize_split(
        "VALIDATION",
        validation,
    )

    summarize_split(
        "TEST",
        test,
    )

    print("\n" + "=" * 80)
    print("TEMPORAL SPLIT COMPLETE")
    print("=" * 80)

    print(
        "\nTrain / validation / test separation verified."
    )

    print(
        f"\nOutput directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()