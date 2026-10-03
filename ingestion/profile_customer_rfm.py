from pathlib import Path
import pandas as pd

GOLD_DIR = Path("data/gold/olist")


def main():
    print("=" * 80)
    print("NEXUS — Customer RFM Distribution Reconnaissance")
    print("=" * 80)

    rfm = pd.read_parquet(GOLD_DIR / "customer_rfm.parquet")

    print("\nDataset")
    print("-" * 80)
    print(f"Customers: {len(rfm):,}")
    print(f"Columns: {len(rfm.columns)}")

    # ------------------------------------------------------------------
    # 1. Frequency distribution
    # ------------------------------------------------------------------
    print("\n1. FREQUENCY DISTRIBUTION")
    print("-" * 80)

    frequency_counts = (
        rfm["frequency"]
        .value_counts()
        .sort_index()
    )

    frequency_pct = (
        frequency_counts / len(rfm) * 100
    )

    frequency_table = pd.DataFrame({
        "customers": frequency_counts,
        "percentage": frequency_pct.round(2)
    })

    print(frequency_table.to_string())

    # ------------------------------------------------------------------
    # 2. Recency distribution
    # ------------------------------------------------------------------
    print("\n2. RECENCY DISTRIBUTION")
    print("-" * 80)

    print(
        rfm["recency_days"]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 3. Monetary distribution
    # ------------------------------------------------------------------
    print("\n3. MONETARY DISTRIBUTION")
    print("-" * 80)

    print(
        rfm["monetary"]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 4. Missingness
    # ------------------------------------------------------------------
    print("\n4. MISSINGNESS")
    print("-" * 80)

    for column in [
        "recency_days",
        "frequency",
        "monetary",
        "payment_observation_status",
    ]:
        missing = rfm[column].isna().sum()
        print(
            f"{column:35s}: "
            f"{missing:,} missing "
            f"({missing / len(rfm) * 100:.2f}%)"
        )

    # ------------------------------------------------------------------
    # 5. Repeat vs one-time
    # ------------------------------------------------------------------
    print("\n5. ONE-TIME VS REPEAT CUSTOMERS")
    print("-" * 80)

    repeat_summary = (
        rfm.groupby("is_repeat_customer")
        .agg(
            customers=("customer_unique_id", "count"),
            median_recency=("recency_days", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary", "median"),
            mean_monetary=("monetary", "mean"),
        )
    )

    repeat_summary["percentage"] = (
        repeat_summary["customers"]
        / len(rfm)
        * 100
    )

    print(
        repeat_summary
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 6. Monetary concentration
    # ------------------------------------------------------------------
    print("\n6. MONETARY CONCENTRATION")
    print("-" * 80)

    monetary_valid = rfm["monetary"].dropna()

    total_value = monetary_valid.sum()

    for percentile in [0.90, 0.95, 0.99]:
        threshold = monetary_valid.quantile(percentile)

        value_above = monetary_valid[
            monetary_valid >= threshold
        ].sum()

        share = value_above / total_value * 100

        customers_above = (
            monetary_valid >= threshold
        ).sum()

        print(
            f"Top {(1 - percentile) * 100:.0f}% threshold "
            f"≥ {threshold:.2f}: "
            f"{customers_above:,} customers, "
            f"{share:.2f}% of monetary value"
        )

    # ------------------------------------------------------------------
    # 7. RFM ties
    # ------------------------------------------------------------------
    print("\n7. UNIQUE VALUES / TIES")
    print("-" * 80)

    for column in [
        "recency_days",
        "frequency",
        "monetary",
    ]:
        unique_count = rfm[column].nunique(
            dropna=True
        )

        print(
            f"{column:20s}: "
            f"{unique_count:,} unique values"
        )

    # ------------------------------------------------------------------
    # 8. Basic correlation
    # ------------------------------------------------------------------
    print("\n8. RFM CORRELATION")
    print("-" * 80)

    print(
        rfm[
            [
                "recency_days",
                "frequency",
                "monetary",
            ]
        ]
        .corr()
        .round(3)
        .to_string()
    )

    print("\n" + "=" * 80)
    print("RFM RECONNAISSANCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()