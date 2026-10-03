from pathlib import Path
import pandas as pd

GOLD_DIR = Path("data/gold/olist")


def main():
    print("=" * 80)
    print("NEXUS — Interpretable RFM Baseline Analysis")
    print("=" * 80)

    rfm = pd.read_parquet(GOLD_DIR / "customer_rfm.parquet")

    # ------------------------------------------------------------------
    # 1. Frequency classes
    # ------------------------------------------------------------------
    print("\n1. FREQUENCY CLASSES")
    print("-" * 80)

    rfm["frequency_class"] = pd.cut(
        rfm["frequency"],
        bins=[0, 1, 2, 3, float("inf")],
        labels=["1 order", "2 orders", "3 orders", "4+ orders"],
        include_lowest=True,
    )

    frequency_summary = (
        rfm.groupby("frequency_class", observed=False)
        .agg(
            customers=("customer_unique_id", "count"),
            monetary=("monetary", "sum"),
            median_monetary=("monetary", "median"),
            median_recency=("recency_days", "median"),
        )
    )

    frequency_summary["customer_pct"] = (
        frequency_summary["customers"]
        / len(rfm)
        * 100
    )

    total_monetary = rfm["monetary"].sum()

    frequency_summary["monetary_pct"] = (
        frequency_summary["monetary"]
        / total_monetary
        * 100
    )

    print(
        frequency_summary[
            [
                "customers",
                "customer_pct",
                "median_recency",
                "median_monetary",
                "monetary",
                "monetary_pct",
            ]
        ]
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 2. Recency bands
    # ------------------------------------------------------------------
    print("\n2. RECENCY BANDS")
    print("-" * 80)

    rfm["recency_band"] = pd.cut(
        rfm["recency_days"],
        bins=[-1, 90, 180, 365, 545, float("inf")],
        labels=[
            "0-90 days",
            "91-180 days",
            "181-365 days",
            "366-545 days",
            "546+ days",
        ],
    )

    recency_summary = (
        rfm.groupby("recency_band", observed=False)
        .agg(
            customers=("customer_unique_id", "count"),
            median_monetary=("monetary", "median"),
            mean_monetary=("monetary", "mean"),
            median_frequency=("frequency", "median"),
        )
    )

    recency_summary["customer_pct"] = (
        recency_summary["customers"]
        / len(rfm)
        * 100
    )

    print(
        recency_summary[
            [
                "customers",
                "customer_pct",
                "median_monetary",
                "mean_monetary",
                "median_frequency",
            ]
        ]
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 3. Monetary bands
    # ------------------------------------------------------------------
    print("\n3. MONETARY BANDS")
    print("-" * 80)

    monetary_bins = [
        -0.01,
        50,
        100,
        250,
        500,
        1000,
        float("inf"),
    ]

    monetary_labels = [
        "<= R$50",
        "R$50-R$100",
        "R$100-R$250",
        "R$250-R$500",
        "R$500-R$1,000",
        "> R$1,000",
    ]

    rfm["monetary_band"] = pd.cut(
        rfm["monetary"],
        bins=monetary_bins,
        labels=monetary_labels,
        include_lowest=True,
    )

    monetary_summary = (
        rfm.groupby("monetary_band", observed=False)
        .agg(
            customers=("customer_unique_id", "count"),
            total_monetary=("monetary", "sum"),
            median_monetary=("monetary", "median"),
            median_recency=("recency_days", "median"),
            median_frequency=("frequency", "median"),
        )
    )

    monetary_summary["customer_pct"] = (
        monetary_summary["customers"]
        / len(rfm)
        * 100
    )

    monetary_summary["value_pct"] = (
        monetary_summary["total_monetary"]
        / total_monetary
        * 100
    )

    print(
        monetary_summary[
            [
                "customers",
                "customer_pct",
                "median_monetary",
                "total_monetary",
                "value_pct",
                "median_recency",
                "median_frequency",
            ]
        ]
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 4. One-time vs repeat
    # ------------------------------------------------------------------
    print("\n4. ONE-TIME VS REPEAT × RECENCY")
    print("-" * 80)

    repeat_recency = pd.crosstab(
        rfm["recency_band"],
        rfm["is_repeat_customer"],
        margins=True,
    )

    repeat_recency = repeat_recency.rename(
        columns={
            False: "one_time",
            True: "repeat",
        }
    )

    print(repeat_recency.to_string())

    # ------------------------------------------------------------------
    # 5. One-time vs repeat × monetary
    # ------------------------------------------------------------------
    print("\n5. ONE-TIME VS REPEAT × MONETARY")
    print("-" * 80)

    repeat_monetary = pd.crosstab(
        rfm["monetary_band"],
        rfm["is_repeat_customer"],
        margins=True,
    )

    repeat_monetary = repeat_monetary.rename(
        columns={
            False: "one_time",
            True: "repeat",
        }
    )

    print(repeat_monetary.to_string())

    # ------------------------------------------------------------------
    # 6. Repeat customer value
    # ------------------------------------------------------------------
    print("\n6. REPEAT CUSTOMER VALUE")
    print("-" * 80)

    repeat = rfm[rfm["is_repeat_customer"]].copy()

    print(f"Repeat customers: {len(repeat):,}")
    print(
        f"Repeat customer monetary value: "
        f"R${repeat['monetary'].sum():,.2f}"
    )

    print(
        f"Share of total monetary value: "
        f"{repeat['monetary'].sum() / total_monetary * 100:.2f}%"
    )

    print(
        f"Median repeat frequency: "
        f"{repeat['frequency'].median():.2f}"
    )

    print(
        f"Median repeat monetary value: "
        f"R${repeat['monetary'].median():,.2f}"
    )

    # ------------------------------------------------------------------
    # 7. Candidate interpretable segments
    # ------------------------------------------------------------------
    print("\n7. CANDIDATE BUSINESS SEGMENTS")
    print("-" * 80)

    def classify(row):
        frequency = row["frequency"]
        recency = row["recency_days"]
        monetary = row["monetary"]

        if pd.isna(monetary):
            return "Payment data unavailable"

        if frequency == 1:
            if recency <= 180 and monetary >= 250:
                return "Recent high-value one-time"

            if recency <= 180:
                return "Recent one-time"

            if recency > 365 and monetary >= 250:
                return "Older high-value one-time"

            return "Older one-time"

        # Repeat customers
        if recency <= 180 and frequency >= 3:
            return "Recent frequent repeat"

        if recency <= 180:
            return "Recent repeat"

        if recency > 365:
            return "Lapsed repeat"

        return "Established repeat"

    rfm["baseline_segment"] = rfm.apply(
        classify,
        axis=1,
    )

    segment_summary = (
        rfm.groupby("baseline_segment", observed=False)
        .agg(
            customers=("customer_unique_id", "count"),
            total_monetary=("monetary", "sum"),
            median_monetary=("monetary", "median"),
            median_recency=("recency_days", "median"),
            median_frequency=("frequency", "median"),
        )
        .sort_values("total_monetary", ascending=False)
    )

    segment_summary["customer_pct"] = (
        segment_summary["customers"]
        / len(rfm)
        * 100
    )

    segment_summary["value_pct"] = (
        segment_summary["total_monetary"]
        / total_monetary
        * 100
    )

    print(
        segment_summary[
            [
                "customers",
                "customer_pct",
                "median_recency",
                "median_frequency",
                "median_monetary",
                "total_monetary",
                "value_pct",
            ]
        ]
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 8. Validation
    # ------------------------------------------------------------------
    print("\n8. VALIDATION")
    print("-" * 80)

    print(
        "Customers covered:",
        f"{len(rfm):,}"
    )

    print(
        "Unassigned customers:",
        rfm["baseline_segment"].isna().sum()
    )

    print(
        "Segments:",
        rfm["baseline_segment"].nunique()
    )

    print("\n" + "=" * 80)
    print("RFM BASELINE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()