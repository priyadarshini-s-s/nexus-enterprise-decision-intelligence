from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


GOLD_DIR = Path("data/gold/olist")


def profile_k(df, k):
    features = [
        "recency_days",
        "frequency",
        "monetary",
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    df = df.copy()
    df["cluster"] = model.fit_predict(X_scaled)

    total_monetary = df["monetary"].sum()

    summary = (
        df.groupby("cluster")
        .agg(
            customers=("customer_unique_id", "count"),
            median_recency=("recency_days", "median"),
            mean_recency=("recency_days", "mean"),
            median_frequency=("frequency", "median"),
            mean_frequency=("frequency", "mean"),
            median_monetary=("monetary", "median"),
            mean_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
            repeat_customers=("is_repeat_customer", "sum"),
        )
        .sort_values("median_monetary")
    )

    summary["customer_pct"] = (
        summary["customers"]
        / len(df)
        * 100
    )

    summary["value_pct"] = (
        summary["total_monetary"]
        / total_monetary
        * 100
    )

    summary["repeat_rate_pct"] = (
        summary["repeat_customers"]
        / summary["customers"]
        * 100
    )

    print("\n" + "=" * 80)
    print(f"K = {k}")
    print("=" * 80)

    print(
        summary[
            [
                "customers",
                "customer_pct",
                "median_recency",
                "mean_recency",
                "median_frequency",
                "mean_frequency",
                "median_monetary",
                "mean_monetary",
                "total_monetary",
                "value_pct",
                "repeat_rate_pct",
            ]
        ]
        .round(2)
        .to_string()
    )

    return df, summary


def main():
    print("=" * 80)
    print("NEXUS — RFM Cluster Profiling")
    print("=" * 80)

    df = pd.read_parquet(
        GOLD_DIR / "rfm_clustering_dataset.parquet"
    )

    all_summaries = []

    for k in [3, 4, 5]:
        clustered, summary = profile_k(df, k)

        summary = summary.reset_index()
        summary["k"] = k

        all_summaries.append(summary)

    combined = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    output_path = (
        GOLD_DIR /
        "rfm_cluster_profiles.parquet"
    )

    combined.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CLUSTER PROFILING COMPLETE")
    print("=" * 80)

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()