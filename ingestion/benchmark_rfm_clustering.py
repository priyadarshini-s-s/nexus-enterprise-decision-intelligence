from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


GOLD_DIR = Path("data/gold/olist")


def evaluate_feature_space(
    df: pd.DataFrame,
    features: list[str],
    k_values: list[int],
):
    print("\n" + "=" * 80)
    print(f"FEATURE SPACE: {', '.join(features)}")
    print("=" * 80)

    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    for k in k_values:
        print(f"\nTesting K={k}...")

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_

        silhouette = silhouette_score(
            X_scaled,
            labels,
            sample_size=20000,
            random_state=42,
        )

        cluster_sizes = pd.Series(labels).value_counts()

        smallest_cluster_pct = (
            cluster_sizes.min()
            / len(labels)
            * 100
        )

        largest_cluster_pct = (
            cluster_sizes.max()
            / len(labels)
            * 100
        )

        results.append(
            {
                "feature_space": "+".join(features),
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette,
                "smallest_cluster_pct": smallest_cluster_pct,
                "largest_cluster_pct": largest_cluster_pct,
            }
        )

        print(
            f"Silhouette: {silhouette:.4f} | "
            f"Inertia: {inertia:.2f} | "
            f"Smallest cluster: {smallest_cluster_pct:.2f}% | "
            f"Largest cluster: {largest_cluster_pct:.2f}%"
        )

    return results


def main():
    print("=" * 80)
    print("NEXUS — RFM Clustering Benchmark")
    print("=" * 80)

    df = pd.read_parquet(
        GOLD_DIR / "rfm_clustering_dataset.parquet"
    )

    print(
        f"\nCustomers available for clustering: "
        f"{len(df):,}"
    )

    k_values = [2, 3, 4, 5, 6, 7, 8]

    all_results = []

    # --------------------------------------------------------------
    # Feature space A — raw monetary
    # --------------------------------------------------------------

    all_results.extend(
        evaluate_feature_space(
            df,
            [
                "recency_days",
                "frequency",
                "monetary",
            ],
            k_values,
        )
    )

    # --------------------------------------------------------------
    # Feature space B — log monetary
    # --------------------------------------------------------------

    all_results.extend(
        evaluate_feature_space(
            df,
            [
                "recency_days",
                "frequency",
                "log_monetary",
            ],
            k_values,
        )
    )

    results = pd.DataFrame(all_results)

    output_path = (
        GOLD_DIR /
        "rfm_clustering_benchmark.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(
        results
        .sort_values(
            ["feature_space", "silhouette"],
            ascending=[True, False],
        )
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("RFM CLUSTERING BENCHMARK COMPLETE")
    print("=" * 80)

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()