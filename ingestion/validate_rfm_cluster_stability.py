from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


GOLD_DIR = Path("data/gold/olist")


def main():
    print("=" * 80)
    print("NEXUS — RFM K-Means Stability Validation")
    print("=" * 80)

    df = pd.read_parquet(
        GOLD_DIR / "rfm_clustering_dataset.parquet"
    )

    features = [
        "recency_days",
        "frequency",
        "monetary",
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    seeds = [
        7,
        21,
        42,
        100,
        2026,
    ]

    labels = {}

    for seed in seeds:
        model = KMeans(
            n_clusters=4,
            random_state=seed,
            n_init=20,
        )

        labels[seed] = model.fit_predict(X_scaled)

    print("\nPairwise Adjusted Rand Index")
    print("-" * 80)

    rows = []

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            ari = adjusted_rand_score(
                labels[seed_a],
                labels[seed_b],
            )

            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "ARI": ari,
                }
            )

            print(
                f"{seed_a:>5} vs {seed_b:<5} → "
                f"ARI = {ari:.4f}"
            )

    results = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("STABILITY SUMMARY")
    print("=" * 80)

    print(
        f"Mean ARI: {results['ARI'].mean():.4f}"
    )

    print(
        f"Minimum ARI: {results['ARI'].min():.4f}"
    )

    print(
        f"Maximum ARI: {results['ARI'].max():.4f}"
    )

    output_path = (
        GOLD_DIR /
        "rfm_cluster_stability.parquet"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()