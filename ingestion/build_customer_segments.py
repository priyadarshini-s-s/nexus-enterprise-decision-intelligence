from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


GOLD_DIR = Path("data/gold/olist")


def main():
    print("=" * 80)
    print("NEXUS — Production Customer Segmentation")
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

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20,
    )

    df["cluster_id"] = model.fit_predict(X_scaled)

    # ------------------------------------------------------------------
    # Determine semantic names from measured cluster profiles
    # ------------------------------------------------------------------

    profiles = (
        df.groupby("cluster_id")
        .agg(
            median_recency=("recency_days", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary", "median"),
            repeat_rate=("is_repeat_customer", "mean"),
        )
    )

    # Repeat cluster:
    repeat_cluster = profiles["median_frequency"].idxmax()

    # High-value cluster among non-repeat clusters:
    non_repeat = profiles[
        profiles.index != repeat_cluster
    ]

    high_value_cluster = non_repeat[
        "median_monetary"
    ].idxmax()

    remaining = [
        cluster
        for cluster in profiles.index
        if cluster not in {
            repeat_cluster,
            high_value_cluster,
        }
    ]

    # Of remaining clusters, lower recency = more recent.
    recent_cluster = profiles.loc[
        remaining,
        "median_recency",
    ].idxmin()

    older_cluster = profiles.loc[
        remaining,
        "median_recency",
    ].idxmax()

    segment_names = {
        recent_cluster: "Recent One-Time",
        older_cluster: "Older One-Time",
        repeat_cluster: "Repeat Customers",
        high_value_cluster: "High-Value One-Time",
    }

    df["segment_name"] = df["cluster_id"].map(
        segment_names
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    df["segmentation_model"] = "KMeans"
    df["segmentation_version"] = "v1.0"
    df["segmentation_features"] = (
        "recency_days,frequency,monetary"
    )
    df["scaling_method"] = "StandardScaler"
    df["n_clusters"] = 4
    df["random_state"] = 42

    output_columns = [
        "customer_unique_id",
        "recency_days",
        "frequency",
        "monetary",
        "cluster_id",
        "segment_name",
        "segmentation_model",
        "segmentation_version",
        "segmentation_features",
        "scaling_method",
        "n_clusters",
        "random_state",
    ]

    segments = df[output_columns].copy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if segments["customer_unique_id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    if segments["segment_name"].isna().any():
        raise ValueError(
            "Some customers have no segment name."
        )

    expected_segments = {
        "Recent One-Time",
        "Older One-Time",
        "Repeat Customers",
        "High-Value One-Time",
    }

    actual_segments = set(
        segments["segment_name"].unique()
    )

    if actual_segments != expected_segments:
        raise ValueError(
            f"Unexpected segment set: {actual_segments}"
        )

    output_path = (
        GOLD_DIR /
        "customer_segments.parquet"
    )

    segments.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CUSTOMER SEGMENTATION VALIDATION: PASS")
    print("=" * 80)

    print(f"Customers: {len(segments):,}")
    print(
        f"Segments: {segments['segment_name'].nunique()}"
    )

    print("\nSegment distribution:")
    print(
        segments["segment_name"]
        .value_counts()
        .to_string()
    )

    print("\nOutput:")
    print(output_path)


if __name__ == "__main__":
    main()