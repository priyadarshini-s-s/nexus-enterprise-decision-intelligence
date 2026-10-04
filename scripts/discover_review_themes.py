from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_nlp_prepared.parquet"
)

EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_transformer_embeddings.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
)

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_theme_assignments.parquet"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "review_theme_clustering_results.json"
)

RANDOM_STATE = 42

CLUSTER_COUNTS = [5, 8, 10, 12, 15]

BATCH_SIZE = 1024

# Use a representative sample for silhouette evaluation.
SILHOUETTE_SAMPLE_SIZE = 10000


# ============================================================
# LOAD REVIEWS
# ============================================================

print("=" * 70)
print("NEXUS — Olist Voice-of-Customer Semantic Theme Discovery")
print("=" * 70)

reviews = pd.read_parquet(REVIEW_PATH)

reviews = reviews[
    reviews["review_text_normalized"].notna()
    & (reviews["review_text_normalized"].str.strip() != "")
].copy()

print(f"\nText-bearing reviews: {len(reviews):,}")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embedding_data = np.load(
    EMBEDDING_PATH,
    allow_pickle=False,
)

embeddings = embedding_data["embeddings"]
row_index = embedding_data["row_index"]

print(f"Embedding matrix: {embeddings.shape}")


# ============================================================
# ALIGN REVIEWS WITH EMBEDDINGS
# ============================================================

review_positions = {
    idx: pos
    for pos, idx in enumerate(row_index)
}

valid_indices = [
    idx for idx in reviews.index
    if idx in review_positions
]

reviews = reviews.loc[valid_indices].copy()

positions = [
    review_positions[idx]
    for idx in reviews.index
]

X = embeddings[positions]

print(f"Aligned reviews: {len(reviews):,}")
print(f"Aligned embeddings: {X.shape}")


# ============================================================
# CLUSTERING EXPERIMENT
# ============================================================

results = []

best_model = None
best_k = None
best_score = -1

print("\nTesting cluster counts...")
print("=" * 70)

for k in CLUSTER_COUNTS:

    print(f"\nK = {k}")

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        n_init=10,
        max_iter=100,
    )

    labels = model.fit_predict(X)

    sample_size = min(
        SILHOUETTE_SAMPLE_SIZE,
        len(X),
    )

    rng = np.random.default_rng(RANDOM_STATE)

    sample_indices = rng.choice(
        len(X),
        size=sample_size,
        replace=False,
    )

    silhouette = silhouette_score(
        X[sample_indices],
        labels[sample_indices],
    )

    counts = np.bincount(labels)

    largest_cluster = int(counts.max())
    smallest_cluster = int(counts.min())

    print(
        f"Silhouette: {silhouette:.4f} | "
        f"smallest cluster: {smallest_cluster:,} | "
        f"largest cluster: {largest_cluster:,}"
    )

    results.append(
        {
            "k": k,
            "silhouette": float(silhouette),
            "smallest_cluster": smallest_cluster,
            "largest_cluster": largest_cluster,
            "cluster_sizes": counts.tolist(),
        }
    )

    if silhouette > best_score:
        best_score = silhouette
        best_k = k
        best_model = model


# ============================================================
# FINAL CLUSTER ASSIGNMENTS
# ============================================================

print("\n" + "=" * 70)
print(f"Selected K: {best_k}")
print(f"Best silhouette: {best_score:.4f}")
print("=" * 70)

labels = best_model.labels_

reviews["semantic_theme_cluster"] = labels


# ============================================================
# CLUSTER PROFILE
# ============================================================

profiles = []

for cluster_id in sorted(
    reviews["semantic_theme_cluster"].unique()
):

    cluster_reviews = reviews[
        reviews["semantic_theme_cluster"] == cluster_id
    ]

    scores = cluster_reviews["review_score"]

    profile = {
        "cluster": int(cluster_id),
        "review_count": int(len(cluster_reviews)),
        "share": float(
            len(cluster_reviews) / len(reviews)
        ),
        "mean_review_score": float(scores.mean()),
        "negative_rate": float(
            (scores <= 2).mean()
        ),
        "neutral_rate": float(
            (scores == 3).mean()
        ),
        "positive_rate": float(
            (scores >= 4).mean()
        ),
    }

    profiles.append(profile)


profiles_df = pd.DataFrame(profiles)

profiles_df = profiles_df.sort_values(
    "review_count",
    ascending=False,
)


# ============================================================
# SAVE ASSIGNMENTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSIGNMENT_COLUMNS = [
    "review_observation_id",
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "review_text_normalized",
    "semantic_theme_cluster",
]

available_columns = [
    c for c in ASSIGNMENT_COLUMNS
    if c in reviews.columns
]

reviews[
    available_columns
].to_parquet(
    ASSIGNMENT_PATH,
    index=False,
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    "embedding_model": (
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "embedding_dimension": int(X.shape[1]),
    "reviews": int(len(reviews)),
    "tested_cluster_counts": CLUSTER_COUNTS,
    "selected_k": int(best_k),
    "best_silhouette": float(best_score),
    "cluster_profiles": profiles,
    "experiments": results,
}

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


# ============================================================
# PRINT PROFILE
# ============================================================

print("\nCluster profiles")
print("=" * 70)

print(
    profiles_df.to_string(
        index=False,
        formatters={
            "share": "{:.2%}".format,
            "mean_review_score": "{:.3f}".format,
            "negative_rate": "{:.2%}".format,
            "neutral_rate": "{:.2%}".format,
            "positive_rate": "{:.2%}".format,
        },
    )
)

print("\nSaved assignments:")
print(ASSIGNMENT_PATH)

print("\nSaved clustering results:")
print(RESULTS_PATH)

print("\n" + "=" * 70)
print("THEME DISCOVERY COMPLETE")
print("=" * 70)