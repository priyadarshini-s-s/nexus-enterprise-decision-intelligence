from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_nlp_prepared.parquet"
)

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_theme_assignments.parquet"
)

EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_transformer_embeddings.npz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
    / "review_theme_representatives.parquet"
)

TOP_N = 15


# ============================================================
# LOAD
# ============================================================

reviews = pd.read_parquet(ASSIGNMENT_PATH)

embedding_data = np.load(
    EMBEDDING_PATH,
    allow_pickle=False,
)

embeddings = embedding_data["embeddings"]
row_index = embedding_data["row_index"]


# ============================================================
# ALIGN
# ============================================================

position_map = {
    idx: position
    for position, idx in enumerate(row_index)
}

valid = [
    idx
    for idx in reviews.index
    if idx in position_map
]

reviews = reviews.loc[valid].copy()

positions = [
    position_map[idx]
    for idx in reviews.index
]

X = embeddings[positions]


# ============================================================
# REPRESENTATIVE REVIEWS
# ============================================================

records = []

for cluster_id in sorted(
    reviews["semantic_theme_cluster"].unique()
):

    mask = (
        reviews["semantic_theme_cluster"].to_numpy()
        == cluster_id
    )

    cluster_positions = np.where(mask)[0]

    cluster_embeddings = X[cluster_positions]

    centroid = cluster_embeddings.mean(axis=0)

    # Cosine distance to centroid.
    distances = pairwise_distances(
        cluster_embeddings,
        centroid.reshape(1, -1),
        metric="cosine",
    ).ravel()

    nearest = np.argsort(distances)[:TOP_N]

    print("\n")
    print("=" * 80)
    print(f"CLUSTER {cluster_id}")
    print("=" * 80)

    for rank, local_position in enumerate(nearest, start=1):

        global_position = cluster_positions[
            local_position
        ]

        row = reviews.iloc[global_position]

        text = row["review_text_normalized"]

        print(
            f"\n[{rank}] "
            f"score={row['review_score']} | "
            f"distance={distances[local_position]:.4f}"
        )

        print(text)

        records.append(
            {
                "cluster": int(cluster_id),
                "rank": int(rank),
                "distance_to_centroid": float(
                    distances[local_position]
                ),
                "review_score": int(
                    row["review_score"]
                ),
                "review_text": text,
                "review_id": row.get(
                    "review_id",
                    None,
                ),
                "order_id": row.get(
                    "order_id",
                    None,
                ),
            }
        )


# ============================================================
# SAVE
# ============================================================

output = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\n")
print("=" * 80)
print("REPRESENTATIVE REVIEW EXTRACTION COMPLETE")
print("=" * 80)

print(f"Saved: {OUTPUT_PATH}")