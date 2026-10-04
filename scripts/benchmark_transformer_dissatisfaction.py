from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_nlp_prepared.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
)

EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_transformer_embeddings.npz"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "transformer_binary_dissatisfaction_results.json"
)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TEXT_COLUMN = "review_text_normalized"
DATE_COLUMN = "review_creation_date"

BATCH_SIZE = 32
RANDOM_STATE = 42


# ============================================================
# TEMPORAL SPLIT
# ============================================================

TRAIN_END = pd.Timestamp("2018-04-30")
VALIDATION_END = pd.Timestamp("2018-06-30")

# Test = 2018-07-01 -> 2018-08-31


# ============================================================
# HELPERS
# ============================================================

def create_target(review_score):
    """
    Rating-derived dissatisfaction proxy.

    1-2 -> dissatisfaction
    3-5 -> not dissatisfaction

    IMPORTANT:
    This is an observational proxy derived from the review score,
    not independently human-labeled sentiment ground truth.
    """
    return (review_score <= 2).astype(int)


def print_split(name, df):
    print(
        f"{name}: "
        f"{len(df):,} rows | "
        f"dissatisfied={df['target'].sum():,} "
        f"({df['target'].mean():.2%})"
    )


def evaluate_binary(model, X, y, name):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y, probabilities)

    metrics = {
        "split": name,
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "negative_f1": float(f1_score(y, predictions, pos_label=1)),
        "negative_roc_auc": float(roc_auc_score(y, probabilities)),
        "negative_pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "negative_precision": float(
            ((predictions == 1) & (y == 1)).sum()
            / max((predictions == 1).sum(), 1)
        ),
        "negative_recall": float(
            ((predictions == 1) & (y == 1)).sum()
            / max((y == 1).sum(), 1)
        ),
    }

    print(f"\n{name}")
    print("-" * 60)

    for key, value in metrics.items():
        if key not in {"split", "rows"}:
            print(f"{key}: {value:.6f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["not_dissatisfied", "dissatisfied"],
            digits=4,
        )
    )

    return metrics


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NEXUS — Transformer Dissatisfaction Benchmark")
print("=" * 70)

print(f"\nLoading: {INPUT_PATH}")

df = pd.read_parquet(INPUT_PATH)

print(f"Loaded {len(df):,} rows")

required_columns = {
    TEXT_COLUMN,
    DATE_COLUMN,
    "review_score",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# TEXT FILTER
# ============================================================

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

df = df[
    df[TEXT_COLUMN].notna()
    & (df[TEXT_COLUMN].str.strip() != "")
    & df[DATE_COLUMN].notna()
].copy()

df["target"] = create_target(df["review_score"])

print(f"\nText-bearing reviews: {len(df):,}")

print(
    f"Dissatisfied: {df['target'].sum():,} "
    f"({df['target'].mean():.2%})"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train = df[df[DATE_COLUMN] <= TRAIN_END].copy()

validation = df[
    (df[DATE_COLUMN] > TRAIN_END)
    & (df[DATE_COLUMN] <= VALIDATION_END)
].copy()

test = df[df[DATE_COLUMN] > VALIDATION_END].copy()

print("\nTemporal split")
print("=" * 70)

print_split("TRAIN", train)
print_split("VALIDATION", validation)
print_split("TEST", test)


# ============================================================
# LOAD TRANSFORMER
# ============================================================

print("\nLoading Transformer model...")
print(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

print(
    f"Embedding dimension: "
    f"{model.get_embedding_dimension()}"
)


# ============================================================
# EMBEDDING GENERATION
# ============================================================

texts = df[TEXT_COLUMN].tolist()

print("\nGenerating embeddings...")
print(f"Reviews: {len(texts):,}")
print(f"Batch size: {BATCH_SIZE}")

start_time = time.time()

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

elapsed = time.time() - start_time

print(
    f"\nEmbedding generation completed in "
    f"{elapsed / 60:.2f} minutes"
)

print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

EMBEDDING_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez_compressed(
    EMBEDDING_PATH,
    embeddings=embeddings,
    row_index=df.index.to_numpy(),
)

print(f"\nSaved embeddings:")
print(EMBEDDING_PATH)


# ============================================================
# MAP EMBEDDINGS TO SPLITS
# ============================================================

train_idx = train.index
validation_idx = validation.index
test_idx = test.index

index_to_position = {
    idx: position
    for position, idx in enumerate(df.index)
}


def get_embeddings(split_df):
    positions = [
        index_to_position[idx]
        for idx in split_df.index
    ]

    return embeddings[positions]


X_train = get_embeddings(train)
X_validation = get_embeddings(validation)
X_test = get_embeddings(test)

y_train = train["target"].to_numpy()
y_validation = validation["target"].to_numpy()
y_test = test["target"].to_numpy()


# ============================================================
# SCALE
# ============================================================

print("\nScaling embeddings...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

classifier.fit(
    X_train_scaled,
    y_train,
)


# ============================================================
# EVALUATION
# ============================================================

validation_metrics = evaluate_binary(
    classifier,
    X_validation_scaled,
    y_validation,
    "VALIDATION",
)

test_metrics = evaluate_binary(
    classifier,
    X_test_scaled,
    y_test,
    "TEST",
)


# ============================================================
# RESULTS
# ============================================================

results = {
    "model": MODEL_NAME,
    "representation": {
        "type": "sentence_embedding",
        "dimension": int(
            model.get_embedding_dimension()
        ),
        "normalized": True,
    },
    "classifier": {
        "type": "LogisticRegression",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "target": {
        "definition": "review_score <= 2",
        "description": (
            "Rating-derived dissatisfaction proxy; "
            "not independently human-labeled sentiment ground truth."
        ),
    },
    "temporal_split": {
        "train_end": str(TRAIN_END.date()),
        "validation_end": str(VALIDATION_END.date()),
        "test_start": "2018-07-01",
    },
    "embedding_generation_minutes": elapsed / 60,
    "validation": validation_metrics,
    "test": test_metrics,
}


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )

print("\nResults saved:")
print(RESULTS_PATH)

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)