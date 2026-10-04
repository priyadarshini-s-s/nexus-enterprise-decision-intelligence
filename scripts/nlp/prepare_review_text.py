from pathlib import Path
import re
import unicodedata

import pandas as pd


INPUT_PATH = Path(
    "data/gold/olist/review_nlp_dataset.parquet"
)

OUTPUT_PATH = Path(
    "data/gold/olist/review_nlp_prepared.parquet"
)


def normalize_text(text: str) -> str:
    """
    Normalize review text while preserving semantic information.

    We intentionally DO NOT remove negation words such as:
    - não
    - nao
    - nunca
    - sem

    because removing them can reverse sentiment meaning.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization.
    # Keeps accented Portuguese characters intact.
    text = unicodedata.normalize("NFKC", text)

    # Lowercase for classical NLP models such as TF-IDF.
    text = text.lower()

    # Normalize common whitespace characters.
    text = re.sub(r"\s+", " ", text)

    # Collapse repeated punctuation.
    # Example:
    # "ótimo!!!" -> "ótimo!"
    # "???!!!" -> "?!"
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{3,}", "...", text)

    # Remove surrounding whitespace.
    text = text.strip()

    return text


def main():
    print("\nNEXUS — Prepare Review Text")
    print("=" * 70)

    df = pd.read_parquet(INPUT_PATH)

    print("\nINPUT")
    print("-" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Preserve original text
    # --------------------------------------------------------

    if "review_text" not in df.columns:
        raise ValueError(
            "Expected column 'review_text' was not found."
        )

    original_text = (
        df["review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Create normalized text
    # --------------------------------------------------------

    df["review_text"] = original_text

    df["review_text_normalized"] = (
        df["review_text"]
        .apply(normalize_text)
    )

    # --------------------------------------------------------
    # Validate row grain
    # --------------------------------------------------------

    if df["review_observation_id"].duplicated().any():
        raise ValueError(
            "review_observation_id grain was violated."
        )

    # --------------------------------------------------------
    # Text coverage
    # --------------------------------------------------------

    df["has_normalized_text"] = (
        df["review_text_normalized"].ne("")
    )

    print("\nTEXT COVERAGE")
    print("-" * 70)

    print(
        f"Original text rows: "
        f"{df['review_text'].ne('').sum():,}"
    )

    print(
        f"Normalized text rows: "
        f"{df['has_normalized_text'].sum():,}"
    )

    # --------------------------------------------------------
    # Compare original vs normalized
    # --------------------------------------------------------

    changed = (
        df["review_text"]
        != df["review_text_normalized"]
    )

    print(
        f"Rows changed by normalization: "
        f"{changed.sum():,}"
    )

    # --------------------------------------------------------
    # Negation preservation check
    # --------------------------------------------------------

    negation_words = [
        "não",
        "nao",
        "nunca",
        "sem",
    ]

    print("\nNEGATION PRESERVATION")
    print("-" * 70)

    for word in negation_words:
        original_count = (
            df["review_text"]
            .str.lower()
            .str.contains(
                rf"\b{word}\b",
                regex=True,
                na=False,
            )
            .sum()
        )

        normalized_count = (
            df["review_text_normalized"]
            .str.contains(
                rf"\b{word}\b",
                regex=True,
                na=False,
            )
            .sum()
        )

        print(
            f"{word:>6}: "
            f"original={original_count:,}, "
            f"normalized={normalized_count:,}"
        )

        if original_count != normalized_count:
            raise ValueError(
                f"Negation preservation failed for '{word}'."
            )

    # --------------------------------------------------------
    # Sample transformations
    # --------------------------------------------------------

    print("\nSAMPLE TRANSFORMATIONS")
    print("-" * 70)

    samples = df.loc[
        df["has_normalized_text"],
        [
            "review_text",
            "review_text_normalized",
        ],
    ].head(20)

    for _, row in samples.iterrows():
        print(f"ORIGINAL : {row['review_text']}")
        print(f"NORMALIZED: {row['review_text_normalized']}")
        print()

    # --------------------------------------------------------
    # Preserve important columns
    # --------------------------------------------------------

    preferred_columns = [
        "review_observation_id",
        "review_id",
        "order_id",
        "customer_unique_id",
        "review_score",
        "review_text",
        "review_text_normalized",
        "has_normalized_text",
        "review_creation_date",
        "review_answer_timestamp",
        "order_purchase_timestamp",
        "product_count",
        "seller_count",
        "product_categories",
    ]

    final_columns = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    # Keep any additional source columns as well.
    remaining_columns = [
        col
        for col in df.columns
        if col not in final_columns
    ]

    df = df[
        final_columns + remaining_columns
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nOUTPUT")
    print("-" * 70)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nReview text preparation complete.\n")


if __name__ == "__main__":
    main()