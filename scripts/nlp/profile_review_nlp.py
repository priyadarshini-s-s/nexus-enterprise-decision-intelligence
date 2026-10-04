from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/gold/olist/review_nlp_dataset.parquet")
OUTPUT_PATH = Path("data/gold/olist/review_nlp_profile.parquet")


def main():
    print("\nNEXUS — Profile Review NLP Dataset")
    print("=" * 70)

    df = pd.read_parquet(INPUT_PATH)

    print("\nDATASET")
    print("-" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique review_id: {df['review_id'].nunique():,}")
    print(
        f"Duplicate review_id observations: "
        f"{df['review_id'].duplicated().sum():,}"
    )

    # --------------------------------------------------------
    # TEXT COVERAGE
    # --------------------------------------------------------

    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()

    df["has_review_text"] = df["review_text"].ne("")

    df["text_length_chars"] = df["review_text"].str.len()

    df["text_word_count"] = (
        df["review_text"]
        .str.split()
        .str.len()
    )

    print("\nTEXT COVERAGE")
    print("-" * 70)

    text_count = int(df["has_review_text"].sum())
    no_text_count = len(df) - text_count

    print(f"Reviews with text: {text_count:,}")
    print(f"Reviews without text: {no_text_count:,}")
    print(f"Text coverage: {df['has_review_text'].mean() * 100:.2f}%")

    print(
        f"Median text length (characters): "
        f"{df.loc[df['has_review_text'], 'text_length_chars'].median():.2f}"
    )

    print(
        f"Mean text length (characters): "
        f"{df.loc[df['has_review_text'], 'text_length_chars'].mean():.2f}"
    )

    print(
        f"Median word count: "
        f"{df.loc[df['has_review_text'], 'text_word_count'].median():.2f}"
    )

    # --------------------------------------------------------
    # REVIEW SCORE
    # --------------------------------------------------------

    print("\nREVIEW SCORE DISTRIBUTION")
    print("-" * 70)

    score_distribution = (
        df["review_score"]
        .value_counts()
        .sort_index()
    )

    for score, count in score_distribution.items():
        pct = count / len(df) * 100
        print(f"Score {score}: {count:>7,} ({pct:>6.2f}%)")

    # --------------------------------------------------------
    # TEXT AVAILABILITY BY SCORE
    # --------------------------------------------------------

    print("\nTEXT AVAILABILITY BY SCORE")
    print("-" * 70)

    score_text = (
        df.groupby("review_score")
        .agg(
            reviews=("review_observation_id", "count"),
            with_text=("has_review_text", "sum"),
        )
    )

    score_text["text_coverage_pct"] = (
        score_text["with_text"]
        / score_text["reviews"]
        * 100
    )

    print(score_text.to_string())

    # --------------------------------------------------------
    # TEXT LENGTH BY SCORE
    # --------------------------------------------------------

    print("\nTEXT LENGTH BY SCORE")
    print("-" * 70)

    length_by_score = (
        df[df["has_review_text"]]
        .groupby("review_score")["text_length_chars"]
        .agg(["count", "mean", "median", "min", "max"])
    )

    print(length_by_score.to_string())

    # --------------------------------------------------------
    # DUPLICATE TEXT
    # --------------------------------------------------------

    print("\nDUPLICATE TEXT ANALYSIS")
    print("-" * 70)

    text_df = df[df["has_review_text"]].copy()

    duplicate_text_rows = text_df["review_text"].duplicated().sum()
    unique_texts = text_df["review_text"].nunique()

    print(f"Text-bearing observations: {len(text_df):,}")
    print(f"Unique text strings: {unique_texts:,}")
    print(f"Repeated text observations: {duplicate_text_rows:,}")

    print("\nMost repeated review texts:")

    repeated_texts = (
        text_df["review_text"]
        .value_counts()
        .head(15)
    )

    for text, count in repeated_texts.items():
        preview = text.replace("\n", " ")[:120]
        print(f"{count:>5} × {preview}")

    # --------------------------------------------------------
    # TEXT SCORE RELATIONSHIP
    # --------------------------------------------------------

    print("\nTEXT VS REVIEW SCORE")
    print("-" * 70)

    text_score_summary = (
        df.groupby("has_review_text")["review_score"]
        .agg(["count", "mean", "median"])
    )

    print(text_score_summary.to_string())

    # --------------------------------------------------------
    # TIME COVERAGE
    # --------------------------------------------------------

    print("\nTIME COVERAGE")
    print("-" * 70)

    df["review_creation_date"] = pd.to_datetime(
        df["review_creation_date"],
        errors="coerce",
    )

    print(
        f"First review: "
        f"{df['review_creation_date'].min()}"
    )

    print(
        f"Last review: "
        f"{df['review_creation_date'].max()}"
    )

    # Monthly text coverage
    df["review_month"] = (
        df["review_creation_date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        df.groupby("review_month")
        .agg(
            reviews=("review_observation_id", "count"),
            text_reviews=("has_review_text", "sum"),
            avg_score=("review_score", "mean"),
        )
    )

    monthly["text_coverage_pct"] = (
        monthly["text_reviews"]
        / monthly["reviews"]
        * 100
    )

    print("\nMonthly review coverage:")
    print(monthly.to_string())

    # --------------------------------------------------------
    # PRODUCT / CATEGORY COVERAGE
    # --------------------------------------------------------

    print("\nPRODUCT / CATEGORY CONTEXT")
    print("-" * 70)

    if "product_count" in df.columns:
        print(
            f"Reviews with product context: "
            f"{df['product_count'].notna().sum():,}"
        )

    if "product_categories" in df.columns:
        category_available = (
            df["product_categories"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

        print(
            f"Reviews with category context: "
            f"{category_available.sum():,}"
        )

    # --------------------------------------------------------
    # SAVE PROFILE ARTIFACT
    # --------------------------------------------------------

    profile = pd.DataFrame(
        {
            "metric": [
                "rows",
                "unique_review_id",
                "duplicate_review_id_observations",
                "reviews_with_text",
                "reviews_without_text",
                "text_coverage_pct",
                "median_text_length_chars",
                "mean_text_length_chars",
                "median_word_count",
                "mean_review_score",
                "unique_text_strings",
                "repeated_text_observations",
            ],
            "value": [
                len(df),
                df["review_id"].nunique(),
                df["review_id"].duplicated().sum(),
                text_count,
                no_text_count,
                df["has_review_text"].mean() * 100,
                df.loc[
                    df["has_review_text"],
                    "text_length_chars"
                ].median(),
                df.loc[
                    df["has_review_text"],
                    "text_length_chars"
                ].mean(),
                df.loc[
                    df["has_review_text"],
                    "text_word_count"
                ].median(),
                df["review_score"].mean(),
                unique_texts,
                duplicate_text_rows,
            ],
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nOUTPUT")
    print("-" * 70)
    print(f"Saved: {OUTPUT_PATH}")

    print("\nNLP profiling complete.\n")


if __name__ == "__main__":
    main()