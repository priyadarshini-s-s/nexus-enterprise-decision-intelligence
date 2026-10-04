from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/gold/olist/review_nlp_dataset.parquet")
OUTPUT_PATH = Path("data/gold/olist/review_language_profile.parquet")


def main():
    print("\nNEXUS — Review Language Profile")
    print("=" * 70)

    df = pd.read_parquet(INPUT_PATH)

    df["review_text"] = (
        df["review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    text_df = df[df["review_text"].ne("")].copy()

    print("\nTEXT CORPUS")
    print("-" * 70)
    print(f"All review observations: {len(df):,}")
    print(f"Text-bearing observations: {len(text_df):,}")

    # --------------------------------------------------------
    # Character / Unicode diagnostics
    # --------------------------------------------------------

    def count_chars(text, predicate):
        return sum(predicate(ch) for ch in text)

    def has_accented_latin(text):
        return any(
            ord(ch) > 127 and (
                "LATIN" in __import__("unicodedata")
                .name(ch, "")
            )
            for ch in text
        )

    text_df["char_count"] = text_df["review_text"].str.len()

    text_df["accented_latin"] = text_df["review_text"].apply(
        has_accented_latin
    )

    text_df["question_mark"] = text_df["review_text"].str.contains(
        r"\?",
        regex=True,
        na=False,
    )

    text_df["exclamation_mark"] = text_df["review_text"].str.contains(
        r"!",
        regex=True,
        na=False,
    )

    print("\nTEXT CHARACTERISTICS")
    print("-" * 70)

    print(
        f"Texts containing accented Latin characters: "
        f"{text_df['accented_latin'].sum():,} "
        f"({text_df['accented_latin'].mean() * 100:.2f}%)"
    )

    print(
        f"Texts containing '?': "
        f"{text_df['question_mark'].sum():,}"
    )

    print(
        f"Texts containing '!': "
        f"{text_df['exclamation_mark'].sum():,}"
    )

    # --------------------------------------------------------
    # Common Portuguese indicators
    # --------------------------------------------------------

    # These are NOT treated as proof of language.
    # They are diagnostic lexical indicators only.

    portuguese_markers = [
        "não",
        "nao",
        "muito",
        "bom",
        "boa",
        "ótimo",
        "otimo",
        "ótima",
        "otima",
        "produto",
        "entrega",
        "chegou",
        "prazo",
        "recomendo",
        "obrigado",
        "obrigada",
        "ruim",
        "problema",
        "perfeito",
        "perfeita",
        "qualidade",
        "gostei",
        "gosta",
        "compra",
        "comprar",
        "vendedor",
        "pedido",
    ]

    lowered = text_df["review_text"].str.lower()

    marker_rows = []

    for marker in portuguese_markers:
        mask = lowered.str.contains(
            rf"\b{marker}\b",
            regex=True,
            na=False,
        )

        count = int(mask.sum())

        marker_rows.append(
            {
                "marker": marker,
                "count": count,
                "coverage_pct": (
                    count / len(text_df) * 100
                    if len(text_df)
                    else 0
                ),
            }
        )

    markers_df = (
        pd.DataFrame(marker_rows)
        .sort_values(
            "count",
            ascending=False,
        )
    )

    print("\nPORTUGUESE LEXICAL INDICATORS")
    print("-" * 70)

    print(markers_df.to_string(index=False))

    # --------------------------------------------------------
    # Sample inspection
    # --------------------------------------------------------

    print("\nSAMPLE TEXTS")
    print("-" * 70)

    sample = text_df[
        [
            "review_text",
            "review_score",
        ]
    ].head(30)

    for _, row in sample.iterrows():
        print(
            f"[{row['review_score']}★] "
            f"{row['review_text'][:180]}"
        )

    # --------------------------------------------------------
    # Simple language evidence score
    # --------------------------------------------------------

    marker_pattern = (
        r"\b("
        + "|".join(
            marker.replace("?", r"\?")
            for marker in portuguese_markers
        )
        + r")\b"
    )

    text_df["portuguese_marker_count"] = (
        lowered.str.count(marker_pattern)
    )

    marker_coverage = (
        text_df["portuguese_marker_count"] > 0
    ).mean() * 100

    print("\nLEXICAL LANGUAGE EVIDENCE")
    print("-" * 70)

    print(
        "Texts containing at least one Portuguese indicator: "
        f"{(text_df['portuguese_marker_count'] > 0).sum():,} "
        f"({marker_coverage:.2f}%)"
    )

    print(
        "\nIMPORTANT:"
        "\nThese lexical indicators are diagnostic evidence, "
        "not a formal language classifier."
    )

    # --------------------------------------------------------
    # Save profile
    # --------------------------------------------------------

    profile = pd.DataFrame(
        {
            "metric": [
                "all_review_observations",
                "text_bearing_observations",
                "accented_latin_texts",
                "accented_latin_pct",
                "texts_with_portuguese_indicator",
                "portuguese_indicator_pct",
                "texts_with_question_mark",
                "texts_with_exclamation_mark",
            ],
            "value": [
                len(df),
                len(text_df),
                int(text_df["accented_latin"].sum()),
                text_df["accented_latin"].mean() * 100,
                int(
                    (
                        text_df["portuguese_marker_count"] > 0
                    ).sum()
                ),
                marker_coverage,
                int(text_df["question_mark"].sum()),
                int(text_df["exclamation_mark"].sum()),
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

    print("\nLanguage profiling complete.\n")


if __name__ == "__main__":
    main()