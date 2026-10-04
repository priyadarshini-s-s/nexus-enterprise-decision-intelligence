from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


INPUT_PATH = Path(
    "data/gold/olist/review_nlp_prepared.parquet"
)

MODEL_DIR = Path(
    "data/gold/olist/nlp_models"
)

MODEL_PATH = MODEL_DIR / "tfidf_sentiment_baseline.joblib"

RESULT_PATH = Path(
    "data/gold/olist/tfidf_sentiment_baseline_results.parquet"
)


def create_sentiment_proxy(score):
    """
    Rating-derived sentiment proxy.

    1–2 stars -> negative
    3 stars   -> neutral
    4–5 stars -> positive

    IMPORTANT:
    This is not independently annotated sentiment ground truth.
    It is used only as a weak/observational benchmark.
    """

    if score <= 2:
        return "negative"

    if score == 3:
        return "neutral"

    return "positive"


def main():
    print("\nNEXUS — TF-IDF Sentiment Baseline")
    print("=" * 70)

    df = pd.read_parquet(INPUT_PATH)

    # --------------------------------------------------------
    # Text-bearing observations only
    # --------------------------------------------------------

    df = df[
        df["review_text_normalized"]
        .fillna("")
        .str.strip()
        .ne("")
    ].copy()

    df["rating_sentiment_proxy"] = (
        df["review_score"]
        .apply(create_sentiment_proxy)
    )

    df["review_creation_date"] = pd.to_datetime(
        df["review_creation_date"],
        errors="coerce",
    )

    df = df.sort_values(
        [
            "review_creation_date",
            "review_observation_id",
        ]
    ).reset_index(drop=True)

    print("\nDATASET")
    print("-" * 70)
    print(f"Text-bearing rows: {len(df):,}")
    print(
        f"First review: "
        f"{df['review_creation_date'].min()}"
    )
    print(
        f"Last review: "
        f"{df['review_creation_date'].max()}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nSENTIMENT PROXY DISTRIBUTION")
    print("-" * 70)

    class_counts = (
        df["rating_sentiment_proxy"]
        .value_counts()
        .reindex(
            ["negative", "neutral", "positive"],
            fill_value=0,
        )
    )

    for label, count in class_counts.items():
        print(
            f"{label:>10}: "
            f"{count:>7,} "
            f"({count / len(df) * 100:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Temporal split
    #
    # We use explicit calendar boundaries rather than row-count
    # percentages.
    #
    # This prevents multiple observations from the same calendar
    # date from being split across train/validation/test.
    # --------------------------------------------------------

    TRAIN_END = pd.Timestamp("2018-04-30")
    VALIDATION_END = pd.Timestamp("2018-06-30")

    train = df[
        df["review_creation_date"] <= TRAIN_END
    ].copy()

    validation = df[
        (df["review_creation_date"] > TRAIN_END)
        & (df["review_creation_date"] <= VALIDATION_END)
    ].copy()

    test = df[
        df["review_creation_date"] > VALIDATION_END
    ].copy()

    print("\nTEMPORAL SPLIT")
    print("-" * 70)

    print(
        f"TRAIN      : {len(train):,} | "
        f"{train['review_creation_date'].min().date()} "
        f"→ "
        f"{train['review_creation_date'].max().date()}"
    )

    print(
        f"VALIDATION : {len(validation):,} | "
        f"{validation['review_creation_date'].min().date()} "
        f"→ "
        f"{validation['review_creation_date'].max().date()}"
    )

    print(
        f"TEST       : {len(test):,} | "
        f"{test['review_creation_date'].min().date()} "
        f"→ "
        f"{test['review_creation_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Validate temporal ordering
    # --------------------------------------------------------

    if (
        train["review_creation_date"].max()
        >= validation["review_creation_date"].min()
    ):
        raise ValueError(
            "Temporal leakage detected between train and validation."
        )

    if (
        validation["review_creation_date"].max()
        >= test["review_creation_date"].min()
    ):
        raise ValueError(
            "Temporal leakage detected between validation and test."
        )

    # --------------------------------------------------------
    # Features / targets
    # --------------------------------------------------------

    X_train = train["review_text_normalized"]
    y_train = train["rating_sentiment_proxy"]

    X_validation = validation[
        "review_text_normalized"
    ]
    y_validation = validation[
        "rating_sentiment_proxy"
    ]

    X_test = test["review_text_normalized"]
    y_test = test["rating_sentiment_proxy"]

    # --------------------------------------------------------
    # TF-IDF + Logistic Regression
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTRAINING")
    print("-" * 70)

    model.fit(
        X_train,
        y_train,
    )

    print("TF-IDF + Logistic Regression training complete.")

    # --------------------------------------------------------
    # Evaluation helper
    # --------------------------------------------------------

    def evaluate(split_name, X, y):
        predictions = model.predict(X)

        accuracy = accuracy_score(
            y,
            predictions,
        )

        macro_f1 = f1_score(
            y,
            predictions,
            average="macro",
        )

        weighted_f1 = f1_score(
            y,
            predictions,
            average="weighted",
        )

        print(
            f"\n{split_name.upper()} RESULTS"
        )
        print("-" * 70)

        print(
            f"Accuracy:    {accuracy:.6f}"
        )

        print(
            f"Macro F1:    {macro_f1:.6f}"
        )

        print(
            f"Weighted F1: {weighted_f1:.6f}"
        )

        print("\nClassification report:")

        print(
            classification_report(
                y,
                predictions,
                labels=[
                    "negative",
                    "neutral",
                    "positive",
                ],
                digits=4,
                zero_division=0,
            )
        )

        print("Confusion matrix:")

        cm = confusion_matrix(
            y,
            predictions,
            labels=[
                "negative",
                "neutral",
                "positive",
            ],
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "actual_negative",
                "actual_neutral",
                "actual_positive",
            ],
            columns=[
                "pred_negative",
                "pred_neutral",
                "pred_positive",
            ],
        )

        print(cm_df.to_string())

        return {
            "split": split_name,
            "rows": len(y),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    train_result = evaluate(
        "train",
        X_train,
        y_train,
    )

    validation_result = evaluate(
        "validation",
        X_validation,
        y_validation,
    )

    test_result = evaluate(
        "test",
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            train_result,
            validation_result,
            test_result,
        ]
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        RESULT_PATH,
        index=False,
    )

    print("\nOUTPUT")
    print("-" * 70)
    print(f"Model:   {MODEL_PATH}")
    print(f"Results: {RESULT_PATH}")

    print("\nNEXUS TF-IDF sentiment baseline complete.\n")


if __name__ == "__main__":
    main()