from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


INPUT_PATH = Path(
    "data/gold/olist/review_nlp_prepared.parquet"
)

MODEL_DIR = Path(
    "data/gold/olist/nlp_models"
)

MODEL_PATH = (
    MODEL_DIR /
    "tfidf_binary_dissatisfaction.joblib"
)

RESULT_PATH = Path(
    "data/gold/olist/tfidf_binary_dissatisfaction_results.parquet"
)


def create_binary_target(score):
    """
    Business-oriented dissatisfaction proxy.

    1–2 stars -> negative / dissatisfied
    3–5 stars -> not_negative

    This remains a rating-derived proxy, not independent
    human-annotated sentiment ground truth.
    """

    return "negative" if score <= 2 else "not_negative"


def main():
    print("\nNEXUS — Binary Dissatisfaction Benchmark")
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

    df["dissatisfaction_proxy"] = (
        df["review_score"]
        .apply(create_binary_target)
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

    print("\nDISSATISFACTION PROXY DISTRIBUTION")
    print("-" * 70)

    class_counts = (
        df["dissatisfaction_proxy"]
        .value_counts()
        .reindex(
            ["negative", "not_negative"],
            fill_value=0,
        )
    )

    for label, count in class_counts.items():
        print(
            f"{label:>12}: "
            f"{count:>7,} "
            f"({count / len(df) * 100:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Explicit temporal split
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
    # Features and targets
    # --------------------------------------------------------

    X_train = train["review_text_normalized"]
    y_train = train["dissatisfaction_proxy"]

    X_validation = validation[
        "review_text_normalized"
    ]
    y_validation = validation[
        "dissatisfaction_proxy"
    ]

    X_test = test["review_text_normalized"]
    y_test = test["dissatisfaction_proxy"]

    # --------------------------------------------------------
    # Model
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

    print(
        "TF-IDF + Logistic Regression training complete."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(split_name, X, y):
        predictions = model.predict(X)

        probabilities = model.predict_proba(X)

        classes = list(
            model.named_steps["classifier"].classes_
        )

        negative_index = classes.index("negative")

        negative_probability = probabilities[
            :, negative_index
        ]

        accuracy = accuracy_score(
            y,
            predictions,
        )

        negative_precision = precision_score(
            y,
            predictions,
            pos_label="negative",
            zero_division=0,
        )

        negative_recall = recall_score(
            y,
            predictions,
            pos_label="negative",
            zero_division=0,
        )

        negative_f1 = f1_score(
            y,
            predictions,
            pos_label="negative",
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            (y == "negative").astype(int),
            negative_probability,
        )

        pr_auc = average_precision_score(
            (y == "negative").astype(int),
            negative_probability,
        )

        print(
            f"\n{split_name.upper()} RESULTS"
        )
        print("-" * 70)

        print(
            f"Accuracy:             {accuracy:.6f}"
        )

        print(
            f"Negative precision:   {negative_precision:.6f}"
        )

        print(
            f"Negative recall:      {negative_recall:.6f}"
        )

        print(
            f"Negative F1:          {negative_f1:.6f}"
        )

        print(
            f"Negative ROC-AUC:     {roc_auc:.6f}"
        )

        print(
            f"Negative PR-AUC:      {pr_auc:.6f}"
        )

        print("\nClassification report:")

        print(
            classification_report(
                y,
                predictions,
                labels=[
                    "negative",
                    "not_negative",
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
                "not_negative",
            ],
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "actual_negative",
                "actual_not_negative",
            ],
            columns=[
                "pred_negative",
                "pred_not_negative",
            ],
        )

        print(cm_df.to_string())

        return {
            "split": split_name,
            "rows": len(y),
            "accuracy": accuracy,
            "negative_precision": negative_precision,
            "negative_recall": negative_recall,
            "negative_f1": negative_f1,
            "negative_roc_auc": roc_auc,
            "negative_pr_auc": pr_auc,
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
    # Save results
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

    print(
        "\nNEXUS binary dissatisfaction benchmark complete.\n"
    )


if __name__ == "__main__":
    main()