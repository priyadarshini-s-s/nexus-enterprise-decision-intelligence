from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "review_theme_assignments.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
)

MONTHLY_PATH = (
    OUTPUT_DIR
    / "review_theme_monthly_trends.parquet"
)

CLUSTER_PATH = (
    OUTPUT_DIR
    / "review_theme_cluster_summary.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("NEXUS — Voice-of-Customer Theme Trend Analysis")
print("=" * 70)

df = pd.read_parquet(ASSIGNMENT_PATH)

print(f"\nLoaded: {len(df):,} review-theme assignments")


# ============================================================
# DATE PREPARATION
# ============================================================

df["review_creation_date"] = pd.to_datetime(
    df["review_creation_date"],
    errors="coerce",
)

df = df.dropna(
    subset=["review_creation_date"]
).copy()

df["month"] = (
    df["review_creation_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ============================================================
# THEME LABELS
# ============================================================

# These are analyst interpretations based on the
# representative-review inspection.
#
# They are NOT ML ground-truth labels.

theme_labels = {
    0: "Overall Positive / Trust & Satisfaction",
    1: "Positive Delivery & Fulfillment",
    2: "Generic Positive Satisfaction",
    3: "Delivery & Service Failures",
    4: "Product Quality / Product Mismatch",
}

df["theme_name"] = (
    df["semantic_theme_cluster"]
    .map(theme_labels)
    .fillna(
        "Uninterpreted Semantic Cluster"
    )
)


# ============================================================
# DISSATISFACTION FLAG
# ============================================================

df["dissatisfied"] = (
    df["review_score"] <= 2
).astype(int)


# ============================================================
# MONTHLY THEME METRICS
# ============================================================

monthly = (
    df.groupby(
        [
            "month",
            "semantic_theme_cluster",
            "theme_name",
        ],
        as_index=False,
    )
    .agg(
        review_count=(
            "review_score",
            "size",
        ),
        mean_review_score=(
            "review_score",
            "mean",
        ),
        dissatisfied_reviews=(
            "dissatisfied",
            "sum",
        ),
    )
)

# Total reviews per month

monthly_totals = (
    df.groupby(
        "month",
        as_index=False,
    )
    .agg(
        total_reviews=(
            "review_score",
            "size",
        ),
        total_dissatisfied=(
            "dissatisfied",
            "sum",
        ),
    )
)

monthly = monthly.merge(
    monthly_totals,
    on="month",
    how="left",
)

monthly["theme_share"] = (
    monthly["review_count"]
    / monthly["total_reviews"]
)

monthly["theme_dissatisfaction_rate"] = (
    monthly["dissatisfied_reviews"]
    / monthly["review_count"]
)

monthly["overall_month_dissatisfaction_rate"] = (
    monthly["total_dissatisfied"]
    / monthly["total_reviews"]
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    df.groupby(
        [
            "semantic_theme_cluster",
            "theme_name",
        ],
        as_index=False,
    )
    .agg(
        review_count=(
            "review_score",
            "size",
        ),
        mean_review_score=(
            "review_score",
            "mean",
        ),
        dissatisfied_reviews=(
            "dissatisfied",
            "sum",
        ),
    )
)

cluster_summary["share"] = (
    cluster_summary["review_count"]
    / len(df)
)

cluster_summary["dissatisfaction_rate"] = (
    cluster_summary["dissatisfied_reviews"]
    / cluster_summary["review_count"]
)


# ============================================================
# MONTH-OVER-MONTH CHANGE
# ============================================================

monthly = monthly.sort_values(
    [
        "semantic_theme_cluster",
        "month",
    ]
)

monthly["theme_share_mom_change"] = (
    monthly.groupby(
        "semantic_theme_cluster"
    )["theme_share"]
    .diff()
)

monthly["dissatisfaction_rate_mom_change"] = (
    monthly.groupby(
        "semantic_theme_cluster"
    )["theme_dissatisfaction_rate"]
    .diff()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

monthly.to_parquet(
    MONTHLY_PATH,
    index=False,
)

cluster_summary.to_parquet(
    CLUSTER_PATH,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\nOverall cluster summary")
print("=" * 70)

display_columns = [
    "semantic_theme_cluster",
    "theme_name",
    "review_count",
    "share",
    "mean_review_score",
    "dissatisfaction_rate",
]

print(
    cluster_summary[
        display_columns
    ].sort_values(
        "review_count",
        ascending=False,
    ).to_string(
        index=False,
        formatters={
            "share": "{:.2%}".format,
            "mean_review_score": "{:.3f}".format,
            "dissatisfaction_rate": "{:.2%}".format,
        },
    )
)


# ============================================================
# MONTHLY DISSATISFACTION THEMES
# ============================================================

negative_themes = monthly[
    monthly["semantic_theme_cluster"].isin(
        [3, 4]
    )
].copy()

print("\nMonthly negative-theme trend")
print("=" * 70)

print(
    negative_themes[
        [
            "month",
            "semantic_theme_cluster",
            "theme_name",
            "review_count",
            "theme_share",
            "theme_dissatisfaction_rate",
        ]
    ].to_string(
        index=False,
        formatters={
            "theme_share": "{:.2%}".format,
            "theme_dissatisfaction_rate": (
                "{:.2%}".format
            ),
        },
    )
)


print("\nSaved monthly trends:")
print(MONTHLY_PATH)

print("\nSaved cluster summary:")
print(CLUSTER_PATH)

print("\n" + "=" * 70)
print("VOC TREND ANALYSIS COMPLETE")
print("=" * 70)