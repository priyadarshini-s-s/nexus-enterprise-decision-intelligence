from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
    / "review_theme_monthly_trends.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "review_theme_alerts.parquet"
)

TRAILING_MONTHS = 6

# Avoid alerts caused by extremely small samples.
MIN_THEME_REVIEWS = 100

# Alert thresholds.
SHARE_Z_WARNING = 2.0
SHARE_Z_HIGH = 3.0

DISSATISFACTION_CHANGE_WARNING = 0.10
DISSATISFACTION_CHANGE_HIGH = 0.20


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("NEXUS — Voice-of-Customer Theme Alert Engine")
print("=" * 70)

df = pd.read_parquet(INPUT_PATH)

df["month"] = pd.to_datetime(df["month"])

df = df.sort_values(
    [
        "semantic_theme_cluster",
        "month",
    ]
).copy()

print(f"\nLoaded monthly theme observations: {len(df):,}")


# ============================================================
# TRAILING BASELINES
# ============================================================

group = df.groupby(
    "semantic_theme_cluster",
    group_keys=False,
)

df["baseline_share"] = (
    group["theme_share"]
    .transform(
        lambda s: s.shift(1)
        .rolling(
            TRAILING_MONTHS,
            min_periods=3,
        )
        .mean()
    )
)

df["baseline_share_std"] = (
    group["theme_share"]
    .transform(
        lambda s: s.shift(1)
        .rolling(
            TRAILING_MONTHS,
            min_periods=3,
        )
        .std()
    )
)

df["baseline_dissatisfaction"] = (
    group["theme_dissatisfaction_rate"]
    .transform(
        lambda s: s.shift(1)
        .rolling(
            TRAILING_MONTHS,
            min_periods=3,
        )
        .mean()
    )
)


# ============================================================
# SHARE ANOMALY
# ============================================================

df["share_deviation"] = (
    df["theme_share"]
    - df["baseline_share"]
)

df["share_z_score"] = np.where(
    df["baseline_share_std"] > 0,
    df["share_deviation"]
    / df["baseline_share_std"],
    np.nan,
)


# ============================================================
# DISSATISFACTION CHANGE
# ============================================================

df["dissatisfaction_change"] = (
    df["theme_dissatisfaction_rate"]
    - df["baseline_dissatisfaction"]
)


# ============================================================
# ELIGIBILITY
# ============================================================

df["eligible_for_alert"] = (
    (df["review_count"] >= MIN_THEME_REVIEWS)
    & df["baseline_share"].notna()
    & df["baseline_dissatisfaction"].notna()
)


# ============================================================
# ALERT LOGIC
# ============================================================

def classify_alert(row):

    if not row["eligible_for_alert"]:
        return "INSUFFICIENT_HISTORY"

    share_z = row["share_z_score"]
    dissatisfaction_change = (
        row["dissatisfaction_change"]
    )

    high_share = (
        pd.notna(share_z)
        and share_z >= SHARE_Z_HIGH
    )

    warning_share = (
        pd.notna(share_z)
        and share_z >= SHARE_Z_WARNING
    )

    high_dissatisfaction = (
        dissatisfaction_change
        >= DISSATISFACTION_CHANGE_HIGH
    )

    warning_dissatisfaction = (
        dissatisfaction_change
        >= DISSATISFACTION_CHANGE_WARNING
    )

    # Highest priority:
    # unusually high theme share AND materially
    # higher dissatisfaction.

    if (
        high_share
        and high_dissatisfaction
    ):
        return "HIGH"

    if (
        (high_share and warning_dissatisfaction)
        or
        (warning_share and high_dissatisfaction)
    ):
        return "HIGH"

    if (
        warning_share
        or warning_dissatisfaction
    ):
        return "WARNING"

    return "NORMAL"


df["alert_level"] = df.apply(
    classify_alert,
    axis=1,
)


# ============================================================
# PRIORITY SCORE
# ============================================================

# Transparent analytical score.
#
# This is NOT an ROI estimate.
# It simply ranks stronger signals above weaker ones.

share_component = (
    df["share_z_score"]
    .clip(
        lower=0,
        upper=5,
    )
    .fillna(0)
    / 5
)

dissatisfaction_component = (
    df["dissatisfaction_change"]
    .clip(
        lower=0,
        upper=0.50,
    )
    .fillna(0)
    / 0.50
)

volume_component = (
    np.log1p(
        df["review_count"]
    )
    / np.log1p(
        df["review_count"].max()
    )
)

df["priority_score"] = (
    0.45 * share_component
    + 0.40 * dissatisfaction_component
    + 0.15 * volume_component
)

df["priority_score"] = (
    df["priority_score"]
    .clip(0, 1)
)


# ============================================================
# DECISION TEXT
# ============================================================

def build_decision(row):

    if row["alert_level"] == "HIGH":
        return (
            "Investigate this customer-experience theme "
            "as a high-priority signal."
        )

    if row["alert_level"] == "WARNING":
        return (
            "Monitor this theme and investigate "
            "the underlying review evidence."
        )

    if row["alert_level"] == "NORMAL":
        return (
            "No abnormal VoC signal detected "
            "relative to the trailing baseline."
        )

    return (
        "Insufficient historical evidence "
        "for a reliable alert."
    )


df["recommended_action"] = df.apply(
    build_decision,
    axis=1,
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    "month",
    "semantic_theme_cluster",
    "theme_name",
    "review_count",
    "theme_share",
    "baseline_share",
    "share_deviation",
    "share_z_score",
    "theme_dissatisfaction_rate",
    "baseline_dissatisfaction",
    "dissatisfaction_change",
    "eligible_for_alert",
    "alert_level",
    "priority_score",
    "recommended_action",
]

output = df[
    output_columns
].sort_values(
    [
        "month",
        "priority_score",
    ],
    ascending=[
        True,
        False,
    ],
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output.to_parquet(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\nAlert summary")
print("=" * 70)

print(
    output["alert_level"]
    .value_counts()
    .to_string()
)


print("\nHigh-priority signals")
print("=" * 70)

high = output[
    output["alert_level"] == "HIGH"
].sort_values(
    "priority_score",
    ascending=False,
)

if len(high) == 0:

    print("No HIGH signals detected.")

else:

    print(
        high[
            [
                "month",
                "theme_name",
                "review_count",
                "theme_share",
                "baseline_share",
                "share_z_score",
                "theme_dissatisfaction_rate",
                "baseline_dissatisfaction",
                "dissatisfaction_change",
                "priority_score",
            ]
        ].head(20).to_string(
            index=False,
            formatters={
                "theme_share": "{:.2%}".format,
                "baseline_share": "{:.2%}".format,
                "share_z_score": "{:.2f}".format,
                "theme_dissatisfaction_rate": (
                    "{:.2%}".format
                ),
                "baseline_dissatisfaction": (
                    "{:.2%}".format
                ),
                "dissatisfaction_change": (
                    "{:+.2%}".format
                ),
                "priority_score": "{:.3f}".format,
            },
        )
    )


print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("VOC ALERT ENGINE COMPLETE")
print("=" * 70)