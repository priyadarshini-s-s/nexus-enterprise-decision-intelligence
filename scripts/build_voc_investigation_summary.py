from pathlib import Path
import json
import pandas as pd


# ============================================================
# NEXUS — VoC INVESTIGATION SUMMARY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
)

NLP_DIR = (
    GOLD_DIR
    / "nlp_models"
)

ROOT_CAUSE_DIR = (
    NLP_DIR
    / "root_cause"
)

ALERT_PATH = (
    NLP_DIR
    / "review_theme_alerts.parquet"
)

DECISION_RECORD_PATH = (
    NLP_DIR
    / "voc_decision_records.json"
)

CATEGORY_PATH = (
    ROOT_CAUSE_DIR
    / "voc_root_cause_category.parquet"
)

SELLER_PATH = (
    ROOT_CAUSE_DIR
    / "voc_root_cause_seller.parquet"
)

GEOGRAPHY_PATH = (
    ROOT_CAUSE_DIR
    / "voc_root_cause_geography.parquet"
)

STATUS_PATH = (
    ROOT_CAUSE_DIR
    / "voc_root_cause_order_status.parquet"
)

DELIVERY_PATH = (
    ROOT_CAUSE_DIR
    / "voc_root_cause_delivery.parquet"
)

OUTPUT_PATH = (
    NLP_DIR
    / "voc_investigation_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_MONTH = "2017-12"
TARGET_CLUSTER = 3

MIN_GROUP_REVIEWS = 20


# ============================================================
# HELPERS
# ============================================================

def load_required(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    return path


def pct(value):
    if pd.isna(value):
        return None

    return round(float(value) * 100, 4)


def num(value):
    if pd.isna(value):
        return None

    return round(float(value), 4)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NEXUS — VoC Investigation Summary")
print("=" * 70)


# ============================================================
# LOAD HIGH ALERT
# ============================================================

load_required(ALERT_PATH)

alerts = pd.read_parquet(
    ALERT_PATH
)

high_alerts = alerts[
    (
        alerts["alert_level"] == "HIGH"
    )
    &
    (
        alerts["month"].dt.strftime("%Y-%m")
        == TARGET_MONTH
    )
    &
    (
        alerts["semantic_theme_cluster"]
        == TARGET_CLUSTER
    )
].copy()


if high_alerts.empty:

    raise ValueError(
        "Expected December 2017 HIGH alert "
        "was not found."
    )


alert = high_alerts.iloc[0]


# ============================================================
# LOAD DECISION RECORD
# ============================================================

load_required(
    DECISION_RECORD_PATH
)

with open(
    DECISION_RECORD_PATH,
    "r",
    encoding="utf-8",
) as f:

    decision_records = json.load(f)


decision_record = next(
    (
        record
        for record in decision_records
        if record["decision_id"]
        == "VOC-2017-12-CLUSTER-3"
    ),
    None,
)


if decision_record is None:

    raise ValueError(
        "Decision record "
        "VOC-2017-12-CLUSTER-3 "
        "was not found."
    )


# ============================================================
# LOAD ROOT-CAUSE OUTPUTS
# ============================================================

load_required(CATEGORY_PATH)
load_required(SELLER_PATH)
load_required(GEOGRAPHY_PATH)
load_required(STATUS_PATH)
load_required(DELIVERY_PATH)


category_df = pd.read_parquet(
    CATEGORY_PATH
)

seller_df = pd.read_parquet(
    SELLER_PATH
)

geography_df = pd.read_parquet(
    GEOGRAPHY_PATH
)

status_df = pd.read_parquet(
    STATUS_PATH
)

delivery_df = pd.read_parquet(
    DELIVERY_PATH
)


# ============================================================
# TOP INVESTIGATION LEADS
# ============================================================

# Category:
# prioritize lift, but retain meaningful volume.

category_candidates = category_df[
    category_df["target_theme_count"]
    >= MIN_GROUP_REVIEWS
].copy()

category_candidates = (
    category_candidates
    .sort_values(
        [
            "theme_share_lift",
            "target_theme_count",
        ],
        ascending=False,
    )
)

top_categories = []

for _, row in (
    category_candidates
    .head(5)
    .iterrows()
):

    top_categories.append(
        {
            "product_category": str(
                row["product_category"]
            ),
            "review_count": int(
                row["review_count"]
            ),
            "target_theme_count": int(
                row["target_theme_count"]
            ),
            "target_theme_share_pct": pct(
                row["target_theme_share"]
            ),
            "theme_share_lift": num(
                row["theme_share_lift"]
            ),
            "target_dissatisfaction_pct": pct(
                row["target_dissatisfaction"]
            ),
            "target_late_delivery_pct": pct(
                row["target_late_rate"]
            ),
        }
    )


# Seller

seller_candidates = seller_df[
    seller_df["target_theme_count"]
    >= MIN_GROUP_REVIEWS
].copy()

seller_candidates = (
    seller_candidates
    .sort_values(
        [
            "theme_share_lift",
            "target_theme_count",
        ],
        ascending=False,
    )
)

top_sellers = []

for _, row in (
    seller_candidates
    .head(5)
    .iterrows()
):

    top_sellers.append(
        {
            "seller_id": str(
                row["seller_id"]
            ),
            "review_count": int(
                row["review_count"]
            ),
            "target_theme_count": int(
                row["target_theme_count"]
            ),
            "target_theme_share_pct": pct(
                row["target_theme_share"]
            ),
            "theme_share_lift": num(
                row["theme_share_lift"]
            ),
            "target_dissatisfaction_pct": pct(
                row["target_dissatisfaction"]
            ),
            "target_late_delivery_pct": pct(
                row["target_late_rate"]
            ),
        }
    )


# Geography

geography_candidates = geography_df[
    geography_df["target_theme_count"]
    >= MIN_GROUP_REVIEWS
].copy()

geography_candidates = (
    geography_candidates
    .sort_values(
        [
            "theme_share_lift",
            "target_theme_count",
        ],
        ascending=False,
    )
)

top_geographies = []

for _, row in (
    geography_candidates
    .head(5)
    .iterrows()
):

    top_geographies.append(
        {
            "customer_state": str(
                row["customer_state"]
            ),
            "review_count": int(
                row["review_count"]
            ),
            "target_theme_count": int(
                row["target_theme_count"]
            ),
            "target_theme_share_pct": pct(
                row["target_theme_share"]
            ),
            "theme_share_lift": num(
                row["theme_share_lift"]
            ),
            "target_dissatisfaction_pct": pct(
                row["target_dissatisfaction"]
            ),
            "target_late_delivery_pct": pct(
                row["target_late_rate"]
            ),
        }
    )


# ============================================================
# DELIVERY EVIDENCE
# ============================================================

delivery_records = []

for _, row in delivery_df.iterrows():

    delivery_records.append(
        {
            "group": str(
                row["group"]
            ),
            "review_count": int(
                row["review_count"]
            ),
            "mean_delivery_delay_days": num(
                row[
                    "mean_delivery_delay_days"
                ]
            ),
            "median_delivery_delay_days": num(
                row[
                    "median_delivery_delay_days"
                ]
            ),
            "late_delivery_rate_pct": pct(
                row[
                    "late_delivery_rate"
                ]
            ),
        }
    )


# ============================================================
# ORDER STATUS EVIDENCE
# ============================================================

status_records = []

for _, row in status_df.iterrows():

    status_records.append(
        {
            "order_status": (
                None
                if pd.isna(
                    row["order_status"]
                )
                else str(
                    row["order_status"]
                )
            ),

            "review_count": int(
                row["review_count"]
            ),

            "target_theme_count": int(
                row["target_theme_count"]
            ),

            "target_theme_share_pct": pct(
                row["target_theme_share"]
            ),

            "theme_share_lift": num(
                row["theme_share_lift"]
            ),

            "target_dissatisfaction_pct": pct(
                row[
                    "target_dissatisfaction"
                ]
            ),

            "target_late_delivery_pct": pct(
                row[
                    "target_late_rate"
                ]
            ),
        }
    )


# ============================================================
# DECISION SYNTHESIS
# ============================================================

summary = {

    "decision_id": (
        "VOC-2017-12-CLUSTER-3"
    ),

    "decision_type": (
        "VOC_ROOT_CAUSE_INVESTIGATION"
    ),

    "status": (
        "INVESTIGATION_REQUIRED"
    ),


    # --------------------------------------------------------
    # ALERT
    # --------------------------------------------------------

    "alert": {

        "month": TARGET_MONTH,

        "theme_cluster": TARGET_CLUSTER,

        "theme_name": str(
            alert["theme_name"]
        ),

        "alert_level": str(
            alert["alert_level"]
        ),

        "review_count": int(
            alert["review_count"]
        ),

        "theme_share_pct": pct(
            alert["theme_share"]
        ),

        "baseline_share_pct": pct(
            alert["baseline_share"]
        ),

        "share_z_score": num(
            alert["share_z_score"]
        ),

        "dissatisfaction_pct": pct(
            alert[
                "theme_dissatisfaction_rate"
            ]
        ),

        "baseline_dissatisfaction_pct": pct(
            alert[
                "baseline_dissatisfaction"
            ]
        ),

        "dissatisfaction_change_pct_points": (
            round(
                float(
                    alert[
                        "dissatisfaction_change"
                    ]
                )
                * 100,
                4,
            )
        ),

        "priority_score": num(
            alert["priority_score"]
        ),
    },


    # --------------------------------------------------------
    # OPERATIONAL EVIDENCE
    # --------------------------------------------------------

    "operational_evidence": {

        "delivery": delivery_records,

        "order_status": status_records,
    },


    # --------------------------------------------------------
    # INVESTIGATION LEADS
    # --------------------------------------------------------

    "investigation_leads": {

        "product_categories": (
            top_categories
        ),

        "sellers": (
            top_sellers
        ),

        "customer_geographies": (
            top_geographies
        ),
    },


    # --------------------------------------------------------
    # RECOMMENDED ACTION
    # --------------------------------------------------------

    "recommended_action": [

        "Investigate delivery performance "
        "for the December 2017 period.",

        "Review high-concentration "
        "product categories.",

        "Investigate sellers with elevated "
        "theme concentration.",

        "Investigate geographic concentrations "
        "where review volume is sufficient.",

        "Cross-check customer-review signals "
        "against operational order and "
        "delivery evidence.",
    ],


    # --------------------------------------------------------
    # ANALYTICAL BOUNDARY
    # --------------------------------------------------------

    "causal_status": (
        "NOT_ESTABLISHED"
    ),

    "analytical_boundary": (
        "The investigation identifies "
        "observational concentrations and "
        "associations in customer-review "
        "data and operational fields. "
        "It does not establish causality, "
        "incremental financial impact, or "
        "root cause."
    ),


    # --------------------------------------------------------
    # SOURCE TRACEABILITY
    # --------------------------------------------------------

    "source_artifacts": [

        "review_theme_alerts.parquet",

        "voc_decision_records.json",

        "voc_root_cause_category.parquet",

        "voc_root_cause_seller.parquet",

        "voc_root_cause_geography.parquet",

        "voc_root_cause_order_status.parquet",

        "voc_root_cause_delivery.parquet",
    ],
}


# ============================================================
# SAVE
# ============================================================

NLP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DECISION SUMMARY")
print("=" * 70)

print(
    "\nDecision ID:",
    summary["decision_id"]
)

print(
    "Theme:",
    summary["alert"]["theme_name"]
)

print(
    "Month:",
    summary["alert"]["month"]
)

print(
    "Alert:",
    summary["alert"]["alert_level"]
)

print(
    "Theme share:",
    f'{summary["alert"]["theme_share_pct"]:.2f}%'
)

print(
    "Baseline:",
    f'{summary["alert"]["baseline_share_pct"]:.2f}%'
)

print(
    "Z-score:",
    f'{summary["alert"]["share_z_score"]:.2f}'
)

print(
    "Dissatisfaction:",
    f'{summary["alert"]["dissatisfaction_pct"]:.2f}%'
)

print(
    "Dissatisfaction change:",
    f'+{summary["alert"]["dissatisfaction_change_pct_points"]:.2f} pp'
)


print("\nTop category investigation leads:")

for item in top_categories[:3]:

    print(
        f'  {item["product_category"]}: '
        f'{item["target_theme_share_pct"]:.2f}% '
        f'({item["theme_share_lift"]:.2f}x lift)'
    )


print("\nTop seller investigation leads:")

for item in top_sellers[:3]:

    print(
        f'  {item["seller_id"]}: '
        f'{item["target_theme_share_pct"]:.2f}% '
        f'({item["theme_share_lift"]:.2f}x lift)'
    )


print("\nTop geographic investigation leads:")

for item in top_geographies[:3]:

    print(
        f'  {item["customer_state"]}: '
        f'{item["target_theme_share_pct"]:.2f}% '
        f'({item["theme_share_lift"]:.2f}x lift)'
    )


print("\nCausal status:")
print(
    summary["causal_status"]
)


print("\nSaved:")
print(OUTPUT_PATH)


print("\n" + "=" * 70)
print("VOC INVESTIGATION SUMMARY COMPLETE")
print("=" * 70)