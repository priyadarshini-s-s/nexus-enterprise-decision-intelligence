from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALERTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
    / "review_theme_alerts.parquet"
)

REPRESENTATIVES_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
    / "review_theme_representatives.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "olist"
    / "nlp_models"
)

OUTPUT_PATH = OUTPUT_DIR / "voc_decision_records.json"


print("=" * 70)
print("NEXUS — VoC Decision Record Generator")
print("=" * 70)


# ============================================================
# LOAD ALERTS
# ============================================================

alerts = pd.read_parquet(ALERTS_PATH)

high_alerts = alerts[
    alerts["alert_level"] == "HIGH"
].copy()

print(f"\nHIGH alerts found: {len(high_alerts)}")


# ============================================================
# LOAD REPRESENTATIVE REVIEWS
# ============================================================

representatives = pd.read_parquet(
    REPRESENTATIVES_PATH
)

print(
    f"Representative review rows loaded: "
    f"{len(representatives):,}"
)

print(
    f"Representative columns: "
    f"{representatives.columns.tolist()}"
)


# ============================================================
# BUILD DECISION RECORDS
# ============================================================

records = []


for _, alert in high_alerts.iterrows():

    cluster = int(
        alert["semantic_theme_cluster"]
    )

    month = pd.Timestamp(
        alert["month"]
    ).strftime("%Y-%m")

    theme_name = str(
        alert["theme_name"]
    )


    # --------------------------------------------------------
    # Retrieve representative reviews
    # --------------------------------------------------------

    cluster_reps = representatives[
        representatives["cluster"] == cluster
    ].copy()

    cluster_reps = cluster_reps.sort_values(
        "rank"
    )


    review_examples = []

    for _, review in cluster_reps.head(5).iterrows():

        text = str(
            review["review_text"]
        ).strip()

        if text:

            review_examples.append(
                {
                    "review_id": str(
                        review["review_id"]
                    ),
                    "order_id": str(
                        review["order_id"]
                    ),
                    "review_score": int(
                        review["review_score"]
                    ),
                    "text": text,
                }
            )


    # --------------------------------------------------------
    # Build auditable decision record
    # --------------------------------------------------------

    record = {

        "decision_id": (
            f"VOC-{month}-"
            f"CLUSTER-{cluster}"
        ),

        "decision_type": (
            "VOC_THEME_ALERT"
        ),

        "status": (
            "REVIEW_REQUIRED"
        ),


        # ----------------------------------------------------
        # Source provenance
        # ----------------------------------------------------

        "source": {

            "dataset": (
                "Olist Brazilian "
                "E-Commerce Public Dataset"
            ),

            "domain": (
                "Customer Reviews / "
                "Voice of Customer"
            ),

            "analytical_grain": (
                "Monthly semantic theme "
                "review observations"
            ),
        },


        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        "signal": {

            "month": month,

            "theme_cluster": cluster,

            "theme_name": theme_name,

            "review_count": int(
                alert["review_count"]
            ),

            "theme_share": float(
                alert["theme_share"]
            ),

            "baseline_share": float(
                alert["baseline_share"]
            ),

            "share_deviation": float(
                alert["share_deviation"]
            ),

            "share_z_score": float(
                alert["share_z_score"]
            ),

            "dissatisfaction_rate": float(
                alert[
                    "theme_dissatisfaction_rate"
                ]
            ),

            "baseline_dissatisfaction_rate": float(
                alert[
                    "baseline_dissatisfaction"
                ]
            ),

            "dissatisfaction_change": float(
                alert[
                    "dissatisfaction_change"
                ]
            ),

            "priority_score": float(
                alert["priority_score"]
            ),

            "alert_level": str(
                alert["alert_level"]
            ),
        },


        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        "evidence": {

            "historical_baseline": (
                "Trailing 6-month "
                "theme baseline"
            ),

            "minimum_review_threshold": 100,

            "representative_reviews": (
                review_examples
            ),
        },


        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        "recommended_action": (
            "Investigate the underlying "
            "customer-review evidence and "
            "determine whether the signal "
            "corresponds to a delivery, "
            "service, seller, or fulfillment "
            "issue."
        ),


        # ----------------------------------------------------
        # Analytical boundary
        # ----------------------------------------------------

        "analytical_boundary": (
            "This alert identifies an "
            "abnormal customer-review "
            "signal. It does not establish "
            "operational causality or "
            "estimate financial impact."
        ),


        # ----------------------------------------------------
        # Root-cause investigation plan
        # ----------------------------------------------------

        "next_investigation": [

            "Break the theme down by "
            "product category.",

            "Break the theme down by "
            "seller.",

            "Compare affected orders "
            "by geography.",

            "Inspect delivery-time and "
            "order-status signals.",

            "Review representative "
            "customer comments.",
        ],
    }


    records.append(record)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        records,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# REPORT
# ============================================================

print("\nDecision records created:")
print(OUTPUT_PATH)


for record in records:

    signal = record["signal"]

    print("\n" + "-" * 70)

    print(
        "Decision ID:",
        record["decision_id"]
    )

    print(
        "Theme:",
        signal["theme_name"]
    )

    print(
        "Month:",
        signal["month"]
    )

    print(
        "Alert:",
        signal["alert_level"]
    )

    print(
        "Review count:",
        signal["review_count"]
    )

    print(
        "Theme share:",
        f"{signal['theme_share']:.2%}"
    )

    print(
        "Baseline share:",
        f"{signal['baseline_share']:.2%}"
    )

    print(
        "Share z-score:",
        f"{signal['share_z_score']:.2f}"
    )

    print(
        "Dissatisfaction:",
        f"{signal['dissatisfaction_rate']:.2%}"
    )

    print(
        "Dissatisfaction change:",
        f"{signal['dissatisfaction_change']:+.2%}"
    )

    print(
        "Representative reviews:",
        len(
            record[
                "evidence"
            ][
                "representative_reviews"
            ]
        )
    )


print("\n" + "=" * 70)
print("DECISION RECORD GENERATION COMPLETE")
print("=" * 70)