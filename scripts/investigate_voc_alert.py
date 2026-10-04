from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NEXUS — VoC ROOT-CAUSE INVESTIGATION
# ============================================================
#
# Purpose:
# Investigate the December 2017 HIGH VoC alert:
#
#   Theme: Delivery & Service Failures
#   Cluster: 3
#
# Dimensions:
#   1. Product category
#   2. Seller
#   3. Customer geography
#   4. Order status
#   5. Delivery performance
#
# Important:
# These are observational associations/concentrations.
# They do NOT establish causality.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_DIR = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist"
)

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

OUTPUT_DIR = (
    NLP_DIR
    / "root_cause"
)

# Alert identified by the previous VoC pipeline.
TARGET_MONTH = pd.Timestamp("2017-12-01")
TARGET_CLUSTER = 3

# Minimum target-theme observations required
# before reporting a group as a meaningful signal.
MIN_GROUP_REVIEWS = 20


# ============================================================
# HELPERS
# ============================================================

def load_silver(filename):
    """
    Load an Olist Silver Parquet artifact.
    """
    path = SILVER_DIR / filename

    print(f"Loading {filename}...")

    if not path.exists():
        raise FileNotFoundError(
            f"Required Silver file not found:\n{path}"
        )

    return pd.read_parquet(path)


def safe_divide(numerator, denominator):
    """
    Safe division that returns NaN when denominator is zero.
    """
    if denominator == 0:
        return np.nan

    return numerator / denominator


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NEXUS — VoC Root-Cause Investigation")
print("=" * 70)


# ============================================================
# LOAD SILVER DATA
# ============================================================

orders = load_silver(
    "orders.parquet"
)

order_items = load_silver(
    "order_items.parquet"
)

products = load_silver(
    "products.parquet"
)

sellers = load_silver(
    "sellers.parquet"
)

customers = load_silver(
    "customers.parquet"
)


# ============================================================
# LOAD CANONICAL NLP THEME ASSIGNMENTS
# ============================================================
#
# IMPORTANT:
# review_theme_assignments.parquet uses:
#
#   semantic_theme_cluster
#
# The representative-review artifact uses:
#
#   cluster
#
# We intentionally preserve those source schemas.
# ============================================================

THEME_ASSIGNMENTS_PATH = (
    GOLD_DIR
    / "review_theme_assignments.parquet"
)

print(
    "Loading review_theme_assignments.parquet..."
)

if not THEME_ASSIGNMENTS_PATH.exists():
    raise FileNotFoundError(
        "Theme assignment artifact not found:\n"
        f"{THEME_ASSIGNMENTS_PATH}"
    )

review_table = pd.read_parquet(
    THEME_ASSIGNMENTS_PATH
)


# ============================================================
# VALIDATE THEME ASSIGNMENT SCHEMA
# ============================================================

required_columns = [
    "review_observation_id",
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "semantic_theme_cluster",
]

missing_columns = [
    column
    for column in required_columns
    if column not in review_table.columns
]

if missing_columns:

    raise ValueError(
        "Theme assignment artifact is missing "
        f"required columns: {missing_columns}"
    )


review_table = review_table[
    required_columns
].copy()


review_table["review_creation_date"] = pd.to_datetime(
    review_table["review_creation_date"],
    errors="coerce",
)


# ============================================================
# NORMALIZE ORDER DATES
# ============================================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce",
)

orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"],
    errors="coerce",
)

orders["order_estimated_delivery_date"] = pd.to_datetime(
    orders["order_estimated_delivery_date"],
    errors="coerce",
)


# ============================================================
# SELECT DECEMBER 2017 REVIEWS
# ============================================================

month_start = TARGET_MONTH

month_end = (
    TARGET_MONTH
    + pd.offsets.MonthBegin(1)
)


month_reviews = review_table[
    (
        review_table[
            "review_creation_date"
        ]
        >= month_start
    )
    &
    (
        review_table[
            "review_creation_date"
        ]
        < month_end
    )
].copy()


month_reviews["is_target_theme"] = (
    month_reviews[
        "semantic_theme_cluster"
    ]
    == TARGET_CLUSTER
)


month_reviews["is_dissatisfied"] = (
    month_reviews[
        "review_score"
    ]
    <= 2
)


print(
    "\nDecember 2017 review observations:",
    f"{len(month_reviews):,}"
)

print(
    "Target-theme observations:",
    int(
        month_reviews[
            "is_target_theme"
        ].sum()
    ),
)


# ============================================================
# JOIN ORDER CONTEXT
# ============================================================

analysis = (
    month_reviews
    .merge(
        orders,
        on="order_id",
        how="left",
        suffixes=(
            "",
            "_order",
        ),
        validate="many_to_one",
    )
)


# ============================================================
# JOIN CUSTOMER CONTEXT
# ============================================================

customer_columns = [
    "customer_id",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
]

missing_customer_columns = [
    column
    for column in customer_columns
    if column not in customers.columns
]

if missing_customer_columns:

    raise ValueError(
        "Customers artifact is missing columns: "
        f"{missing_customer_columns}"
    )


analysis = (
    analysis
    .merge(
        customers[
            customer_columns
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
)


# ============================================================
# BUILD ORDER-LEVEL PRODUCT / SELLER CONTEXT
# ============================================================
#
# order_items is line-item grain.
#
# We DO NOT directly join order_items to reviews because
# doing so would duplicate review observations for
# multi-item orders.
#
# Instead, we first aggregate item information to order grain.
# ============================================================

item_context = (
    order_items
    .merge(
        products[
            [
                "product_id",
                "product_category_name",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    .merge(
        sellers[
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ]
        ],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )
)


def unique_join(series):

    values = (
        series
        .dropna()
        .astype(str)
        .unique()
    )

    values = sorted(values)

    return " | ".join(values)


order_product_context = (
    item_context
    .groupby(
        "order_id",
        as_index=False,
    )
    .agg(
        product_categories=(
            "product_category_name",
            unique_join,
        ),
        seller_ids=(
            "seller_id",
            unique_join,
        ),
        seller_states=(
            "seller_state",
            unique_join,
        ),
    )
)


analysis = (
    analysis
    .merge(
        order_product_context,
        on="order_id",
        how="left",
        validate="many_to_one",
    )
)


# ============================================================
# DELIVERY METRICS
# ============================================================

analysis[
    "delivery_delay_days"
] = (
    analysis[
        "order_delivered_customer_date"
    ]
    - analysis[
        "order_estimated_delivery_date"
    ]
).dt.total_seconds() / 86400


analysis[
    "purchase_to_delivery_days"
] = (
    analysis[
        "order_delivered_customer_date"
    ]
    - analysis[
        "order_purchase_timestamp"
    ]
).dt.total_seconds() / 86400


analysis[
    "delivered_late"
] = (
    analysis[
        "delivery_delay_days"
    ]
    > 0
)


analysis[
    "has_delivery_date"
] = (
    analysis[
        "order_delivered_customer_date"
    ]
    .notna()
)


# ============================================================
# OVERALL BENCHMARK
# ============================================================

overall_review_count = len(
    analysis
)

target_reviews = analysis[
    analysis["is_target_theme"]
].copy()

target_review_count = len(
    target_reviews
)


overall_dissatisfaction = (
    analysis[
        "is_dissatisfied"
    ].mean()
)


target_dissatisfaction = (
    target_reviews[
        "is_dissatisfied"
    ].mean()
)


overall_late_rate = (
    analysis[
        "delivered_late"
    ].mean()
)


target_late_rate = (
    target_reviews[
        "delivered_late"
    ].mean()
)


overall_theme_share = safe_divide(
    target_review_count,
    overall_review_count,
)


print("\nOverall benchmark")
print("-" * 70)

print(
    "Reviews:",
    overall_review_count
)

print(
    "Target-theme reviews:",
    target_review_count
)

print(
    "Target-theme share:",
    f"{overall_theme_share:.2%}"
)

print(
    "Overall dissatisfaction:",
    f"{overall_dissatisfaction:.2%}"
)

print(
    "Target-theme dissatisfaction:",
    f"{target_dissatisfaction:.2%}"
)

print(
    "Overall late-delivery rate:",
    f"{overall_late_rate:.2%}"
)

print(
    "Target-theme late-delivery rate:",
    f"{target_late_rate:.2%}"
)


# ============================================================
# 1. PRODUCT CATEGORY INVESTIGATION
# ============================================================

category_rows = []


category_values = (
    item_context[
        "product_category_name"
    ]
    .dropna()
    .astype(str)
    .unique()
)


for category in category_values:

    category_orders = set(
        item_context.loc[
            item_context[
                "product_category_name"
            ].astype(str)
            == category,
            "order_id",
        ]
    )

    subset = analysis[
        analysis[
            "order_id"
        ].isin(category_orders)
    ]

    target = subset[
        subset["is_target_theme"]
    ]

    if (
        len(target)
        < MIN_GROUP_REVIEWS
    ):
        continue

    category_rows.append(
        {
            "product_category": category,

            "review_count": len(
                subset
            ),

            "target_theme_count": len(
                target
            ),

            "target_theme_share": safe_divide(
                len(target),
                len(subset),
            ),

            "theme_share_lift": safe_divide(
                safe_divide(
                    len(target),
                    len(subset),
                ),
                overall_theme_share,
            ),

            "target_dissatisfaction": (
                target[
                    "is_dissatisfied"
                ].mean()
            ),

            "target_late_rate": (
                target[
                    "delivered_late"
                ].mean()
            ),
        }
    )


category_df = pd.DataFrame(
    category_rows
)


if len(category_df):

    category_df = category_df.sort_values(
        [
            "theme_share_lift",
            "target_theme_count",
        ],
        ascending=False,
    )


# ============================================================
# 2. SELLER INVESTIGATION
# ============================================================

seller_rows = []


seller_values = (
    item_context[
        "seller_id"
    ]
    .dropna()
    .astype(str)
    .unique()
)


for seller in seller_values:

    seller_orders = set(
        item_context.loc[
            item_context[
                "seller_id"
            ].astype(str)
            == seller,
            "order_id",
        ]
    )

    subset = analysis[
        analysis[
            "order_id"
        ].isin(seller_orders)
    ]

    target = subset[
        subset["is_target_theme"]
    ]

    if (
        len(target)
        < MIN_GROUP_REVIEWS
    ):
        continue

    seller_rows.append(
        {
            "seller_id": seller,

            "review_count": len(
                subset
            ),

            "target_theme_count": len(
                target
            ),

            "target_theme_share": safe_divide(
                len(target),
                len(subset),
            ),

            "theme_share_lift": safe_divide(
                safe_divide(
                    len(target),
                    len(subset),
                ),
                overall_theme_share,
            ),

            "target_dissatisfaction": (
                target[
                    "is_dissatisfied"
                ].mean()
            ),

            "target_late_rate": (
                target[
                    "delivered_late"
                ].mean()
            ),
        }
    )


seller_df = pd.DataFrame(
    seller_rows
)


if len(seller_df):

    seller_df = seller_df.sort_values(
        [
            "theme_share_lift",
            "target_theme_count",
        ],
        ascending=False,
    )


# ============================================================
# 3. CUSTOMER GEOGRAPHY
# ============================================================

state_df = (
    analysis
    .groupby(
        "customer_state",
        dropna=False,
    )
    .agg(
        review_count=(
            "review_observation_id",
            "count",
        ),

        target_theme_count=(
            "is_target_theme",
            "sum",
        ),

        target_dissatisfaction=(
            "is_dissatisfied",
            "mean",
        ),

        target_late_rate=(
            "delivered_late",
            "mean",
        ),
    )
    .reset_index()
)


state_df[
    "target_theme_share"
] = (
    state_df[
        "target_theme_count"
    ]
    / state_df[
        "review_count"
    ]
)


state_df[
    "theme_share_lift"
] = (
    state_df[
        "target_theme_share"
    ]
    / overall_theme_share
)


state_df = state_df[
    state_df[
        "target_theme_count"
    ]
    >= MIN_GROUP_REVIEWS
].copy()


if len(state_df):

    state_df = state_df.sort_values(
        [
            "theme_share_lift",
            "target_theme_count",
        ],
        ascending=False,
    )


# ============================================================
# 4. ORDER STATUS INVESTIGATION
# ============================================================

status_df = (
    analysis
    .groupby(
        "order_status",
        dropna=False,
    )
    .agg(
        review_count=(
            "review_observation_id",
            "count",
        ),

        target_theme_count=(
            "is_target_theme",
            "sum",
        ),

        target_dissatisfaction=(
            "is_dissatisfied",
            "mean",
        ),

        target_late_rate=(
            "delivered_late",
            "mean",
        ),
    )
    .reset_index()
)


status_df[
    "target_theme_share"
] = (
    status_df[
        "target_theme_count"
    ]
    / status_df[
        "review_count"
    ]
)


status_df[
    "theme_share_lift"
] = (
    status_df[
        "target_theme_share"
    ]
    / overall_theme_share
)


# ============================================================
# 5. DELIVERY PERFORMANCE INVESTIGATION
# ============================================================

delivery_summary = pd.DataFrame(
    [
        {
            "group": (
                "All December reviews"
            ),

            "review_count": len(
                analysis
            ),

            "mean_delivery_delay_days": (
                analysis[
                    "delivery_delay_days"
                ].mean()
            ),

            "median_delivery_delay_days": (
                analysis[
                    "delivery_delay_days"
                ].median()
            ),

            "late_delivery_rate": (
                analysis[
                    "delivered_late"
                ].mean()
            ),
        },

        {
            "group": (
                "Delivery & Service "
                "Failure reviews"
            ),

            "review_count": len(
                target_reviews
            ),

            "mean_delivery_delay_days": (
                target_reviews[
                    "delivery_delay_days"
                ].mean()
            ),

            "median_delivery_delay_days": (
                target_reviews[
                    "delivery_delay_days"
                ].median()
            ),

            "late_delivery_rate": (
                target_reviews[
                    "delivered_late"
                ].mean()
            ),
        },
    ]
)


# ============================================================
# 6. ORDER STATUS + DELIVERY CROSS-TAB
# ============================================================

status_target = (
    analysis
    .groupby(
        [
            "order_status",
            "is_target_theme",
        ],
        dropna=False,
    )
    .agg(
        review_count=(
            "review_observation_id",
            "count",
        ),

        dissatisfaction_rate=(
            "is_dissatisfied",
            "mean",
        ),

        late_delivery_rate=(
            "delivered_late",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


category_df.to_parquet(
    OUTPUT_DIR
    / "voc_root_cause_category.parquet",
    index=False,
)


seller_df.to_parquet(
    OUTPUT_DIR
    / "voc_root_cause_seller.parquet",
    index=False,
)


state_df.to_parquet(
    OUTPUT_DIR
    / "voc_root_cause_geography.parquet",
    index=False,
)


status_df.to_parquet(
    OUTPUT_DIR
    / "voc_root_cause_order_status.parquet",
    index=False,
)


delivery_summary.to_parquet(
    OUTPUT_DIR
    / "voc_root_cause_delivery.parquet",
    index=False,
)


status_target.to_parquet(
    OUTPUT_DIR
    / "voc_root_cause_status_theme_crosscheck.parquet",
    index=False,
)


# ============================================================
# DISPLAY CATEGORY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT CATEGORY SIGNALS")
print("=" * 70)


if len(category_df):

    print(
        category_df[
            [
                "product_category",
                "review_count",
                "target_theme_count",
                "target_theme_share",
                "theme_share_lift",
                "target_dissatisfaction",
                "target_late_rate",
            ]
        ]
        .head(15)
        .to_string(
            index=False,
            formatters={
                "target_theme_share": (
                    "{:.2%}".format
                ),

                "theme_share_lift": (
                    "{:.2f}x".format
                ),

                "target_dissatisfaction": (
                    "{:.2%}".format
                ),

                "target_late_rate": (
                    "{:.2%}".format
                ),
            },
        )
    )

else:

    print(
        "No category met the minimum "
        "target-theme volume."
    )


# ============================================================
# DISPLAY SELLER RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SELLER SIGNALS")
print("=" * 70)


if len(seller_df):

    print(
        seller_df[
            [
                "seller_id",
                "review_count",
                "target_theme_count",
                "target_theme_share",
                "theme_share_lift",
                "target_dissatisfaction",
                "target_late_rate",
            ]
        ]
        .head(15)
        .to_string(
            index=False,
            formatters={
                "target_theme_share": (
                    "{:.2%}".format
                ),

                "theme_share_lift": (
                    "{:.2f}x".format
                ),

                "target_dissatisfaction": (
                    "{:.2%}".format
                ),

                "target_late_rate": (
                    "{:.2%}".format
                ),
            },
        )
    )

else:

    print(
        "No seller met the minimum "
        "target-theme volume."
    )


# ============================================================
# DISPLAY GEOGRAPHIC RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GEOGRAPHIC SIGNALS")
print("=" * 70)


if len(state_df):

    print(
        state_df[
            [
                "customer_state",
                "review_count",
                "target_theme_count",
                "target_theme_share",
                "theme_share_lift",
                "target_dissatisfaction",
                "target_late_rate",
            ]
        ]
        .head(15)
        .to_string(
            index=False,
            formatters={
                "target_theme_share": (
                    "{:.2%}".format
                ),

                "theme_share_lift": (
                    "{:.2f}x".format
                ),

                "target_dissatisfaction": (
                    "{:.2%}".format
                ),

                "target_late_rate": (
                    "{:.2%}".format
                ),
            },
        )
    )

else:

    print(
        "No geography met the minimum "
        "target-theme volume."
    )


# ============================================================
# DISPLAY ORDER STATUS
# ============================================================

print("\n" + "=" * 70)
print("ORDER STATUS")
print("=" * 70)


print(
    status_df.to_string(
        index=False,
        formatters={
            "target_theme_share": (
                "{:.2%}".format
            ),

            "theme_share_lift": (
                "{:.2f}x".format
            ),

            "target_dissatisfaction": (
                "{:.2%}".format
            ),

            "target_late_rate": (
                "{:.2%}".format
            ),
        },
    )
)


# ============================================================
# DISPLAY DELIVERY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DELIVERY SIGNAL")
print("=" * 70)


print(
    delivery_summary.to_string(
        index=False,
        formatters={
            "mean_delivery_delay_days": (
                "{:.2f}".format
            ),

            "median_delivery_delay_days": (
                "{:.2f}".format
            ),

            "late_delivery_rate": (
                "{:.2%}".format
            ),
        },
    )
)


# ============================================================
# DISPLAY STATUS/THEME CROSS-CHECK
# ============================================================

print("\n" + "=" * 70)
print("ORDER STATUS × THEME CROSS-CHECK")
print("=" * 70)


print(
    status_target.to_string(
        index=False,
        formatters={
            "dissatisfaction_rate": (
                "{:.2%}".format
            ),

            "late_delivery_rate": (
                "{:.2%}".format
            ),
        },
    )
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ROOT-CAUSE INVESTIGATION COMPLETE")
print("=" * 70)

print(
    "\nInterpretation boundary:"
)

print(
    "These outputs identify observational "
    "concentrations and associations."
)

print(
    "They do NOT establish operational "
    "causality."
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)