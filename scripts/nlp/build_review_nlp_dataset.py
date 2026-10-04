from pathlib import Path

import pandas as pd


SILVER_DIR = Path("data/silver/olist")
OUTPUT_PATH = Path(
    "data/gold/olist/review_nlp_dataset.parquet"
)


def load_data():
    reviews = pd.read_parquet(
        SILVER_DIR / "order_reviews.parquet"
    )

    orders = pd.read_parquet(
        SILVER_DIR / "orders.parquet"
    )

    customers = pd.read_parquet(
        SILVER_DIR / "customers.parquet"
    )

    order_items = pd.read_parquet(
        SILVER_DIR / "order_items.parquet"
    )

    products = pd.read_parquet(
        SILVER_DIR / "products.parquet"
    )

    return (
        reviews,
        orders,
        customers,
        order_items,
        products,
    )


def main():

    print(
        "NEXUS — Build Review NLP Dataset"
    )
    print("=" * 70)

    (
        reviews,
        orders,
        customers,
        order_items,
        products,
    ) = load_data()

    # --------------------------------------------------------
    # Validate review grain
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Establish source observation grain
    # --------------------------------------------------------
    #
    # Olist review_id is not unique in the observed source.
    # Therefore review_id is retained as a source identifier,
    # but it is not treated as the analytical primary key.
    #
    # One row in the source review table becomes one
    # review observation in the NLP dataset.

    reviews = reviews.copy()

    reviews.insert(
        0,
        "review_observation_id",
        range(1, len(reviews) + 1),
    )

    if reviews["review_observation_id"].duplicated().any():
        raise ValueError(
            "review_observation_id is not unique."
        )

    print(
        f"Source review observations: "
        f"{len(reviews):,}"
    )

    print(
        f"Unique source review_id values: "
        f"{reviews['review_id'].nunique():,}"
    )

    print(
        f"Review observations sharing review_id: "
        f"{reviews['review_id'].duplicated().sum():,}"
    )

    print("\nSOURCE")
    print("-" * 70)

    print(
        f"Reviews: {len(reviews):,}"
    )

    print(
        f"Orders: {len(orders):,}"
    )

    print(
        f"Order items: {len(order_items):,}"
    )

    # --------------------------------------------------------
    # Build review text
    # --------------------------------------------------------

    reviews = reviews.copy()

    reviews["review_comment_title"] = (
        reviews["review_comment_title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reviews["review_comment_message"] = (
        reviews["review_comment_message"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reviews["review_text"] = (
        reviews["review_comment_title"]
        + " "
        + reviews["review_comment_message"]
    ).str.strip()

    reviews["has_review_text"] = (
        reviews["review_text"]
        .str.len()
        .gt(0)
    )

    reviews["review_text_length"] = (
        reviews["review_text"]
        .str.len()
    )

    # --------------------------------------------------------
    # Customer identity
    # --------------------------------------------------------

    customer_map = (
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ]
        .drop_duplicates(
            "customer_id"
        )
    )

    reviews = reviews.merge(
        orders[
            [
                "order_id",
                "customer_id",
                "order_purchase_timestamp",
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    reviews = reviews.merge(
        customer_map,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Aggregate order-level product context
    # --------------------------------------------------------

    item_context = (
        order_items[
            [
                "order_id",
                "product_id",
                "seller_id",
            ]
        ]
        .drop_duplicates()
    )

    order_product_counts = (
        item_context
        .groupby("order_id")
        .agg(
            product_count=(
                "product_id",
                "nunique",
            ),
            seller_count=(
                "seller_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    reviews = reviews.merge(
        order_product_counts,
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Product categories
    # --------------------------------------------------------

    product_category = (
        products[
            [
                "product_id",
                "product_category_name",
            ]
        ]
        .drop_duplicates(
            "product_id"
        )
    )

    item_categories = (
        item_context
        .merge(
            product_category,
            on="product_id",
            how="left",
        )
    )

    category_by_order = (
        item_categories
        .dropna(
            subset=[
                "product_category_name"
            ]
        )
        .groupby("order_id")[
            "product_category_name"
        ]
        .agg(
            lambda values:
            " | ".join(
                sorted(
                    set(values)
                )
            )
        )
        .reset_index(
            name="product_categories"
        )
    )

    reviews = reviews.merge(
        category_by_order,
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    reviews["review_creation_date"] = (
        pd.to_datetime(
            reviews["review_creation_date"],
            errors="coerce",
        )
    )

    reviews["order_purchase_timestamp"] = (
        pd.to_datetime(
            reviews["order_purchase_timestamp"],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # Final columns
    # --------------------------------------------------------

    final_columns = [
	"review_observation_id",
        "review_id",
        "order_id",
        "customer_unique_id",
        "review_score",
        "review_text",
        "has_review_text",
        "review_text_length",
        "review_creation_date",
        "review_answer_timestamp",
        "order_purchase_timestamp",
        "product_count",
        "seller_count",
        "product_categories",
    ]

    result = reviews[
        final_columns
    ].copy()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if result["review_observation_id"].duplicated().any():
    	raise ValueError(
        	"Review observation grain was violated."
    	)

    if result["review_id"].isna().any():
        raise ValueError(
            "Missing review_id."
        )

    if result["review_text_length"].lt(0).any():
        raise ValueError(
            "Negative text length detected."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nRESULT")
    print("-" * 70)

    print(
        f"Rows: {len(result):,}"
    )

    print(
        f"Unique reviews: "
        f"{result['review_id'].nunique():,}"
    )

    text_count = int(
        result["has_review_text"].sum()
    )

    text_rate = (
        text_count / len(result)
        * 100
    )

    print(
        f"Reviews with text: "
        f"{text_count:,}"
    )

    print(
        f"Text coverage: "
        f"{text_rate:.2f}%"
    )

    print(
        f"Median text length: "
        f"{result['review_text_length'].median():.0f}"
    )

    print(
        f"Mean review score: "
        f"{result['review_score'].mean():.4f}"
    )

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()