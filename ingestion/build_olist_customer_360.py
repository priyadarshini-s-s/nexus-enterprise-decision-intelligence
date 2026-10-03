from pathlib import Path
import pandas as pd


SILVER_DIR = Path("data/silver/olist")
GOLD_DIR = Path("data/gold/olist")


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(SILVER_DIR / name)


def build_customer_identity(customers):
    """
    Grain:
        one row per customer_unique_id
    """

    return (
        customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
            ]
        ]
        .groupby("customer_unique_id", as_index=False)
        .agg(
            customer_records=("customer_id", "nunique"),
            customer_zip_code_prefix=(
                "customer_zip_code_prefix",
                "first",
            ),
            customer_city=("customer_city", "first"),
            customer_state=("customer_state", "first"),
        )
    )


def build_order_metrics(customers, orders):
    customer_orders = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    return (
        customer_orders
        .groupby("customer_unique_id", as_index=False)
        .agg(
            total_orders=("order_id", "nunique"),

            delivered_orders=(
                "order_status",
                lambda x: (x == "delivered").sum(),
            ),

            cancelled_orders=(
                "order_status",
                lambda x: x.isin(
                    ["canceled", "unavailable"]
                ).sum(),
            ),

            first_order_date=(
                "order_purchase_timestamp",
                "min",
            ),

            last_order_date=(
                "order_purchase_timestamp",
                "max",
            ),
        )
    )


def build_payment_metrics(customers, orders, payments):
    customer_orders = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    customer_payments = payments.merge(
        customer_orders[
            [
                "order_id",
                "customer_unique_id",
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    return (
        customer_payments
        .groupby("customer_unique_id", as_index=False)
        .agg(
            total_payment_value=(
                "payment_value",
                "sum",
            ),

            total_payment_records=(
                "payment_sequential",
                "count",
            ),
        )
    )


def build_product_metrics(
    customers,
    orders,
    order_items,
    products,
):
    customer_orders = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    customer_items = order_items.merge(
        customer_orders[
            [
                "order_id",
                "customer_unique_id",
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    customer_items = customer_items.merge(
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

    return (
        customer_items
        .groupby("customer_unique_id", as_index=False)
        .agg(
            total_items=("product_id", "count"),
            unique_products=("product_id", "nunique"),
            unique_categories=(
                "product_category_name",
                "nunique",
            ),
            unique_sellers=(
                "seller_id",
                "nunique",
            ),
        )
    )


def build_review_metrics(customers, orders, reviews):
    customer_orders = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    customer_reviews = reviews.merge(
        customer_orders[
            [
                "order_id",
                "customer_unique_id",
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    return (
        customer_reviews
        .groupby("customer_unique_id", as_index=False)
        .agg(
            review_count=("review_id", "nunique"),

            average_review_score=(
                "review_score",
                "mean",
            ),

            low_review_count=(
                "review_score",
                lambda x: (x <= 2).sum(),
            ),
        )
    )


def build_customer_360():

    print("=" * 80)
    print("NEXUS — Customer 360 Gold Pipeline")
    print("=" * 80)

    customers = load("customers.parquet")
    orders = load("orders.parquet")
    payments = load("order_payments.parquet")
    order_items = load("order_items.parquet")
    products = load("products.parquet")
    reviews = load("order_reviews.parquet")

    print("\nBuilding customer identity...")
    identity = build_customer_identity(customers)

    print("Building order metrics...")
    order_metrics = build_order_metrics(
        customers,
        orders,
    )

    print("Building payment metrics...")
    payment_metrics = build_payment_metrics(
        customers,
        orders,
        payments,
    )

    print("Building product metrics...")
    product_metrics = build_product_metrics(
        customers,
        orders,
        order_items,
        products,
    )

    print("Building review metrics...")
    review_metrics = build_review_metrics(
        customers,
        orders,
        reviews,
    )

    print("\nAssembling customer_360...")

    customer_360 = identity.merge(
        order_metrics,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    customer_360 = customer_360.merge(
        payment_metrics,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    customer_360 = customer_360.merge(
        product_metrics,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    customer_360 = customer_360.merge(
        review_metrics,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Observation-status fields
    # ---------------------------------------------------------------

    customer_360["payment_observation_status"] = (
        customer_360["total_payment_value"]
        .notna()
        .map({
            True: "observed",
            False: "not_observed",
        })
    )

    customer_360["product_observation_status"] = (
        customer_360["total_items"]
        .notna()
        .map({
            True: "observed",
            False: "not_observed",
        })
    )

    customer_360["review_observation_status"] = (
        customer_360["review_count"]
        .notna()
        .map({
            True: "observed",
            False: "not_observed",
        })
    )

    # ---------------------------------------------------------------
    # Derived metrics
    # ---------------------------------------------------------------

    customer_360["is_repeat_customer"] = (
        customer_360["total_orders"] > 1
    )

    customer_360["customer_lifetime_days"] = (
        customer_360["last_order_date"]
        - customer_360["first_order_date"]
    ).dt.days

    customer_360["average_order_value"] = (
        customer_360["total_payment_value"]
        / customer_360["total_orders"]
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    expected_customer_count = customers[
        "customer_unique_id"
    ].nunique()

    actual_customer_count = customer_360[
        "customer_unique_id"
    ].nunique()

    if expected_customer_count != actual_customer_count:
        raise ValueError(
            "Customer identity reconciliation failed: "
            f"expected {expected_customer_count:,}, "
            f"got {actual_customer_count:,}"
        )

    if customer_360[
        "customer_unique_id"
    ].duplicated().any():
        raise ValueError(
            "customer_360 contains duplicate "
            "customer_unique_id values"
        )

    # ---------------------------------------------------------------
    # Write
    # ---------------------------------------------------------------

    GOLD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = GOLD_DIR / "customer_360.parquet"

    customer_360.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CUSTOMER 360 VALIDATION: PASS")
    print("=" * 80)

    print(
        f"Customers: {len(customer_360):,}"
    )

    print(
        f"Columns: {len(customer_360.columns)}"
    )

    print(
        "Payment observations: "
        f"{(customer_360['payment_observation_status'] == 'observed').sum():,}"
    )

    print(
        "Product observations: "
        f"{(customer_360['product_observation_status'] == 'observed').sum():,}"
    )

    print(
        "Review observations: "
        f"{(customer_360['review_observation_status'] == 'observed').sum():,}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    build_customer_360()