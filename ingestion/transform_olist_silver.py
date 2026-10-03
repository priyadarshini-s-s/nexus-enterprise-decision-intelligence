from pathlib import Path
import pandas as pd


BRONZE_DIR = Path("data/bronze/olist")
SILVER_DIR = Path("data/silver/olist")


def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(BRONZE_DIR / filename)


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def write_parquet(df: pd.DataFrame, filename: str):
    output_path = SILVER_DIR / filename
    df.to_parquet(output_path, index=False)
    return output_path


def validate_required(df, columns, table_name):
    for column in columns:
        null_count = df[column].isna().sum()

        if null_count > 0:
            raise ValueError(
                f"{table_name}: {column} contains "
                f"{null_count:,} null values"
            )


def validate_unique(df, columns, table_name):
    duplicate_count = df.duplicated(subset=columns).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{table_name}: composite key {columns} contains "
            f"{duplicate_count:,} duplicate rows"
        )


def transform_customers():
    print("[1/9] customers")

    df = load_csv("olist_customers_dataset.csv")

    df["customer_id"] = df["customer_id"].astype("string")
    df["customer_unique_id"] = df["customer_unique_id"].astype("string")
    df["customer_zip_code_prefix"] = (
        df["customer_zip_code_prefix"].astype("Int64")
    )
    df["customer_city"] = clean_text(df["customer_city"])
    df["customer_state"] = clean_text(df["customer_state"]).str.upper()

    validate_required(
        df,
        ["customer_id", "customer_unique_id"],
        "customers"
    )

    validate_unique(
        df,
        ["customer_id"],
        "customers"
    )

    return df


def transform_orders():
    print("[2/9] orders")

    df = load_csv("olist_orders_dataset.csv")

    df["order_id"] = df["order_id"].astype("string")
    df["customer_id"] = df["customer_id"].astype("string")
    df["order_status"] = clean_text(df["order_status"]).str.lower()

    timestamp_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in timestamp_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    validate_required(
        df,
        ["order_id", "customer_id", "order_purchase_timestamp"],
        "orders"
    )

    validate_unique(
        df,
        ["order_id"],
        "orders"
    )

    return df


def transform_order_items():
    print("[3/9] order_items")

    df = load_csv("olist_order_items_dataset.csv")

    df["order_id"] = df["order_id"].astype("string")
    df["product_id"] = df["product_id"].astype("string")
    df["seller_id"] = df["seller_id"].astype("string")
    df["order_item_id"] = df["order_item_id"].astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["freight_value"] = pd.to_numeric(
        df["freight_value"],
        errors="coerce"
    )
    df["shipping_limit_date"] = pd.to_datetime(
        df["shipping_limit_date"],
        errors="coerce"
    )

    validate_required(
        df,
        ["order_id", "order_item_id", "product_id", "seller_id"],
        "order_items"
    )

    validate_unique(
        df,
        ["order_id", "order_item_id"],
        "order_items"
    )

    return df


def transform_payments():
    print("[4/9] order_payments")

    df = load_csv("olist_order_payments_dataset.csv")

    df["order_id"] = df["order_id"].astype("string")
    df["payment_sequential"] = df["payment_sequential"].astype("Int64")
    df["payment_type"] = clean_text(df["payment_type"]).str.lower()
    df["payment_installments"] = (
        df["payment_installments"].astype("Int64")
    )
    df["payment_value"] = pd.to_numeric(
        df["payment_value"],
        errors="coerce"
    )

    validate_required(
        df,
        ["order_id", "payment_sequential", "payment_type"],
        "order_payments"
    )

    validate_unique(
        df,
        ["order_id", "payment_sequential"],
        "order_payments"
    )

    return df


def transform_reviews():
    print("[5/9] order_reviews")

    df = load_csv("olist_order_reviews_dataset.csv")

    df["review_id"] = df["review_id"].astype("string")
    df["order_id"] = df["order_id"].astype("string")
    df["review_score"] = df["review_score"].astype("Int64")

    text_columns = [
        "review_comment_title",
        "review_comment_message",
    ]

    for column in text_columns:
        df[column] = clean_text(df[column])

    df["review_creation_date"] = pd.to_datetime(
        df["review_creation_date"],
        errors="coerce"
    )

    df["review_answer_timestamp"] = pd.to_datetime(
        df["review_answer_timestamp"],
        errors="coerce"
    )

    validate_required(
        df,
        ["review_id", "order_id", "review_score"],
        "order_reviews"
    )

    return df


def transform_products():
    print("[6/9] products")

    df = load_csv("olist_products_dataset.csv")

    df["product_id"] = df["product_id"].astype("string")
    df["product_category_name"] = clean_text(
        df["product_category_name"]
    )

    numeric_columns = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    validate_required(
        df,
        ["product_id"],
        "products"
    )

    validate_unique(
        df,
        ["product_id"],
        "products"
    )

    return df


def transform_sellers():
    print("[7/9] sellers")

    df = load_csv("olist_sellers_dataset.csv")

    df["seller_id"] = df["seller_id"].astype("string")
    df["seller_zip_code_prefix"] = (
        df["seller_zip_code_prefix"].astype("Int64")
    )
    df["seller_city"] = clean_text(df["seller_city"])
    df["seller_state"] = clean_text(df["seller_state"]).str.upper()

    validate_required(
        df,
        ["seller_id"],
        "sellers"
    )

    validate_unique(
        df,
        ["seller_id"],
        "sellers"
    )

    return df


def transform_geolocation():
    print("[8/9] geolocation")

    df = load_csv("olist_geolocation_dataset.csv")

    df["geolocation_zip_code_prefix"] = (
        df["geolocation_zip_code_prefix"].astype("Int64")
    )
    df["geolocation_lat"] = pd.to_numeric(
        df["geolocation_lat"],
        errors="coerce"
    )
    df["geolocation_lng"] = pd.to_numeric(
        df["geolocation_lng"],
        errors="coerce"
    )
    df["geolocation_city"] = clean_text(
        df["geolocation_city"]
    )
    df["geolocation_state"] = clean_text(
        df["geolocation_state"]
    ).str.upper()

    validate_required(
        df,
        [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
        ],
        "geolocation"
    )

    return df


def transform_category_translation():
    print("[9/9] category_translation")

    df = load_csv(
        "product_category_name_translation.csv"
    )

    df["product_category_name"] = clean_text(
        df["product_category_name"]
    )
    df["product_category_name_english"] = clean_text(
        df["product_category_name_english"]
    )

    validate_required(
        df,
        [
            "product_category_name",
            "product_category_name_english",
        ],
        "category_translation"
    )

    validate_unique(
        df,
        ["product_category_name"],
        "category_translation"
    )

    return df


def main():

    print("=" * 80)
    print("NEXUS — Olist Silver Transformation Pipeline")
    print("=" * 80)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    transformations = {
        "customers.parquet": transform_customers(),
        "orders.parquet": transform_orders(),
        "order_items.parquet": transform_order_items(),
        "order_payments.parquet": transform_payments(),
        "order_reviews.parquet": transform_reviews(),
        "products.parquet": transform_products(),
        "sellers.parquet": transform_sellers(),
        "geolocation.parquet": transform_geolocation(),
        "category_translation.parquet": transform_category_translation(),
    }

    print("\nWriting Silver tables...")

    for filename, df in transformations.items():
        path = write_parquet(df, filename)

        print(
            f"  {filename:<32}"
            f"{len(df):>10,} rows"
            f"  → {path}"
        )

    print("\n" + "=" * 80)
    print("SILVER PIPELINE VALIDATION: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()