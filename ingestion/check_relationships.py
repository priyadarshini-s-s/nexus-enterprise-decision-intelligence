from pathlib import Path
import pandas as pd


BRONZE_DIR = Path("data/bronze/olist")


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(BRONZE_DIR / name)


def check_fk(
    child_df: pd.DataFrame,
    child_column: str,
    parent_df: pd.DataFrame,
    parent_column: str,
):
    child_values = set(child_df[child_column].dropna().unique())
    parent_values = set(parent_df[parent_column].dropna().unique())

    orphan_values = child_values - parent_values

    print(f"\n{child_column} → {parent_column}")
    print("-" * 70)
    print(f"Child unique values : {len(child_values):,}")
    print(f"Parent unique values: {len(parent_values):,}")
    print(f"Orphan values       : {len(orphan_values):,}")

    if orphan_values:
        print("Sample orphan values:")
        for value in list(orphan_values)[:10]:
            print(f"  {value}")
    else:
        print("Status: PASS")


def main():

    customers = load("olist_customers_dataset.csv")
    orders = load("olist_orders_dataset.csv")
    items = load("olist_order_items_dataset.csv")
    payments = load("olist_order_payments_dataset.csv")
    reviews = load("olist_order_reviews_dataset.csv")
    products = load("olist_products_dataset.csv")
    sellers = load("olist_sellers_dataset.csv")

    print("=" * 80)
    print("NEXUS — Olist Referential Integrity Checks")
    print("=" * 80)

    check_fk(
        orders,
        "customer_id",
        customers,
        "customer_id",
    )

    check_fk(
        items,
        "order_id",
        orders,
        "order_id",
    )

    check_fk(
        items,
        "product_id",
        products,
        "product_id",
    )

    check_fk(
        items,
        "seller_id",
        sellers,
        "seller_id",
    )

    check_fk(
        payments,
        "order_id",
        orders,
        "order_id",
    )

    check_fk(
        reviews,
        "order_id",
        orders,
        "order_id",
    )


if __name__ == "__main__":
    main()