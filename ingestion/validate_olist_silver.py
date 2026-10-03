from pathlib import Path
import pandas as pd


SILVER_DIR = Path("data/silver/olist")


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(SILVER_DIR / name)


def check_fk(
    child_df,
    child_column,
    parent_df,
    parent_column,
    relationship_name,
):
    child_values = set(
        child_df[child_column].dropna().unique()
    )

    parent_values = set(
        parent_df[parent_column].dropna().unique()
    )

    orphan_values = child_values - parent_values

    print(f"\n{relationship_name}")
    print("-" * 70)
    print(f"Child unique values : {len(child_values):,}")
    print(f"Parent unique values: {len(parent_values):,}")
    print(f"Orphan values       : {len(orphan_values):,}")

    if orphan_values:
        print("STATUS: FAIL")

        for value in list(orphan_values)[:10]:
            print(f"  {value}")

        return False

    print("STATUS: PASS")
    return True


def check_unique(df, columns, table_name):
    duplicates = df.duplicated(
        subset=columns
    ).sum()

    print(f"\n{table_name} uniqueness")
    print("-" * 70)
    print(f"Key: {columns}")
    print(f"Duplicate rows: {duplicates:,}")

    if duplicates > 0:
        print("STATUS: FAIL")
        return False

    print("STATUS: PASS")
    return True


def main():

    print("=" * 80)
    print("NEXUS — Olist Silver Data Quality Gate")
    print("=" * 80)

    customers = load("customers.parquet")
    orders = load("orders.parquet")
    items = load("order_items.parquet")
    payments = load("order_payments.parquet")
    reviews = load("order_reviews.parquet")
    products = load("products.parquet")
    sellers = load("sellers.parquet")

    checks = []

    # ------------------------------------------------------------------
    # Primary / composite key checks
    # ------------------------------------------------------------------

    checks.append(
        check_unique(
            customers,
            ["customer_id"],
            "customers"
        )
    )

    checks.append(
        check_unique(
            orders,
            ["order_id"],
            "orders"
        )
    )

    checks.append(
        check_unique(
            items,
            ["order_id", "order_item_id"],
            "order_items"
        )
    )

    checks.append(
        check_unique(
            payments,
            ["order_id", "payment_sequential"],
            "order_payments"
        )
    )

    checks.append(
        check_unique(
            products,
            ["product_id"],
            "products"
        )
    )

    checks.append(
        check_unique(
            sellers,
            ["seller_id"],
            "sellers"
        )
    )

    # ------------------------------------------------------------------
    # Referential integrity
    # ------------------------------------------------------------------

    checks.append(
        check_fk(
            orders,
            "customer_id",
            customers,
            "customer_id",
            "orders.customer_id → customers.customer_id"
        )
    )

    checks.append(
        check_fk(
            items,
            "order_id",
            orders,
            "order_id",
            "order_items.order_id → orders.order_id"
        )
    )

    checks.append(
        check_fk(
            items,
            "product_id",
            products,
            "product_id",
            "order_items.product_id → products.product_id"
        )
    )

    checks.append(
        check_fk(
            items,
            "seller_id",
            sellers,
            "seller_id",
            "order_items.seller_id → sellers.seller_id"
        )
    )

    checks.append(
        check_fk(
            payments,
            "order_id",
            orders,
            "order_id",
            "order_payments.order_id → orders.order_id"
        )
    )

    checks.append(
        check_fk(
            reviews,
            "order_id",
            orders,
            "order_id",
            "order_reviews.order_id → orders.order_id"
        )
    )

    # ------------------------------------------------------------------
    # Row-count reconciliation
    # ------------------------------------------------------------------

    expected_counts = {
        "customers": 99_441,
        "orders": 99_441,
        "order_items": 112_650,
        "order_payments": 103_886,
        "order_reviews": 99_224,
        "products": 32_951,
        "sellers": 3_095,
    }

    loaded_tables = {
        "customers": customers,
        "orders": orders,
        "order_items": items,
        "order_payments": payments,
        "order_reviews": reviews,
        "products": products,
        "sellers": sellers,
    }

    print("\n" + "=" * 80)
    print("ROW COUNT RECONCILIATION")
    print("=" * 80)

    for table_name, expected in expected_counts.items():

        actual = len(loaded_tables[table_name])

        print(
            f"{table_name:<20}"
            f"expected={expected:>10,}  "
            f"actual={actual:>10,}",
            end=" "
        )

        if actual != expected:
            print("FAIL")
            checks.append(False)
        else:
            print("PASS")
            checks.append(True)

    # ------------------------------------------------------------------
    # Final gate
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)

    if all(checks):
        print("SILVER DATA QUALITY GATE: PASS")
        print("All uniqueness, relationship, and row-count checks passed.")
    else:
        print("SILVER DATA QUALITY GATE: FAIL")
        raise SystemExit(1)

    print("=" * 80)


if __name__ == "__main__":
    main()