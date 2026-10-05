from pathlib import Path
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORDERS_PATH = PROJECT_ROOT / "data" / "silver" / "olist_spark" / "orders"


def fail(message: str):
    print(f"FAIL: {message}")
    return False


def main():
    print("=" * 70)
    print("NEXUS — Olist Orders Data Contract Validation")
    print("=" * 70)

    spark = (
        SparkSession.builder
        .appName("NEXUS-Olist-Orders-Contract-Validation")
        .master("local[2]")
        .getOrCreate()
    )

    try:
        df = spark.read.parquet(str(ORDERS_PATH))

        total_rows = df.count()

        print(f"Dataset: {ORDERS_PATH}")
        print(f"Rows: {total_rows}")
        print()

        failures = []

        # ------------------------------------------------------------
        # ORD-001: order_id uniqueness
        # ------------------------------------------------------------
        duplicate_order_ids = (
            df.groupBy("order_id")
            .count()
            .filter(col("count") > 1)
            .count()
        )

        print(f"ORD-001 | order_id unique: duplicates={duplicate_order_ids}")

        if duplicate_order_ids != 0:
            failures.append("ORD-001")

        # ------------------------------------------------------------
        # ORD-002: order_id not null
        # ------------------------------------------------------------
        null_order_ids = df.filter(col("order_id").isNull()).count()

        print(f"ORD-002 | order_id not null: nulls={null_order_ids}")

        if null_order_ids != 0:
            failures.append("ORD-002")

        # ------------------------------------------------------------
        # ORD-003: customer_id not null
        # ------------------------------------------------------------
        null_customer_ids = df.filter(col("customer_id").isNull()).count()

        print(f"ORD-003 | customer_id not null: nulls={null_customer_ids}")

        if null_customer_ids != 0:
            failures.append("ORD-003")

        # ------------------------------------------------------------
        # ORD-004: purchase timestamp not null
        # ------------------------------------------------------------
        null_purchase_ts = (
            df.filter(col("order_purchase_timestamp").isNull())
            .count()
        )

        print(
            f"ORD-004 | order_purchase_timestamp not null: "
            f"nulls={null_purchase_ts}"
        )

        if null_purchase_ts != 0:
            failures.append("ORD-004")

        # ------------------------------------------------------------
        # ORD-005: order status domain
        # ------------------------------------------------------------
        allowed_statuses = {
            "delivered",
            "shipped",
            "canceled",
            "unavailable",
            "invoiced",
            "processing",
            "created",
            "approved",
        }

        actual_statuses = {
            row["order_status"]
            for row in df.select("order_status").distinct().collect()
        }

        invalid_statuses = sorted(actual_statuses - allowed_statuses)

        print(
            f"ORD-005 | order_status domain: "
            f"invalid={invalid_statuses}"
        )

        if invalid_statuses:
            failures.append("ORD-005")

        # ------------------------------------------------------------
        # ORD-006: is_late_delivery must be 0 or 1
        # ------------------------------------------------------------
        invalid_late_flags = (
            df.filter(
                ~col("is_late_delivery").isin([0, 1])
            )
            .count()
        )

        print(
            f"ORD-006 | is_late_delivery domain: "
            f"invalid_rows={invalid_late_flags}"
        )

        if invalid_late_flags != 0:
            failures.append("ORD-006")

        # ------------------------------------------------------------
        # ORD-007: delivery cannot precede purchase
        # ------------------------------------------------------------
        invalid_delivery_dates = (
            df.filter(
                col("order_delivered_customer_date").isNotNull()
                & (
                    col("order_delivered_customer_date")
                    < col("order_purchase_timestamp")
                )
            )
            .count()
        )

        print(
            f"ORD-007 | delivered date >= purchase date: "
            f"violations={invalid_delivery_dates}"
        )

        if invalid_delivery_dates != 0:
            failures.append("ORD-007")

        # ------------------------------------------------------------
        # ORD-008: delivered + missing timestamp
        #
        # This is not a failure. We explicitly measure it.
        # ------------------------------------------------------------
        delivered_missing_timestamp = (
            df.filter(
                (col("order_status") == "delivered")
                & col("order_delivered_customer_date").isNull()
            )
            .count()
        )

        print(
            f"ORD-008 | delivered with missing customer timestamp: "
            f"observations={delivered_missing_timestamp}"
        )

        print()

        # ------------------------------------------------------------
        # Reconciliation
        # ------------------------------------------------------------
        expected_rows = 99441

        print(
            f"RECON | expected order population: "
            f"{expected_rows}"
        )
        print(
            f"RECON | actual order population:   "
            f"{total_rows}"
        )

        if total_rows != expected_rows:
            failures.append("RECON")

        # ------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------
        print()
        print("=" * 70)

        if failures:
            print("CONTRACT VALIDATION: FAILED")
            print(f"Failed rules: {', '.join(failures)}")
            return 1

        print("CONTRACT VALIDATION: PASSED")
        print("All critical contract rules passed.")
        print("=" * 70)

        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())