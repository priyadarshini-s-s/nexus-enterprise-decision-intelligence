from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    IntegerType,
)


# ============================================================
# NEXUS — Module 4 PySpark Verification
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORDERS_PATH = PROJECT_ROOT / "data" / "silver" / "olist_spark" / "orders"
CUSTOMERS_PATH = PROJECT_ROOT / "data" / "silver" / "olist" / "customers.parquet"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "spark_module4_customer_orders"
)


def create_spark_session():
    return (
        SparkSession.builder
        .appName("NEXUS-Module4-Verification")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def load_data(spark):

    orders_schema = StructType([
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField(
            "order_purchase_timestamp",
            TimestampType(),
            True
        ),
        StructField(
            "order_approved_at",
            TimestampType(),
            True
        ),
        StructField(
            "order_delivered_carrier_date",
            TimestampType(),
            True
        ),
        StructField(
            "order_delivered_customer_date",
            TimestampType(),
            True
        ),
        StructField(
            "order_estimated_delivery_date",
            TimestampType(),
            True
        ),
        StructField(
            "delivery_delay_days",
            IntegerType(),
            True
        ),
        StructField(
            "is_late_delivery",
            IntegerType(),
            True
        ),
    ])

    orders = (
        spark.read
        .schema(orders_schema)
        .parquet(str(ORDERS_PATH))
    )

    customers = (
        spark.read
        .parquet(str(CUSTOMERS_PATH))
    )

    return orders, customers


def verify_basic_data(orders, customers):

    print("\n" + "=" * 70)
    print("1. BASIC DATAFRAME VERIFICATION")
    print("=" * 70)

    print(f"Orders rows     : {orders.count():,}")
    print(f"Customers rows  : {customers.count():,}")

    print("\nOrders schema:")
    orders.printSchema()

    print("\nCustomers schema:")
    customers.printSchema()

    print(f"\nOrders partitions: {orders.rdd.getNumPartitions():,}")
    print(f"Customer partitions: {customers.rdd.getNumPartitions():,}")


def perform_join(orders, customers):

    print("\n" + "=" * 70)
    print("2. JOIN VERIFICATION")
    print("=" * 70)

    customer_dimension = customers.select(
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
    )

    joined = (
        orders.alias("o")
        .join(
            customer_dimension.alias("c"),
            F.col("o.customer_id") == F.col("c.customer_id"),
            "left"
        )
        .select(
            F.col("o.order_id"),
            F.col("o.customer_id"),
            F.col("c.customer_unique_id"),
            F.col("c.customer_city"),
            F.col("c.customer_state"),
            F.col("o.order_status"),
            F.col("o.order_purchase_timestamp"),
            F.col("o.delivery_delay_days"),
            F.col("o.is_late_delivery"),
        )
    )

    total_orders = orders.count()
    joined_orders = joined.count()
    unmatched = joined.filter(
        F.col("customer_unique_id").isNull()
    ).count()

    print(f"Source orders        : {total_orders:,}")
    print(f"Joined rows          : {joined_orders:,}")
    print(f"Unmatched customers  : {unmatched:,}")

    assert joined_orders == total_orders, \
        "LEFT JOIN changed the order population."

    return joined


def perform_aggregation(joined):

    print("\n" + "=" * 70)
    print("3. AGGREGATION VERIFICATION")
    print("=" * 70)

    customer_summary = (
        joined
        .groupBy("customer_unique_id")
        .agg(
            F.countDistinct("order_id").alias("total_orders"),
            F.sum(
                F.when(
                    F.col("order_status") == "delivered",
                    1
                ).otherwise(0)
            ).alias("delivered_orders"),
            F.sum(
                F.when(
                    F.col("is_late_delivery") == 1,
                    1
                ).otherwise(0)
            ).alias("late_orders"),
            F.avg("delivery_delay_days").alias(
                "avg_delivery_delay_days"
            ),
        )
    )

    print(
        f"Customer summary rows: "
        f"{customer_summary.count():,}"
    )

    customer_summary.orderBy(
        F.desc("total_orders")
    ).show(10, truncate=False)

    return customer_summary


def perform_window_function(joined):

    print("\n" + "=" * 70)
    print("4. WINDOW FUNCTION VERIFICATION")
    print("=" * 70)

    window_spec = (
        Window
        .partitionBy("customer_unique_id")
        .orderBy(
            F.col("order_purchase_timestamp").asc(),
            F.col("order_id").asc()
        )
    )

    ordered = (
        joined
        .withColumn(
            "customer_order_number",
            F.row_number().over(window_spec)
        )
        .withColumn(
            "previous_order_date",
            F.lag(
                "order_purchase_timestamp"
            ).over(window_spec)
        )
    )

    print("Window functions used:")
    print("  - row_number()")
    print("  - lag()")

    ordered.select(
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp",
        "customer_order_number",
        "previous_order_date",
    ).orderBy(
        "customer_unique_id",
        "customer_order_number"
    ).show(10, truncate=False)

    return ordered


def perform_spark_sql(joined):

    print("\n" + "=" * 70)
    print("5. SPARK SQL VERIFICATION")
    print("=" * 70)

    joined.createOrReplaceTempView(
        "nexus_customer_orders"
    )

    sql_result = spark.sql("""
        SELECT
            customer_state,
            COUNT(DISTINCT customer_unique_id)
                AS unique_customers,
            COUNT(DISTINCT order_id)
                AS total_orders,
            SUM(
                CASE
                    WHEN is_late_delivery = 1
                    THEN 1
                    ELSE 0
                END
            ) AS late_orders
        FROM nexus_customer_orders
        GROUP BY customer_state
        ORDER BY total_orders DESC
    """)

    sql_result.show(10, truncate=False)

    return sql_result


def verify_partitioning(joined):

    print("\n" + "=" * 70)
    print("6. PARTITIONING VERIFICATION")
    print("=" * 70)

    before = joined.rdd.getNumPartitions()

    partitioned = joined.repartition(
        8,
        "customer_state"
    )

    after = partitioned.rdd.getNumPartitions()

    print(f"Partitions before repartition : {before}")
    print(f"Partitions after repartition  : {after}")
    print("Partition key                : customer_state")

    return partitioned


def reconcile(joined, customer_summary):

    print("\n" + "=" * 70)
    print("7. RECONCILIATION")
    print("=" * 70)

    source_orders = joined.count()

    customer_order_sum = (
        customer_summary
        .agg(F.sum("total_orders"))
        .collect()[0][0]
    )

    print(f"Joined order population : {source_orders:,}")
    print(
        f"Customer-level order sum: "
        f"{customer_order_sum:,}"
    )

    assert source_orders == customer_order_sum, (
        "Reconciliation failed: customer aggregation "
        "does not preserve order population."
    )

    print("RECONCILIATION: PASS")


def write_output(partitioned):

    print("\n" + "=" * 70)
    print("8. REUSABLE JOB OUTPUT")
    print("=" * 70)

    (
        partitioned
        .write
        .mode("overwrite")
        .partitionBy("customer_state")
        .parquet(str(OUTPUT_PATH))
    )

    print(f"Output: {OUTPUT_PATH}")


def main():

    global spark

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    start = time.perf_counter()

    try:

        print("=" * 70)
        print("NEXUS — PYSPARK MODULE 4 VERIFICATION")
        print("=" * 70)

        orders, customers = load_data(spark)

        verify_basic_data(
            orders,
            customers
        )

        joined = perform_join(
            orders,
            customers
        )

        customer_summary = perform_aggregation(
            joined
        )

        ordered = perform_window_function(
            joined
        )

        sql_result = perform_spark_sql(
            joined
        )

        partitioned = verify_partitioning(
            joined
        )

        reconcile(
            joined,
            customer_summary
        )

        write_output(
            partitioned
        )

        elapsed = time.perf_counter() - start

        print("\n" + "=" * 70)
        print("MODULE 4 VERIFICATION COMPLETE")
        print("=" * 70)
        print(f"Execution time: {elapsed:.2f} seconds")
        print("Join             : PASS")
        print("Aggregation      : PASS")
        print("Window functions : PASS")
        print("Spark SQL        : PASS")
        print("Partitioning     : PASS")
        print("Reconciliation   : PASS")
        print("Reusable output  : PASS")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()