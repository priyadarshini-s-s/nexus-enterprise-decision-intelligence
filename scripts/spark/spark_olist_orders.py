from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
)


# ============================================================
# NEXUS — PySpark Olist Orders Silver Pipeline
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "olist"
    / "olist_orders_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "olist_spark"
    / "orders"
)


# ============================================================
# 1. SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("NEXUS-Olist-Orders-Silver")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. EXPLICIT SCHEMA
# ============================================================

orders_schema = StructType(
    [
        StructField(
            "order_id",
            StringType(),
            False,
        ),
        StructField(
            "customer_id",
            StringType(),
            False,
        ),
        StructField(
            "order_status",
            StringType(),
            False,
        ),
        StructField(
            "order_purchase_timestamp",
            StringType(),
            True,
        ),
        StructField(
            "order_approved_at",
            StringType(),
            True,
        ),
        StructField(
            "order_delivered_carrier_date",
            StringType(),
            True,
        ),
        StructField(
            "order_delivered_customer_date",
            StringType(),
            True,
        ),
        StructField(
            "order_estimated_delivery_date",
            StringType(),
            True,
        ),
    ]
)


# ============================================================
# 3. READ BRONZE DATA
# ============================================================

print("\nReading Bronze orders...")

orders_raw = (
    spark.read
    .option("header", True)
    .schema(orders_schema)
    .csv(str(INPUT_PATH))
)


# ============================================================
# 4. TYPE CONVERSION
# ============================================================

timestamp_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


orders = orders_raw

for column in timestamp_columns:
    orders = orders.withColumn(
        column,
        F.to_timestamp(
            F.col(column)
        )
    )


# ============================================================
# 5. DATA STANDARDIZATION
# ============================================================

orders = (
    orders
    .withColumn(
        "order_status",
        F.lower(
            F.trim(
                F.col("order_status")
            )
        )
    )
)


# ============================================================
# 6. DERIVED BUSINESS FEATURES
# ============================================================

orders = (
    orders
    .withColumn(
        "delivery_delay_days",
        F.when(
            F.col(
                "order_delivered_customer_date"
            ).isNotNull()
            &
            F.col(
                "order_estimated_delivery_date"
            ).isNotNull(),

            F.datediff(
                F.to_date(
                    F.col(
                        "order_delivered_customer_date"
                    )
                ),
                F.to_date(
                    F.col(
                        "order_estimated_delivery_date"
                    )
                ),
            )
        )
    )
    .withColumn(
        "is_late_delivery",
        F.when(
            F.col("delivery_delay_days") > 0,
            F.lit(1),
        )
        .otherwise(F.lit(0))
    )
)


# ============================================================
# 7. DATA QUALITY CHECKS
# ============================================================

print("\nRunning data-quality checks...")

row_count = orders.count()

print(
    f"Row count: {row_count:,}"
)

if row_count != 99_441:
    raise ValueError(
        f"Unexpected row count: {row_count}"
    )


duplicate_order_ids = (
    orders
    .groupBy("order_id")
    .count()
    .filter(
        F.col("count") > 1
    )
    .count()
)

print(
    f"Duplicate order IDs: "
    f"{duplicate_order_ids}"
)

if duplicate_order_ids != 0:
    raise ValueError(
        "Duplicate order_id values detected."
    )


null_order_ids = (
    orders
    .filter(
        F.col("order_id").isNull()
    )
    .count()
)

print(
    f"Null order IDs: {null_order_ids}"
)

if null_order_ids != 0:
    raise ValueError(
        "Null order_id values detected."
    )


# ============================================================
# 8. PARTITION INSPECTION
# ============================================================

partition_count = (
    orders
    .rdd
    .getNumPartitions()
)

print(
    f"Spark partitions: "
    f"{partition_count}"
)


# ============================================================
# 9. DATA PROFILE
# ============================================================

print("\nOrder status distribution:")

(
    orders
    .groupBy("order_status")
    .count()
    .orderBy(
        F.desc("count")
    )
    .show()
)


print("\nDelivery statistics:")

(
    orders
    .select(
        F.count("*").alias(
            "orders"
        ),
        F.sum(
            "is_late_delivery"
        ).alias(
            "late_orders"
        ),
        F.avg(
            "delivery_delay_days"
        ).alias(
            "mean_delay_days"
        ),
    )
    .show()
)


# ============================================================
# 10. WRITE SILVER PARQUET
# ============================================================

print(
    "\nWriting Spark Silver dataset..."
)

(
    orders
    .write
    .mode("overwrite")
    .parquet(
        str(OUTPUT_PATH)
    )
)


print(
    f"Written to:\n{OUTPUT_PATH}"
)


# ============================================================
# 11. CLEAN SHUTDOWN
# ============================================================

spark.stop()

print(
    "\nNEXUS PySpark orders pipeline "
    "completed successfully."
)