from pathlib import Path

from pyspark.sql import SparkSession


# ---------------------------------------------------------
# NEXUS - Apache Iceberg Snapshot + Time Travel Demo
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAREHOUSE = PROJECT_ROOT / "data" / "iceberg"
SOURCE = PROJECT_ROOT / "data" / "silver" / "olist_spark" / "orders"

CATALOG_NAME = "nexus"
TABLE_NAME = "nexus.olist.orders"


def build_spark():
    return (
        SparkSession.builder
        .appName("NEXUS-Iceberg-Snapshot-TimeTravel")
        .master("local[2]")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.12.0",
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(
            f"spark.sql.catalog.{CATALOG_NAME}",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config(
            f"spark.sql.catalog.{CATALOG_NAME}.type",
            "hadoop",
        )
        .config(
            f"spark.sql.catalog.{CATALOG_NAME}.warehouse",
            str(WAREHOUSE),
        )
        .getOrCreate()
    )


def main():
    spark = build_spark()

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 70)
    print("NEXUS - ICEBERG SNAPSHOT + TIME TRAVEL DEMO")
    print("=" * 70)

    print(f"Spark version : {spark.version}")
    print(f"Warehouse     : {WAREHOUSE}")
    print(f"Source        : {SOURCE}")
    print()

    # -----------------------------------------------------
    # 1. Read existing Silver Parquet
    # -----------------------------------------------------

    source_df = spark.read.parquet(str(SOURCE))

    source_count = source_df.count()

    print(f"Silver source row count : {source_count:,}")

    # -----------------------------------------------------
    # 2. Ensure namespace exists
    # -----------------------------------------------------

    spark.sql(
        "CREATE NAMESPACE IF NOT EXISTS nexus.olist"
    )

    # -----------------------------------------------------
    # 3. Create Iceberg table if it does not exist
    # -----------------------------------------------------

    table_exists = (
        spark.catalog.tableExists(TABLE_NAME)
    )

    if not table_exists:
        print("\nCreating Iceberg table...")

        source_df.writeTo(TABLE_NAME).using("iceberg").create()

        print("Iceberg table created.")

    else:
        print("\nIceberg table already exists.")

    # -----------------------------------------------------
    # 4. Capture snapshots BEFORE second write
    # -----------------------------------------------------

    print("\n--- Snapshots BEFORE second write ---")

    spark.sql(
        f"""
        SELECT
            committed_at,
            snapshot_id,
            operation,
            summary
        FROM {TABLE_NAME}.snapshots
        ORDER BY committed_at
        """
    ).show(truncate=False)

    # -----------------------------------------------------
    # 5. Create a SECOND genuine Iceberg snapshot
    #
    # Same real Olist Silver data.
    # No synthetic business records are introduced.
    # -----------------------------------------------------

    print("\nCreating second Iceberg snapshot...")

    (
        source_df
        .writeTo(TABLE_NAME)
        .overwritePartitions()
    )

    print("Second snapshot created.")

    # -----------------------------------------------------
    # 6. Current table validation
    # -----------------------------------------------------

    current_count = spark.table(TABLE_NAME).count()

    print("\n--- Current Iceberg table ---")
    print(f"Current row count : {current_count:,}")

    # -----------------------------------------------------
    # 7. Snapshot history
    # -----------------------------------------------------

    print("\n--- Iceberg Snapshot History ---")

    snapshots = spark.sql(
        f"""
        SELECT
            committed_at,
            snapshot_id,
            parent_id,
            operation,
            summary
        FROM {TABLE_NAME}.snapshots
        ORDER BY committed_at
        """
    )

    snapshots.show(truncate=False)

    # -----------------------------------------------------
    # 8. Capture snapshot IDs
    # -----------------------------------------------------

    snapshot_rows = snapshots.collect()

    print(f"Number of snapshots : {len(snapshot_rows)}")

    if len(snapshot_rows) >= 2:

        first_snapshot = snapshot_rows[0]
        latest_snapshot = snapshot_rows[-1]

        first_snapshot_id = first_snapshot["snapshot_id"]
        latest_snapshot_id = latest_snapshot["snapshot_id"]

        print()
        print(f"First snapshot ID  : {first_snapshot_id}")
        print(f"Latest snapshot ID : {latest_snapshot_id}")

        # -------------------------------------------------
        # 9. Time travel to FIRST snapshot
        # -------------------------------------------------

        print("\n--- Time Travel: FIRST Snapshot ---")

        historical_df = spark.sql(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            VERSION AS OF {first_snapshot_id}
            """
        )

        historical_count = historical_df.count()

        print(
            f"Historical row count : {historical_count:,}"
        )

        historical_df.show(5, truncate=False)

        # -------------------------------------------------
        # 10. Compare historical and current versions
        # -------------------------------------------------

        print("\n--- Snapshot Comparison ---")

        print(
            f"First snapshot rows  : {historical_count:,}"
        )

        print(
            f"Latest snapshot rows : {current_count:,}"
        )

        print(
            f"Row-count difference : "
            f"{current_count - historical_count:,}"
        )

    else:
        print(
            "\nWARNING: Fewer than two snapshots found."
        )

    # -----------------------------------------------------
    # 11. Table metadata
    # -----------------------------------------------------

    print("\n--- Iceberg Table Properties ---")

    spark.sql(
        f"SHOW TBLPROPERTIES {TABLE_NAME}"
    ).show(truncate=False)

    print("\n" + "=" * 70)
    print("ICEBERG SNAPSHOT + TIME TRAVEL DEMO COMPLETE")
    print("=" * 70)

    spark.stop()


if __name__ == "__main__":
    main()