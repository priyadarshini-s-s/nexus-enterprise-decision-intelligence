from pathlib import Path
import argparse
import sys

import duckdb


RAW_DIR = Path("data/raw/dunnhumby")
SILVER_DIR = Path("data/silver/dunnhumby")

# SQL expressions explicitly preserve source semantics.
TABLES = {
    "campaign_desc": """
        SELECT
            trim(DESCRIPTION) AS DESCRIPTION,
            CAST(CAMPAIGN AS BIGINT) AS CAMPAIGN,
            CAST(START_DAY AS BIGINT) AS START_DAY,
            CAST(END_DAY AS BIGINT) AS END_DAY
        FROM {source}
    """,
    "campaign_table": """
        SELECT
            trim(DESCRIPTION) AS DESCRIPTION,
            CAST(household_key AS BIGINT) AS household_key,
            CAST(CAMPAIGN AS BIGINT) AS CAMPAIGN
        FROM {source}
    """,
    "causal_data": """
        SELECT
            CAST(PRODUCT_ID AS BIGINT) AS PRODUCT_ID,
            CAST(STORE_ID AS BIGINT) AS STORE_ID,
            CAST(WEEK_NO AS BIGINT) AS WEEK_NO,
            trim(display) AS display,
            trim(mailer) AS mailer
        FROM {source}
    """,
    "coupon": """
        SELECT
            CAST(COUPON_UPC AS BIGINT) AS COUPON_UPC,
            CAST(PRODUCT_ID AS BIGINT) AS PRODUCT_ID,
            CAST(CAMPAIGN AS BIGINT) AS CAMPAIGN
        FROM {source}
    """,
    "coupon_redempt": """
        SELECT
            CAST(household_key AS BIGINT) AS household_key,
            CAST(DAY AS BIGINT) AS DAY,
            CAST(COUPON_UPC AS BIGINT) AS COUPON_UPC,
            CAST(CAMPAIGN AS BIGINT) AS CAMPAIGN
        FROM {source}
    """,
    "hh_demographic": """
        SELECT
            trim(AGE_DESC) AS AGE_DESC,
            trim(MARITAL_STATUS_CODE) AS MARITAL_STATUS_CODE,
            trim(INCOME_DESC) AS INCOME_DESC,
            trim(HOMEOWNER_DESC) AS HOMEOWNER_DESC,
            trim(HH_COMP_DESC) AS HH_COMP_DESC,
            trim(HOUSEHOLD_SIZE_DESC) AS HOUSEHOLD_SIZE_DESC,
            trim(KID_CATEGORY_DESC) AS KID_CATEGORY_DESC,
            CAST(household_key AS BIGINT) AS household_key
        FROM {source}
    """,
    "product": """
        SELECT
            CAST(PRODUCT_ID AS BIGINT) AS PRODUCT_ID,
            CAST(MANUFACTURER AS BIGINT) AS MANUFACTURER,
            trim(DEPARTMENT) AS DEPARTMENT,
            trim(BRAND) AS BRAND,
            trim(COMMODITY_DESC) AS COMMODITY_DESC,
            trim(SUB_COMMODITY_DESC) AS SUB_COMMODITY_DESC,
            trim(CURR_SIZE_OF_PRODUCT) AS CURR_SIZE_OF_PRODUCT
        FROM {source}
    """,
    "transaction_data": """
        SELECT
            CAST(household_key AS BIGINT) AS household_key,
            CAST(BASKET_ID AS BIGINT) AS BASKET_ID,
            CAST(DAY AS BIGINT) AS DAY,
            CAST(PRODUCT_ID AS BIGINT) AS PRODUCT_ID,
            CAST(QUANTITY AS BIGINT) AS QUANTITY,
            CAST(SALES_VALUE AS DOUBLE) AS SALES_VALUE,
            CAST(STORE_ID AS BIGINT) AS STORE_ID,
            CAST(RETAIL_DISC AS DOUBLE) AS RETAIL_DISC,
            trim(TRANS_TIME) AS TRANS_TIME,
            CAST(WEEK_NO AS BIGINT) AS WEEK_NO,
            CAST(COUPON_DISC AS DOUBLE) AS COUPON_DISC,
            CAST(COUPON_MATCH_DISC AS DOUBLE) AS COUPON_MATCH_DISC
        FROM {source}
    """,
}

CANDIDATE_KEYS = {
    "campaign_desc": ["CAMPAIGN"],
    "campaign_table": ["household_key", "CAMPAIGN"],
    "coupon_redempt": ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"],
    "hh_demographic": ["household_key"],
    "product": ["PRODUCT_ID"],
}


def sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def transform_table(con, table_name: str) -> None:
    source_path = RAW_DIR / f"{table_name}.csv"
    output_path = SILVER_DIR / f"{table_name}.parquet"

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source: {source_path}")

    source = f"read_csv_auto({sql_path(source_path)}, sample_size=10000)"
    query = TABLES[table_name].format(source=source)

    source_rows = con.execute(
        f"SELECT COUNT(*) FROM {source}"
    ).fetchone()[0]

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # Do not overwrite an existing output without an explicit decision.
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Inspect it before deciding whether to replace it."
        )

    con.execute(
        f"COPY ({query}) TO {sql_path(output_path)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    output_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_path(output_path)})"
    ).fetchone()[0]

    if source_rows != output_rows:
        raise ValueError(
            f"{table_name}: row-count mismatch "
            f"(source={source_rows:,}, silver={output_rows:,})"
        )

    if table_name in CANDIDATE_KEYS:
        columns = CANDIDATE_KEYS[table_name]
        key_sql = ", ".join(f'"{column}"' for column in columns)

        null_condition = " OR ".join(
            f'"{column}" IS NULL' for column in columns
        )
        null_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_path(output_path)}) "
            f"WHERE {null_condition}"
        ).fetchone()[0]

        duplicate_count = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {key_sql}
                FROM read_parquet({sql_path(output_path)})
                GROUP BY {key_sql}
                HAVING COUNT(*) > 1
            ) duplicates
            """
        ).fetchone()[0]

        if null_count or duplicate_count:
            raise ValueError(
                f"{table_name}: candidate-key validation failed; "
                f"null-key rows={null_count:,}, "
                f"duplicate key groups={duplicate_count:,}"
            )

    print(
        f"PASS {table_name:<20} "
        f"source={source_rows:>10,} "
        f"silver={output_rows:>10,} "
        f"row_count_match=True"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=sorted(TABLES),
        required=True,
        help="Explicit list of tables to transform",
    )
    args = parser.parse_args()

    print(f"DuckDB version: {duckdb.__version__}")
    print("Dunnhumby Silver transformation")
    print("Raw source files will not be modified.")

    con = duckdb.connect(":memory:")
    try:
        for table_name in args.tables:
            transform_table(con, table_name)
    finally:
        con.close()

    print("Transformation and configured checks completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
