from pathlib import Path
import sys

import duckdb

RAW_DIR = Path("data/raw/dunnhumby")
SILVER_DIR = Path("data/silver/dunnhumby")

TABLES = [
    "campaign_desc",
    "campaign_table",
    "causal_data",
    "coupon",
    "coupon_redempt",
    "hh_demographic",
    "product",
    "transaction_data",
]

CANDIDATE_KEYS = {
    "campaign_desc": ["CAMPAIGN"],
    "campaign_table": ["household_key", "CAMPAIGN"],
    "coupon_redempt": ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"],
    "hh_demographic": ["household_key"],
    "product": ["PRODUCT_ID"],
}

RELATIONSHIPS = [
    ("campaign_table", "CAMPAIGN", "campaign_desc", "CAMPAIGN"),
    ("coupon", "PRODUCT_ID", "product", "PRODUCT_ID"),
    ("causal_data", "PRODUCT_ID", "product", "PRODUCT_ID"),
    ("transaction_data", "PRODUCT_ID", "product", "PRODUCT_ID"),
]


def sql_path(path):
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def source_relation(path):
    return f"read_csv_auto({sql_path(path)}, sample_size=10000)"


def silver_relation(path):
    return f"read_parquet({sql_path(path)})"


def columns(con, relation):
    return [
        row[0]
        for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    ]


def main():
    con = duckdb.connect(":memory:")
    failures = []
    total_rows = 0

    print("=" * 76)
    print("NEXUS MODULE 10 - REPRODUCIBLE SILVER VALIDATION")
    print("=" * 76)

    try:
        for table in TABLES:
            source_path = RAW_DIR / f"{table}.csv"
            silver_path = SILVER_DIR / f"{table}.parquet"

            if not source_path.is_file() or not silver_path.is_file():
                failures.append(f"{table}: source or Silver file missing")
                print(f"FAIL {table}: source or Silver file missing")
                continue

            src = source_relation(source_path)
            dst = silver_relation(silver_path)

            src_count = con.execute(
                f"SELECT COUNT(*) FROM {src}"
            ).fetchone()[0]
            dst_count = con.execute(
                f"SELECT COUNT(*) FROM {dst}"
            ).fetchone()[0]

            src_columns = columns(con, src)
            dst_columns = columns(con, dst)
            schema_match = src_columns == dst_columns
            count_match = src_count == dst_count

            total_rows += dst_count

            print(
                f"{'PASS' if count_match and schema_match else 'FAIL'} "
                f"{table:<20} rows={dst_count:>10,} "
                f"columns={len(dst_columns):>2} "
                f"column_order_match={schema_match}"
            )

            if not count_match:
                failures.append(f"{table}: source/Silver row counts differ")
            if not schema_match:
                failures.append(f"{table}: source/Silver column order differs")

            if table in CANDIDATE_KEYS:
                keys = CANDIDATE_KEYS[table]
                key_sql = ", ".join(f'"{key}"' for key in keys)
                null_condition = " OR ".join(
                    f'"{key}" IS NULL' for key in keys
                )

                null_rows = con.execute(
                    f"SELECT COUNT(*) FROM {dst} WHERE {null_condition}"
                ).fetchone()[0]

                duplicate_groups = con.execute(
                    f"""
                    SELECT COUNT(*) FROM (
                        SELECT {key_sql}
                        FROM {dst}
                        GROUP BY {key_sql}
                        HAVING COUNT(*) > 1
                    ) d
                    """
                ).fetchone()[0]

                key_pass = null_rows == 0 and duplicate_groups == 0
                print(
                    f"    candidate key: "
                    f"{'PASS' if key_pass else 'FAIL'} "
                    f"(null rows={null_rows:,}; "
                    f"duplicate groups={duplicate_groups:,})"
                )

                if not key_pass:
                    failures.append(f"{table}: candidate-key check failed")

        print("\nREFERENTIAL INTEGRITY")
        for left_table, left_col, right_table, right_col in RELATIONSHIPS:
            left_path = SILVER_DIR / f"{left_table}.parquet"
            right_path = SILVER_DIR / f"{right_table}.parquet"

            if not left_path.is_file() or not right_path.is_file():
                failures.append(
                    f"{left_table}.{left_col}: relationship input missing"
                )
                continue

            left = silver_relation(left_path)
            right = silver_relation(right_path)

            unmatched = con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT "{left_col}" AS k
                    FROM {left}
                    WHERE "{left_col}" IS NOT NULL
                ) l
                LEFT JOIN (
                    SELECT DISTINCT "{right_col}" AS k
                    FROM {right}
                    WHERE "{right_col}" IS NOT NULL
                ) r ON l.k = r.k
                WHERE r.k IS NULL
                """
            ).fetchone()[0]

            passed = unmatched == 0
            print(
                f"{'PASS' if passed else 'FAIL'} "
                f"{left_table}.{left_col} -> "
                f"{right_table}.{right_col} "
                f"(unmatched distinct keys={unmatched:,})"
            )

            if not passed:
                failures.append(
                    f"{left_table}.{left_col}: referential integrity failed"
                )

    finally:
        con.close()

    print("\nTotal Silver rows across all tables:", f"{total_rows:,}")
    print("=" * 76)

    if failures:
        print(f"VALIDATION FAILED: {len(failures)} issue(s)")
        for failure in failures:
            print(" -", failure)
        return 1

    print("VALIDATION PASSED: all configured checks completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
