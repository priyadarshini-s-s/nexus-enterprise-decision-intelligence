import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from scripts.data_quality.engine.contract_loader import load_contract
from scripts.data_quality.engine.reporter import (
    build_report,
    print_report,
    write_json_report,
)
from scripts.data_quality.engine.validator import validate


def create_spark_session() -> SparkSession:
    """Create the local Spark session used by the DQ engine."""

    return (
        SparkSession.builder
        .appName("NEXUS-Data-Quality")
        .master("local[2]")
        .getOrCreate()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEXUS reusable data quality runner"
    )

    parser.add_argument(
        "--contract",
        required=True,
        help="Path to the YAML data contract.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the Spark-readable dataset.",
    )

    parser.add_argument(
        "--format",
        default="parquet",
        choices=["parquet"],
        help="Input dataset format.",
    )

    parser.add_argument(
        "--report",
        default="reports/data_quality/dq_report.json",
        help="Output JSON report path.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    contract_path = Path(args.contract)
    input_path = Path(args.input)

    print("NEXUS — DATA QUALITY ENGINE")
    print("=" * 40)
    print(f"Contract : {contract_path}")
    print(f"Input    : {input_path}")
    print(f"Format   : {args.format}")

    contract = load_contract(contract_path)

    dataset_name = contract["dataset"]["name"]

    spark = create_spark_session()

    try:
        df = spark.read.format(args.format).load(str(input_path))

        results = validate(
            df=df,
            contract=contract,
        )

        report = build_report(
            dataset=dataset_name,
            results=results,
        )

        print_report(report)

        write_json_report(
            report=report,
            output_path=args.report,
        )

        print()
        print(f"JSON report : {args.report}")

        if report["summary"]["status"] == "FAIL":
            print("DQ ENGINE: FAILED")
            return 1

        print("DQ ENGINE: PASSED")
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())