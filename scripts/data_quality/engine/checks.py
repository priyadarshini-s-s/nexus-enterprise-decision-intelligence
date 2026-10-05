from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def check_not_null(
    df: DataFrame,
    column: str,
) -> dict[str, Any]:
    """Check whether a column contains null values."""

    null_count = df.filter(F.col(column).isNull()).count()

    return {
        "check": "not_null",
        "column": column,
        "status": "PASS" if null_count == 0 else "FAIL",
        "observed": null_count,
        "expected": 0,
    }


def check_unique(
    df: DataFrame,
    column: str,
) -> dict[str, Any]:
    """Check whether a column contains duplicate values."""

    total_count = df.count()
    distinct_count = df.select(column).distinct().count()
    duplicate_count = total_count - distinct_count

    return {
        "check": "unique",
        "column": column,
        "status": "PASS" if duplicate_count == 0 else "FAIL",
        "observed": duplicate_count,
        "expected": 0,
    }


def check_accepted_values(
    df: DataFrame,
    column: str,
    accepted_values: list[Any],
) -> dict[str, Any]:
    """Check whether all non-null values belong to the accepted domain."""

    invalid_count = (
        df.filter(
            F.col(column).isNotNull()
            & (~F.col(column).isin(accepted_values))
        )
        .count()
    )

    return {
        "check": "accepted_values",
        "column": column,
        "status": "PASS" if invalid_count == 0 else "FAIL",
        "observed": invalid_count,
        "expected": 0,
        "accepted_values": accepted_values,
    }


def check_row_count(
    df: DataFrame,
    expected_count: int,
) -> dict[str, Any]:
    """Check whether the dataset row count matches the contract."""

    actual_count = df.count()

    return {
        "check": "row_count",
        "column": None,
        "status": "PASS" if actual_count == expected_count else "FAIL",
        "observed": actual_count,
        "expected": expected_count,
    }