from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def check_ord_007(df: DataFrame) -> dict[str, Any]:
    """
    ORD-007:
    When delivered customer timestamp is present, it must not
    precede the order purchase timestamp.
    """

    violations = df.filter(
        F.col("order_delivered_customer_date").isNotNull()
        & (
            F.col("order_delivered_customer_date")
            < F.col("order_purchase_timestamp")
        )
    ).count()

    return {
        "check": "ORD-007",
        "column": "order_delivered_customer_date",
        "status": "PASS" if violations == 0 else "FAIL",
        "observed": violations,
        "expected": 0,
    }


def check_ord_008(df: DataFrame) -> dict[str, Any]:
    """
    ORD-008:
    Delivered orders with a missing customer delivery timestamp
    must be explicitly tracked as observations.

    This is an informational/warning rule rather than a failure.
    """

    observations = df.filter(
        (F.col("order_status") == "delivered")
        & F.col("order_delivered_customer_date").isNull()
    ).count()

    return {
        "check": "ORD-008",
        "column": "order_delivered_customer_date",
        "status": "PASS",
        "observed": observations,
        "expected": "tracked explicitly",
        "severity": "warning",
    }