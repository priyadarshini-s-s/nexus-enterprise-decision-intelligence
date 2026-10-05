from typing import Any

from pyspark.sql import DataFrame

from scripts.data_quality.engine.checks import (
    check_accepted_values,
    check_not_null,
    check_row_count,
    check_unique,
)
from scripts.data_quality.engine.custom_rules import (
    check_ord_007,
    check_ord_008,
)


CUSTOM_RULES = {
    "ORD-007": check_ord_007,
    "ORD-008": check_ord_008,
}


def validate_schema(
    df: DataFrame,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute generic schema-level checks defined in a data contract."""

    results: list[dict[str, Any]] = []

    for column_spec in contract.get("schema", []):
        column = column_spec["name"]

        if column_spec.get("nullable") is False:
            results.append(check_not_null(df, column))

        if column_spec.get("unique") is True:
            results.append(check_unique(df, column))

        if "accepted_values" in column_spec:
            results.append(
                check_accepted_values(
                    df,
                    column,
                    column_spec["accepted_values"],
                )
            )

    return results


def validate_reconciliation(
    df: DataFrame,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute dataset-level reconciliation checks."""

    results: list[dict[str, Any]] = []

    reconciliation = contract.get("reconciliation", {})

    expected_count = reconciliation.get(
        "expected_order_population"
    )

    if expected_count is not None:
        results.append(
            check_row_count(
                df,
                expected_count,
            )
        )

    return results


def validate_custom_rules(
    df: DataFrame,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute registered custom quality rules."""

    results: list[dict[str, Any]] = []

    for rule in contract.get("quality_rules", []):
        rule_id = rule.get("id")

        rule_function = CUSTOM_RULES.get(rule_id)

        if rule_function is not None:
            result = rule_function(df)

            # Preserve severity declared in the contract.
            result["severity"] = rule.get(
                "severity",
                "critical",
            )

            results.append(result)

    return results


def validate(
    df: DataFrame,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute all supported contract-driven checks."""

    results: list[dict[str, Any]] = []

    results.extend(
        validate_schema(
            df,
            contract,
        )
    )

    results.extend(
        validate_reconciliation(
            df,
            contract,
        )
    )

    results.extend(
        validate_custom_rules(
            df,
            contract,
        )
    )

    return results


def validation_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a compact validation summary."""

    total = len(results)

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = sum(
        result["status"] == "FAIL"
        for result in results
    )

    return {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
    }