import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_report(
    dataset: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a machine-readable DQ validation report."""

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
        "dataset": dataset,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "status": "PASS" if failed == 0 else "FAIL",
        },
        "checks": results,
    }


def write_json_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a DQ report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable DQ summary."""

    summary = report["summary"]

    print()
    print("NEXUS — DATA QUALITY REPORT")
    print("=" * 40)
    print(f"Dataset       : {report['dataset']}")
    print(f"Timestamp UTC : {report['timestamp_utc']}")
    print()
    print(f"Total checks  : {summary['total_checks']}")
    print(f"Passed        : {summary['passed']}")
    print(f"Failed        : {summary['failed']}")
    print(f"Status        : {summary['status']}")
    print()

    for result in report["checks"]:
        print(
            f"[{result['status']}] "
            f"{result['check']} "
            f"{result.get('column') or 'dataset'} "
            f"(observed={result['observed']}, "
            f"expected={result['expected']})"
        )