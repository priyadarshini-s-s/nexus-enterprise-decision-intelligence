
"""Read-only audit collector for NEXUS Airflow DAG runs."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


DEFAULT_DAG_ID = "nexus_data_pipeline"
DEFAULT_RUN_ID = "manual__2026-10-10T06:32:48.561568+00:00"
DEFAULT_BASE_URL = "http://localhost:8080"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "monitoring"


def get_json(session, url, params=None):
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Collect read-only Airflow run and task audit metadata."
    )
    parser.add_argument("--dag-id", default=DEFAULT_DAG_ID)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--base-url",
        default=os.getenv("AIRFLOW_API_URL", DEFAULT_BASE_URL).rstrip("/"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()


    # Use an Airflow JWT obtained through the supported login endpoint.
    # Never store the password or token in this source file.
    token = os.getenv("AIRFLOW_API_TOKEN")

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if token:
        session.headers["Authorization"] = f"Bearer {token}"


    api_root = f"{args.base_url}/api/v2"
    dag_id = requests.utils.quote(args.dag_id, safe="")
    run_id = requests.utils.quote(args.run_id, safe="")

    try:
        run = get_json(
            session,
            f"{api_root}/dags/{dag_id}/dagRuns/{run_id}",
        )
        task_response = get_json(
            session,
            f"{api_root}/dags/{dag_id}/dagRuns/{run_id}/taskInstances",
            params={"limit": 100, "offset": 0, "order_by": "map_index"},
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(
            f"Airflow API request failed (HTTP {status}). "
            "Check API access and authentication.",
            file=sys.stderr,
        )
        return 1
    except requests.RequestException as exc:
        print(f"Cannot reach Airflow API: {exc}", file=sys.stderr)
        return 1

    tasks = task_response.get("task_instances", [])
    if task_response.get("next_cursor"):
        print(
            "Warning: API returned a pagination cursor; "
            "this report may not contain every task.",
            file=sys.stderr,
        )

    task_records = []
    for task in sorted(tasks, key=lambda item: item.get("task_id", "")):
        task_records.append(
            {
                "task_id": task.get("task_id"),
                "state": task.get("state"),
                "start_date": task.get("start_date"),
                "end_date": task.get("end_date"),
                "duration_seconds": task.get("duration"),
                "try_number": task.get("try_number"),
                "max_tries": task.get("max_tries"),
                "operator": task.get("operator"),
            }
        )

    failed_tasks = [
        task["task_id"]
        for task in task_records
        if task["state"] in {"failed", "upstream_failed"}
    ]

    report = {
        "audit_schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Airflow REST API v2",
        "dag": {
            "dag_id": run.get("dag_id"),
            "dag_run_id": run.get("dag_run_id"),
            "run_type": run.get("run_type"),
            "state": run.get("state"),
            "start_date": run.get("start_date"),
            "end_date": run.get("end_date"),
            "duration_seconds": run.get("duration"),
            "dag_version": [
                version.get("version_number")
                for version in run.get("dag_versions", [])
            ],
        },
        "task_summary": {
            "total": len(task_records),
            "success": sum(t["state"] == "success" for t in task_records),
            "failed": sum(t["state"] == "failed" for t in task_records),
            "upstream_failed": sum(
                t["state"] == "upstream_failed" for t in task_records
            ),
            "other_or_incomplete": sum(
                t["state"] not in {
                    "success", "failed", "upstream_failed"
                }
                for t in task_records
            ),
            "failed_task_ids": failed_tasks,
        },
        "tasks": task_records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in args.run_id
    )
    output_file = args.output_dir / f"airflow_audit_{safe_run_id}.json"
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"DAG: {report['dag']['dag_id']}")
    print(f"Run status: {report['dag']['state']}")
    print(f"Tasks recorded: {report['task_summary']['total']}")
    print(f"Successful tasks: {report['task_summary']['success']}")
    print(f"Failed tasks: {report['task_summary']['failed']}")
    print(f"Audit report: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
