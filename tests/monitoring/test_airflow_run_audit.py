
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from scripts.monitoring import airflow_run_audit as audit


class TestAirflowRunAudit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        self.run = {
            "dag_id": "nexus_data_pipeline",
            "dag_run_id": "manual__test-run",
            "run_type": "manual",
            "state": "success",
            "start_date": "2026-10-10T06:32:49Z",
            "end_date": "2026-10-10T06:34:38Z",
            "duration": 109.0,
            "dag_versions": [{"version_number": 1}],
        }

        self.tasks = {
            "task_instances": [
                {
                    "task_id": "validate_silver",
                    "state": "success",
                    "start_date": "2026-10-10T06:32:50Z",
                    "end_date": "2026-10-10T06:33:22Z",
                    "duration": 32.0,
                    "try_number": 1,
                    "max_tries": 2,
                    "operator": "DockerOperator",
                },
                {
                    "task_id": "dq_gate",
                    "state": "success",
                    "start_date": "2026-10-10T06:33:23Z",
                    "end_date": "2026-10-10T06:33:51Z",
                    "duration": 28.0,
                    "try_number": 1,
                    "max_tries": 2,
                    "operator": "DockerOperator",
                },
            ]
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_collector(self, side_effect, extra_args=None):
        args = [
            "airflow_run_audit.py",
            "--run-id",
            "manual__test-run",
            "--output-dir",
            str(self.output_dir),
        ]
        if extra_args:
            args.extend(extra_args)

        with patch.object(audit.sys, "argv", args):
            with patch.dict(
                "os.environ",
                {"AIRFLOW_API_TOKEN": "test-token"},
            ):
                with patch.object(
                    audit.requests,
                    "Session",
                ) as session_factory:
                    session = session_factory.return_value
                    session.get.side_effect = side_effect
                    return_code = audit.main()

        return return_code, session

    @staticmethod
    def response(payload):
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def test_success_writes_valid_audit_report(self):
        return_code, session = self.run_collector(
            [
                self.response(self.run),
                self.response(self.tasks),
            ]
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(session.get.call_count, 2)

        reports = list(self.output_dir.glob("airflow_audit_*.json"))
        self.assertEqual(len(reports), 1)

        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["dag"]["state"], "success")
        self.assertEqual(report["task_summary"]["total"], 2)
        self.assertEqual(report["task_summary"]["success"], 2)
        self.assertEqual(report["task_summary"]["failed"], 0)
        self.assertEqual(report["task_summary"]["failed_task_ids"], [])

    def test_http_error_returns_failure_without_report(self):
        response = Mock()
        response.status_code = 401

        error = requests.HTTPError(
            "Unauthorized",
            response=response,
        )

        return_code, _ = self.run_collector([error])

        self.assertEqual(return_code, 1)
        self.assertEqual(
            list(self.output_dir.glob("airflow_audit_*.json")),
            [],
        )

    def test_connection_error_returns_failure(self):
        error = requests.ConnectionError("Connection refused")

        return_code, _ = self.run_collector([error])

        self.assertEqual(return_code, 1)
        self.assertEqual(
            list(self.output_dir.glob("airflow_audit_*.json")),
            [],
        )

    def test_failed_task_is_counted(self):
        failed_tasks = {
            "task_instances": [
                {
                    "task_id": "dq_gate",
                    "state": "failed",
                    "start_date": None,
                    "end_date": None,
                    "duration": None,
                    "try_number": 3,
                    "max_tries": 2,
                    "operator": "DockerOperator",
                }
            ]
        }

        return_code, _ = self.run_collector(
            [
                self.response(self.run),
                self.response(failed_tasks),
            ]
        )

        self.assertEqual(return_code, 0)

        report_path = next(
            self.output_dir.glob("airflow_audit_*.json")
        )
        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )

        self.assertEqual(report["task_summary"]["failed"], 1)
        self.assertEqual(
            report["task_summary"]["failed_task_ids"],
            ["dq_gate"],
        )

    def test_pagination_cursor_is_reported(self):
        paginated_tasks = {
            "task_instances": self.tasks["task_instances"],
            "next_cursor": "next-page",
        }

        return_code, _ = self.run_collector(
            [
                self.response(self.run),
                self.response(paginated_tasks),
            ]
        )

        self.assertEqual(return_code, 0)

        report_path = next(
            self.output_dir.glob("airflow_audit_*.json")
        )
        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        self.assertEqual(report["task_summary"]["total"], 2)


if __name__ == "__main__":
    unittest.main()
