import json
import subprocess
import sys

from rich.console import Console

from mlx import cli
from mlx.core.doctor import InspectEnvironment
from mlx.core import ui


def test_empty_environment_has_no_required_variables():
    report = InspectEnvironment({}).execute()
    assert report["required_variables"] == []
    assert all(row["status"] == "unset" for row in report["optional_variables"])


def test_report_exposes_presence_without_values_or_unrelated_variables():
    environment = {
        "AWS_SECRET_ACCESS_KEY": "synthetic-secret",
        "AWS_PROFILE": "",
        "UNRELATED_SECRET": "unrelated-secret",
    }
    original = environment.copy()
    report = InspectEnvironment(environment).execute()
    statuses = {row["name"]: row["status"] for row in report["optional_variables"]}
    assert statuses["AWS_SECRET_ACCESS_KEY"] == "set"
    assert statuses["AWS_PROFILE"] == "unset"
    assert "synthetic-secret" not in json.dumps(report)
    assert "UNRELATED_SECRET" not in json.dumps(report)
    assert "unrelated-secret" not in json.dumps(report)
    assert environment == original


def test_doctor_json_bypasses_workflows_and_seeding(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        raise AssertionError("Doctor must not initialize workflows or random seeds")

    monkeypatch.setattr(cli, "_resolve_mode_runner", unexpected)
    monkeypatch.setattr(cli, "seed_everything", unexpected)
    monkeypatch.setattr(cli, "apply_global_seed", unexpected)
    monkeypatch.setattr(cli.os, "environ", {})
    assert cli.main(["--doctor", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == InspectEnvironment({}).execute()
    assert captured.err == ""


def test_doctor_table_without_mode(monkeypatch):
    output = Console(record=True, width=120)
    monkeypatch.setattr(ui, "console", output)
    monkeypatch.setattr(cli.os, "environ", {"AWS_SECRET_ACCESS_KEY": "synthetic-secret"})
    assert cli.main(["--doctor"]) == 0
    rendered = output.export_text()
    assert "No environment variables are required" in rendered
    assert "AWS_SECRET_ACCESS_KEY" in rendered
    assert "synthetic-secret" not in rendered


def test_doctor_module_entrypoint_stays_lightweight():
    code = """
import runpy
import sys
sys.argv = ['mlx', '--doctor', '--format', 'json']
try:
    runpy.run_module('mlx', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not {'torch', 'cv2', 'boto3', 'ultralytics', 'libreyolo'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["required_variables"] == []


def test_doctor_option_is_discoverable_and_not_forwarded():
    parser = cli.build_parser()
    assert "--doctor" in parser.format_help()
    namespace = parser.parse_args(["--mode", "segmentation"])
    assert namespace.doctor is False
    assert "doctor" not in cli._build_config(namespace)
