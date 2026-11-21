"""Exercise standalone guide projects without installing them into the test environment."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "guides"
PROJECT_FILES = re.compile(r"### File: `([^`]+)`\n\n```\w+\n(.*?)```", re.DOTALL)


def extract_project(path: Path, destination: Path) -> None:
    files = PROJECT_FILES.findall(path.read_text())
    assert files, "Guide must contain an executable project."
    for relative, source in files:
        target = destination / relative
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)


@pytest.mark.parametrize("name", ["kalman-filter", "deep-sort"])
def test_guide_project_python_is_valid(name, tmp_path):
    extract_project(GUIDES / f"tracking-plugin-{name}.md", tmp_path)
    for source in tmp_path.rglob("*.py"):
        ast.parse(source.read_text(), filename=str(source))


@pytest.mark.parametrize("name, module, class_name", [
    ("kalman-filter", "kalman", "KalmanIoUTracker"),
    ("deep-sort", "deep_sort", "DeepSortTracker"),
])
def test_standalone_tracking_guide_contract(name, module, class_name, tmp_path):
    if module == "deep_sort" and importlib.util.find_spec("deep_sort_realtime") is None:
        pytest.skip("Optional guide backend: install deep-sort-realtime==1.3.2 to exercise it.")
    extract_project(GUIDES / f"tracking-plugin-{name}.md", tmp_path)
    environment = os.environ.copy()
    # Isolate identically named tutorial packages in child interpreters. Users
    # install their project; tests avoid modifying the active Python environment.
    environment["PYTHONPATH"] = os.pathsep.join([
        str(tmp_path / "src"), str(ROOT), environment.get("PYTHONPATH", ""),
    ])
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=tmp_path, env=environment, text=True, capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    guide = (GUIDES / f"tracking-plugin-{name}.md").read_text()
    loader_example = re.search(r"python - <<'PYTHON'\n(.*?)\nPYTHON", guide, re.DOTALL)
    assert loader_example, "Guide must include a runnable plugin-loader verification."
    result = subprocess.run(
        [sys.executable, "-c", loader_example.group(1)], cwd=tmp_path,
        env=environment, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert class_name in result.stdout
    if module == "kalman":
        result = subprocess.run(
            [sys.executable, "examples/run_observations.py"], cwd=tmp_path,
            env=environment, text=True, capture_output=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Confirmed track ID: 1" in result.stdout
        assert "After an empty frame: lost" in result.stdout
    script = r'''
import json
import numpy as np
from mlx.cli import build_parser
from mlx.modes.object_detection.models import Detection, DetectionResult
from mlx.modes.object_detection.tracking import CreateTrackingAlgorithm, RunTrackingVideo, TrackingRequest
from mlx.modes.object_detection.tracking.mot import read_mot_file

reference = REFERENCE
module = MODULE
# The documented JSON must load through MLX unchanged.
configured = CreateTrackingAlgorithm(tracker=reference, config_path=f"configs/{module}.json").execute()
configured.reset()
options = {"max_age": 1}
if module == "deep_sort":
    options.update(n_init=2, encoder=lambda crops: np.tile([1., 0.], (len(crops), 1)))
else:
    options.update(min_hits=2)
algorithm = CreateTrackingAlgorithm(tracker=reference, options=options).execute()

class Detector:
    def __init__(self):
        self.calls = 0
    def predict(self, frame):
        self.calls += 1
        detections = (Detection(xyxy=(4., 4., 20., 24.), confidence=.9, class_id=0, label="object"),)
        return DetectionResult(detections=detections if self.calls <= 2 else (), names={0: "object"})

class Source:
    def __init__(self):
        self.remaining = 4
        self.released = False
    def read(self):
        self.remaining -= 1
        return (True, np.zeros((32, 32, 3), dtype=np.uint8)) if self.remaining >= 0 else (False, None)
    def release(self):
        self.released = True

source = Source()
result = RunTrackingVideo(TrackingRequest(output_path="output", display=False),
                          detector=Detector(), algorithm=algorithm, frame_source=source).execute()
assert result.frames_processed == 4
assert source.released
records = read_mot_file(result.output_path)
assert [(r.frame_id, r.track_id) for r in records] == [(2, 1)]
rows = [json.loads(line) for line in result.class_aware_output_path.read_text().splitlines()]
assert len(rows) == 1
assert result.replay_html_path.is_file()
# Session cleanup reset the plugin index and tracks.
assert algorithm.update(frame_index=1, detections=[]).tracks == ()
args = build_parser().parse_args([
    "--mode", "track", "--action", "run", "--provider", "libreyolo",
    "--model-path", "detector.pt", "--file-path", "video.mp4", "--device", "cpu",
    "--confidence", "0.25", "--tracker", reference,
    "--tracker-config", f"configs/{module}.json", "--output", "tracking-output", "--no-display",
])
assert args.tracker == reference
'''.replace("REFERENCE", repr(f"custom_tracking.mlx_plugins.{module}:{class_name}")).replace("MODULE", repr(module))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=environment,
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
