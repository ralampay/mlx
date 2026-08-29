from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mlx.core.artifacts import json_safe, sha256_file, write_csv, write_json_atomic


@dataclass(frozen=True)
class ArtifactValue:
    path: Path
    score: float


def test_json_artifacts_are_deterministic_and_strict(tmp_path) -> None:
    target = tmp_path / "result.json"
    write_json_atomic(
        target,
        {"value": ArtifactValue(Path("model.pth"), float("nan")), "labels": {"b", "a"}},
    )

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "labels": ["a", "b"],
        "value": {"path": "model.pth", "score": None},
    }
    assert sha256_file(target) == sha256_file(target)


def test_csv_artifacts_normalize_nested_values(tmp_path) -> None:
    target = tmp_path / "records.csv"
    write_csv(target, [{"source": Path("clip"), "frames": [1, 2]}])

    assert target.read_text(encoding="utf-8").splitlines() == [
        "source,frames",
        'clip,"[1,2]"',
    ]


def test_json_safe_handles_nonfinite_values() -> None:
    assert json_safe({"positive": float("inf"), "negative": float("-inf")}) == {
        "positive": None,
        "negative": None,
    }
