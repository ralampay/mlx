from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.comparison_experiment import (
    AnalyzeCSPDraxComparison,
    CANDIDATE,
    CONTROL,
    CSPDraxComparisonRequest,
    PrepareCSPDraxComparison,
    RunCSPDraxComparison,
)


def _request(tmp_path: Path) -> CSPDraxComparisonRequest:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: [object]\n", encoding="utf-8")
    source = tmp_path / "yolox_m.pth"
    source.write_bytes(b"source")
    return CSPDraxComparisonRequest(
        dataset=dataset,
        output=tmp_path / "experiment",
        source_checkpoint=source,
        seeds=(3, 5),
        bootstrap_draws=100,
    )


def test_schedule_reverses_pair_order_between_seeds(tmp_path):
    command = RunCSPDraxComparison(_request(tmp_path))
    assert command._balanced_schedule((3, 5)) == [
        (3, CONTROL),
        (3, CANDIDATE),
        (5, CANDIDATE),
        (5, CONTROL),
    ]


def test_scratch_request_never_supplies_initial_weights(tmp_path):
    request = replace(_request(tmp_path), scratch=True)
    request.source_checkpoint.unlink()
    request.validate()
    command = RunCSPDraxComparison(request)
    training = command._training_request(CANDIDATE, 3, None, request.output, False)
    assert training.model_path is None
    assert training.pretrained is False
    assert training.random_seed == 3


def test_training_request_disables_accumulation_and_uses_common_protocol(tmp_path):
    request = _request(tmp_path)
    command = RunCSPDraxComparison(request)
    training = command._training_request(
        CANDIDATE,
        3,
        request.source_checkpoint,
        request.output / "run",
        False,
    )
    assert training.batch_size == 8
    assert training.nbs == training.batch_size
    assert training.loss_clip is None
    assert training.epochs == 100
    assert training.lr0 == 0.001


def test_dataset_record_requires_the_predeclared_split_volume(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in ("train", "val", "test"):
        (dataset / f"{split}_manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(MLXUserError, match="Expected 1000 train images"):
        PrepareCSPDraxComparison._dataset_record(dataset)


def test_incomplete_report_refuses_an_inferential_conclusion(tmp_path):
    request = _request(tmp_path)
    run = request.output / "runs" / CONTROL / "seed-3"
    run.mkdir(parents=True)
    (run / "complete.json").write_text(
        json.dumps(
            {
                "model": CONTROL,
                "seed": 3,
                "parameters": 10,
                "training_seconds": 1.0,
                "checkpoint_sha256": "abc",
                "metrics": {
                    "map_50_95": 0.2,
                    "map_50": 0.3,
                    "precision": 0.4,
                    "recall": 0.5,
                    "f1": 0.44,
                },
            }
        ),
        encoding="utf-8",
    )
    result = AnalyzeCSPDraxComparison(request).execute()
    assert result["decision"] == "incomplete"
    report = (request.output / "REPORT.md").read_text(encoding="utf-8")
    assert "no inferential conclusion" in report
    assert "Invalidated pilot" in report
