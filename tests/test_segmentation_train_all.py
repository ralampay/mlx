from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation import runner as segmentation_runner
from mlx.modes.segmentation.requests import TrainSegmentationRequest
from mlx.modes.segmentation.models import SMALL_MODEL_NAMES
from mlx.modes.segmentation.train_all import (
    TrainAllSegmentationModels,
    validate_all_models_request,
)
from mlx.modes.segmentation.utils import resolve_train_output_paths


def _dataset_with_test(root: Path) -> Path:
    for child in (
        "train/images",
        "train/masks",
        "val/images",
        "val/masks",
        "test/images",
        "test/masks",
    ):
        (root / child).mkdir(parents=True, exist_ok=True)
    (root / "test/images/sample.png").write_bytes(b"image")
    (root / "test/masks/sample.png").write_bytes(b"mask")
    return root


class FakeTrainer:
    def __init__(self, request, calls, *, failing_model=None):
        self.request = request
        self.calls = calls
        self.failing_model = failing_model

    def execute(self):
        self.calls.append(self.request.model)
        if self.request.model == self.failing_model:
            raise MLXUserError("training failed")
        paths = resolve_train_output_paths(
            self.request.to_config(),
            model_name=str(self.request.model),
        )
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        paths["checkpoint_path"].write_bytes(b"checkpoint")
        paths["training_csv_path"].write_text(
            "epoch,val_loss\n1,0.4\n2,0.2\n",
            encoding="utf-8",
        )


class FakeBenchmark:
    def __init__(self, request, calls):
        self.request = request
        self.calls = calls

    def execute(self):
        self.calls.append(self.request)
        score = {"a-model": 0.6, "b-model": 0.8}.get(
            str(self.request.model),
            0.1,
        )
        return {
            "mean_foreground_dice": score,
            "mean_foreground_iou": score - 0.1,
            "macro_dice": score,
            "mean_iou": score - 0.1,
            "pixel_accuracy": 0.9,
            "cross_entropy_loss": 0.2,
            "generalized_dice": score,
            "cohen_kappa": 0.7,
            "images_per_second_forward": 10.0,
        }


def _request(tmp_path: Path, **overrides) -> TrainSegmentationRequest:
    values = {
        "model": "all",
        "dataset_path": str(_dataset_with_test(tmp_path / "dataset")),
        "output_path": str(tmp_path / "output"),
        "pretrained": False,
    }
    values.update(overrides)
    return TrainSegmentationRequest.from_config(values)


def test_train_all_uses_every_model_and_writes_ranked_results(tmp_path: Path) -> None:
    training_calls = []
    benchmark_calls = []
    request = _request(tmp_path)
    result = TrainAllSegmentationModels(
        request,
        model_names=["b-model", "a-model"],
        trainer_factory=lambda value, reporter: FakeTrainer(value, training_calls),
        benchmark_factory=lambda value, reporter: FakeBenchmark(value, benchmark_calls),
    ).execute()

    assert training_calls == ["a-model", "b-model"]
    assert [item.model_name for item in result.results] == ["b-model", "a-model"]
    assert all(call.split == "test" for call in benchmark_calls)
    assert [Path(str(call.model_path)).name for call in benchmark_calls] == [
        "a-model.pth",
        "b-model.pth",
    ]
    assert [Path(str(call.output_path)).parent.name for call in benchmark_calls] == [
        "a-model",
        "b-model",
    ]
    leaderboard = (tmp_path / "output/leaderboard.csv").read_text(encoding="utf-8")
    assert leaderboard.index("b-model") < leaderboard.index("a-model")
    summary = json.loads(
        (tmp_path / "output/all-models.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "completed"
    assert summary["completed_models"] == 2
    assert summary["results"][0]["best_validation_loss"] == pytest.approx(0.2)


def test_train_all_fails_fast_and_preserves_partial_summary(tmp_path: Path) -> None:
    training_calls = []
    benchmark_calls = []
    command = TrainAllSegmentationModels(
        _request(tmp_path),
        model_names=["a-model", "b-model", "c-model"],
        trainer_factory=lambda value, reporter: FakeTrainer(
            value,
            training_calls,
            failing_model="b-model",
        ),
        benchmark_factory=lambda value, reporter: FakeBenchmark(value, benchmark_calls),
    )

    with pytest.raises(MLXUserError, match="failed for 'b-model'"):
        command.execute()

    assert training_calls == ["a-model", "b-model"]
    assert len(benchmark_calls) == 1
    summary = json.loads(
        (tmp_path / "output/all-models.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["current_model"] == "b-model"
    assert summary["completed_models"] == 1


def test_train_all_ranks_ties_by_name_and_nan_last(tmp_path: Path) -> None:
    training_calls = []
    scores = {
        "a-model": 0.5,
        "b-model": float("nan"),
        "c-model": 0.5,
    }

    class ScoreBenchmark:
        def __init__(self, request):
            self.request = request

        def execute(self):
            return {"mean_foreground_dice": scores[str(self.request.model)]}

    result = TrainAllSegmentationModels(
        _request(tmp_path),
        model_names=["c-model", "b-model", "a-model"],
        trainer_factory=lambda value, reporter: FakeTrainer(value, training_calls),
        benchmark_factory=lambda value, reporter: ScoreBenchmark(value),
    ).execute()

    assert [item.model_name for item in result.results] == [
        "a-model",
        "c-model",
        "b-model",
    ]


def test_all_small_uses_only_the_registered_small_group(tmp_path: Path) -> None:
    training_calls = []
    request = _request(tmp_path, model="all-small")

    result = TrainAllSegmentationModels(
        request,
        trainer_factory=lambda value, reporter: FakeTrainer(value, training_calls),
        benchmark_factory=lambda value, reporter: FakeBenchmark(value, []),
    ).execute()

    assert training_calls == sorted(SMALL_MODEL_NAMES)
    assert {item.model_name for item in result.results} == set(SMALL_MODEL_NAMES)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pretrained": True}, "scratch initialization"),
        ({"model_path": "resume.pth"}, "cannot use one --model-path"),
        ({"output_path": "models.pth"}, "directory output"),
    ],
)
def test_train_all_rejects_incompatible_options(
    tmp_path: Path,
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(MLXUserError, match=message):
        validate_all_models_request(_request(tmp_path, **overrides))


def test_train_all_requires_test_and_empty_output(tmp_path: Path) -> None:
    dataset = tmp_path / "without-test"
    dataset.mkdir()
    request = _request(tmp_path, dataset_path=str(dataset))
    with pytest.raises(MLXUserError, match="requires a test/images"):
        TrainAllSegmentationModels(request, model_names=["unet"]).execute()

    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(MLXUserError, match="must be empty"):
        validate_all_models_request(_request(tmp_path, output_path=str(output)))

    source_only = tmp_path / "source-only"
    source_only.mkdir()
    (source_only / "dataset_source.json").write_text("{}", encoding="utf-8")
    source_request = _request(tmp_path, output_path=str(source_only))
    with pytest.raises(MLXUserError, match="must be empty"):
        validate_all_models_request(source_request)
    assert validate_all_models_request(
        source_request,
        allow_dataset_source_manifest=True,
    ) == source_only


def test_segmentation_runner_clears_implicit_dataset_for_s3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeDatasetTraining:
        def __init__(self, request, **kwargs):
            captured["request"] = request

        def execute(self):
            return "trained"

    monkeypatch.setattr(
        segmentation_runner,
        "TrainWithDatasetSource",
        FakeDatasetTraining,
    )

    result = segmentation_runner._train(
        {
            "dataset_path": "./tmp/dataset",
            "dataset_s3_uri": "s3://datasets/segmentation.zip",
            "dataset_cache_dir": str(tmp_path / "cache"),
            "device": "cpu",
            "model": "unet",
            "output_path": str(tmp_path / "output"),
            "_explicit_options": {"dataset_s3_uri"},
        }
    )

    assert result == "trained"
    assert captured["request"].dataset_path == ""


def test_segmentation_runner_rejects_explicit_local_and_s3_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(MLXUserError, match="either a local dataset path"):
        segmentation_runner._train(
            {
                "dataset_path": str(tmp_path / "local"),
                "dataset_s3_uri": "s3://datasets/segmentation.zip",
                "device": "cpu",
                "model": "unet",
                "output_path": str(tmp_path / "output"),
                "_explicit_options": {"dataset_path", "dataset_s3_uri"},
            }
        )


@pytest.mark.parametrize("model_group", ["all", "all-small"])
def test_segmentation_runner_stages_once_around_all_model_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_group: str,
) -> None:
    captured = {"wrappers": 0, "commands": 0}
    resolved_dataset = _dataset_with_test(tmp_path / "resolved")

    class FakeAllModels:
        def __init__(self, request, **kwargs):
            captured["commands"] += 1
            captured["resolved_request"] = request

        def execute(self):
            return "all-trained"

    class FakeDatasetTraining:
        def __init__(self, request, *, trainer_factory, **kwargs):
            captured["wrappers"] += 1
            self.request = request
            self.trainer_factory = trainer_factory

        def execute(self):
            resolved = replace(self.request, dataset_path=str(resolved_dataset))
            return self.trainer_factory(resolved).execute()

    monkeypatch.setattr(segmentation_runner, "TrainAllSegmentationModels", FakeAllModels)
    monkeypatch.setattr(
        segmentation_runner,
        "TrainWithDatasetSource",
        FakeDatasetTraining,
    )

    result = segmentation_runner._train(
        {
            "dataset_path": "./tmp/dataset",
            "dataset_s3_uri": "s3://datasets/segmentation.zip",
            "dataset_cache_dir": str(tmp_path / "cache"),
            "device": "cpu",
            "model": model_group,
            "output_path": str(tmp_path / "output"),
            "pretrained": False,
            "_explicit_options": {"dataset_s3_uri"},
        }
    )

    assert result == "all-trained"
    assert captured["wrappers"] == 1
    assert captured["commands"] == 1
    assert captured["resolved_request"].dataset_path == str(resolved_dataset)


@pytest.mark.parametrize("model_group", ["all", "all-small"])
def test_model_all_is_rejected_outside_training(model_group: str) -> None:
    with pytest.raises(MLXUserError, match="only with --action train"):
        segmentation_runner.run_segmentation(
            {"action": "benchmark", "model": model_group}
        )
