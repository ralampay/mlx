from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary, count_model_parameters
from mlx.core.ui import build_model_parameter_table
from mlx.modes.image_classification import list_models as classification_listing
from mlx.modes.image_classification import runner as classification_runner
from mlx.modes.object_detection.ultralytics import list_models as detection_listing
from mlx.modes.object_detection.ultralytics import runner as detection_runner
from mlx.modes.object_detection.ultralytics import utils as detection_utils
from mlx.modes.segmentation import list_models as segmentation_listing
from mlx.modes.segmentation import runner as segmentation_runner


def test_parameter_count_includes_frozen_parameters() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)

    assert count_model_parameters(model) == 11


def test_model_table_has_exact_columns_and_formatted_counts() -> None:
    table = build_model_parameter_table(
        [
            ModelParameterSummary("large-model", 5000),
            ModelParameterSummary("tiny-model", 1234),
            ModelParameterSummary("another-tiny-model", 1234),
        ]
    )
    output = Console(record=True, width=80)
    output.print(table)
    rendered = output.export_text()

    assert [column.header for column in table.columns] == [
        "Model Name",
        "Parameter Count",
    ]
    assert len(table.columns) == 2
    assert "tiny-model" in rendered
    assert "1,234" in rendered
    assert rendered.index("another-tiny-model") < rendered.index("tiny-model")
    assert rendered.index("tiny-model") < rendered.index("large-model")


def test_image_classification_listing_builds_every_registered_model(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        classification_listing,
        "supported_model_names",
        lambda: ["one-shot-model", "standard-model"],
    )
    monkeypatch.setattr(
        classification_listing,
        "model_family_for",
        lambda name: "one-shot" if name == "one-shot-model" else "standard",
    )

    def fake_build(model_name, config, **kwargs):
        calls.append((model_name, config, kwargs))
        return nn.Linear(2, 3)

    monkeypatch.setattr(
        classification_listing,
        "build_image_classification_model",
        fake_build,
    )

    summaries = classification_listing.ListImageClassificationModels(
        {"num_classes": 4, "pretrained": True, "embedding_size": 7}
    ).execute()

    assert summaries == [
        ModelParameterSummary("one-shot-model", 9),
        ModelParameterSummary("standard-model", 9),
    ]
    assert calls[0][2] == {}
    assert calls[1][2] == {"num_classes": 4}
    assert all(call[1]["pretrained"] is False for call in calls)
    assert all(call[1]["embedding_size"] == 7 for call in calls)


def test_segmentation_listing_builds_registered_models_with_class_count(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        segmentation_listing,
        "supported_model_names",
        lambda: ["segment-a", "segment-b"],
    )

    def fake_build(model_name, config, *, num_classes):
        calls.append((model_name, config, num_classes))
        return nn.Linear(2, num_classes)

    monkeypatch.setattr(segmentation_listing, "build_segmentation_model", fake_build)

    summaries = segmentation_listing.ListSegmentationModels(
        {"num_classes": 3, "colored": True}
    ).execute()

    assert summaries == [
        ModelParameterSummary("segment-a", 9),
        ModelParameterSummary("segment-b", 9),
    ]
    assert [call[0] for call in calls] == ["segment-a", "segment-b"]
    assert all(call[2] == 3 for call in calls)


def test_invalid_listing_class_count_is_user_facing() -> None:
    with pytest.raises(MLXUserError, match="num-classes must be at least 1"):
        segmentation_listing.ListSegmentationModels({"num_classes": 0}).execute()


def test_detection_listing_uses_canonical_aliases_and_underlying_model(monkeypatch) -> None:
    resolved = []
    initialized = []

    def fake_resolve(config, *, require_yaml, require_weights):
        resolved.append((config, require_yaml, require_weights))
        return Path(f"/models/{config['model']}.yaml"), None

    def fake_initialize(model_path, weights_path, *, prefer_cfg):
        initialized.append((model_path, weights_path, prefer_cfg))
        return SimpleNamespace(model=nn.Linear(3, 2))

    monkeypatch.setattr(detection_listing, "resolve_model_paths", fake_resolve)
    monkeypatch.setattr(detection_listing, "initialize_model", fake_initialize)

    summaries = detection_listing.ListObjectDetectionModels().execute()

    assert summaries == [
        ModelParameterSummary("draxnet-ave-yolo26", 8),
        ModelParameterSummary("draxnet-sknet-yolo26", 8),
        ModelParameterSummary("yolo26", 8),
    ]
    assert [call[0]["model"] for call in resolved] == [
        "draxnet-ave-yolo26",
        "draxnet-sknet-yolo26",
        "yolo26",
    ]
    assert all(call[1:] == (True, False) for call in resolved)
    assert all(call[1:] == (None, True) for call in initialized)


def test_detection_listing_propagates_missing_yaml_error(monkeypatch) -> None:
    def missing_yaml(*args, **kwargs):
        raise MLXUserError("Model YAML not found: draxnet-ave-yolo26")

    monkeypatch.setattr(detection_listing, "resolve_model_paths", missing_yaml)

    with pytest.raises(MLXUserError, match="draxnet-ave-yolo26"):
        detection_listing.ListObjectDetectionModels().execute()


@pytest.mark.parametrize(
    ("alias", "filename"),
    [
        ("draxnet-yolo26", "draxnet-ave-yolo26.yaml"),
        ("draxnet-yolo26.yaml", "draxnet-ave-yolo26.yaml"),
        ("draxnet-ave-yolo26", "draxnet-ave-yolo26.yaml"),
        ("draxnet-ave-yolo26.yml", "draxnet-ave-yolo26.yaml"),
        ("draxnet-sknet-yolo26", "draxnet-sknet-yolo26.yaml"),
        ("draxnet-sknet-yolo26.yml", "draxnet-sknet-yolo26.yaml"),
    ],
)
def test_detection_draxnet_aliases_resolve_packaged_yaml(
    monkeypatch,
    tmp_path,
    alias,
    filename,
) -> None:
    config_path = tmp_path / "cfg" / "models" / "ext" / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.touch()
    monkeypatch.setattr(detection_utils, "_ultralytics_package_root", lambda: tmp_path)

    assert detection_utils.resolve_weights_source(alias) == config_path.resolve()


@pytest.mark.parametrize(
    ("runner_module", "runner_name", "config"),
    [
        (
            classification_runner,
            "run_image_classification",
            {"action": "ls-models", "model": "not-a-model"},
        ),
        (
            segmentation_runner,
            "run_segmentation",
            {"action": "ls-models", "model": "not-a-model"},
        ),
        (
            detection_runner,
            "run_object_detection",
            {"action": "ls-models"},
        ),
    ],
)
def test_runners_dispatch_listing_without_selected_model(
    monkeypatch,
    runner_module,
    runner_name,
    config,
) -> None:
    expected = [ModelParameterSummary("listed", 1)]
    monkeypatch.setattr(runner_module, "_list_models", lambda *args: expected)
    if runner_module is classification_runner:
        monkeypatch.setitem(
            classification_runner.ACTION_HANDLERS,
            "ls-models",
            classification_runner._list_models,
        )
    if runner_module is segmentation_runner:
        monkeypatch.setitem(
            segmentation_runner.ACTION_HANDLERS,
            "ls-models",
            segmentation_runner._list_models,
        )

    assert getattr(runner_module, runner_name)(config) == expected
