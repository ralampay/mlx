from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

from mlx.core.exceptions import MLXUserError
from mlx.core.model_listing import ModelParameterSummary
from mlx.modes.object_detection.commands import CreateObjectDetector
from mlx.modes.object_detection.libreyolo.adapters import result_to_detection_result
from mlx.modes.object_detection.libreyolo.conversion import (
    ConvertLibreYOLOObjectDetectionModel,
)
from mlx.modes.object_detection.libreyolo.list_models import ListLibreYOLOModels
from mlx.modes.object_detection.libreyolo.training import TrainLibreYOLOObjectDetection
from mlx.modes.object_detection.providers import PROVIDER_REGISTRY
from mlx.modes.object_detection.requests import ObjectDetectionRequest


def _install_fake_libreyolo(monkeypatch, *, LibreYOLO, LibreYOLO9=None) -> None:
    class FakeDraxConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module = SimpleNamespace(LibreYOLO=LibreYOLO)
    if LibreYOLO9 is not None:
        module.LibreYOLO9 = LibreYOLO9
    monkeypatch.setitem(sys.modules, "libreyolo", module)
    monkeypatch.setitem(
        sys.modules,
        "libreyolo.models.yolo9",
        SimpleNamespace(DraxConfig=FakeDraxConfig),
    )


def test_libreyolo_provider_is_registered_lazily() -> None:
    assert PROVIDER_REGISTRY["libreyolo"] == (
        "mlx.modes.object_detection.libreyolo.provider:get_provider"
    )


def test_dependency_metadata_uses_ralampay_release_fork() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    extras = project["optional-dependencies"]
    fork_reference = (
        "libreyolo[onnx] @ git+https://github.com/ralampay/libreyolo.git@release"
    )

    assert project["requires-python"] == ">=3.10"
    assert fork_reference in extras["object-detection-libreyolo"]
    assert fork_reference in extras["object-detection"]
    assert all("libreyolo" not in dependency.lower() for dependency in project["dependencies"])

    all_metadata = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "github.com/LibreYOLO/libreyolo" not in all_metadata
    assert fork_reference in requirements


def test_libreyolo_result_decodes_normalized_detections() -> None:
    result = SimpleNamespace(
        names=["person", "bike"],
        boxes=SimpleNamespace(
            xyxy=np.array([[1.2, 2.9, 30.8, 40.1]], dtype=np.float32),
            conf=np.array([0.75], dtype=np.float32),
            cls=np.array([1], dtype=np.float32),
        ),
    )

    decoded = result_to_detection_result(result)

    assert decoded.names == {0: "person", 1: "bike"}
    assert len(decoded.detections) == 1
    assert decoded.detections[0].xyxy == pytest.approx((1.2, 2.9, 30.8, 40.1))
    assert decoded.detections[0].confidence == pytest.approx(0.75)
    assert decoded.detections[0].class_id == 1
    assert decoded.detections[0].label == "bike"


@pytest.mark.parametrize(
    "boxes",
    [
        None,
        SimpleNamespace(
            xyxy=np.empty((0, 4)),
            conf=np.empty((0,)),
            cls=np.empty((0,)),
        ),
    ],
)
def test_libreyolo_result_supports_empty_detections(boxes) -> None:
    decoded = result_to_detection_result(SimpleNamespace(names={0: "thing"}, boxes=boxes))

    assert decoded.detections == ()
    assert decoded.names == {0: "thing"}


def test_libreyolo_detector_loads_checkpoint_and_predicts(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    calls = []

    class FakeModel:
        def predict(self, frame, **kwargs):
            calls.append((frame, kwargs))
            return [
                SimpleNamespace(
                    names={0: "object"},
                    boxes=SimpleNamespace(
                        xyxy=np.array([[0, 1, 2, 3]]),
                        conf=np.array([0.9]),
                        cls=np.array([0]),
                    ),
                )
            ]

    def fake_factory(model_path, **kwargs):
        assert model_path == str(checkpoint.resolve())
        assert kwargs == {"device": "cpu", "task": "detect"}
        return FakeModel()

    _install_fake_libreyolo(monkeypatch, LibreYOLO=fake_factory)
    detector = CreateObjectDetector(
        ObjectDetectionRequest(provider="libreyolo", model_path=str(checkpoint))
    ).execute()
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    result = detector.predict(frame)

    assert result.detections[0].label == "object"
    assert calls[0][0] is frame
    assert calls[0][1]["color_format"] == "bgr"
    assert calls[0][1]["stream"] is False


def test_libreyolo_training_maps_options_and_selects_best(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: {0: thing}\n", encoding="utf-8")
    project = tmp_path / "runs"
    constructor_calls = []
    train_calls = []

    class FakeYOLO9:
        def __init__(self, **kwargs):
            constructor_calls.append(kwargs)

        def train(self, **kwargs):
            train_calls.append(kwargs)
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            best = weights / "best.pt"
            last = weights / "last.pt"
            best.touch()
            last.touch()
            return {
                "save_dir": str(save_dir),
                "best_checkpoint": str(best),
                "last_checkpoint": str(last),
                "best_mAP50_95": 0.4,
            }

    _install_fake_libreyolo(
        monkeypatch,
        LibreYOLO=lambda *args, **kwargs: pytest.fail("checkpoint loader was not expected"),
        LibreYOLO9=FakeYOLO9,
    )
    results = TrainLibreYOLOObjectDetection(
        {
            "model": "yolo9-t",
            "dataset_path": str(dataset),
            "output_path": str(project),
            "run_name": "trial",
            "epochs": 2,
            "batch_size": 3,
            "height": 320,
            "width": 512,
            "device": "cpu",
            "optimizer": "auto",
            "nbs": 32,
            "warmup_epochs": 2,
            "amp": False,
            "plots": True,
            "pretrained": True,
            "random_seed": 7,
            "use_best": True,
        }
    ).execute()

    assert constructor_calls == [
        {"model_path": None, "size": "t", "device": "cpu", "task": "detect"}
    ]
    assert train_calls[0]["data"] == str((dataset / "data.yaml").resolve())
    assert train_calls[0]["imgsz"] == (320, 512)
    assert train_calls[0]["pretrained"] is True
    assert train_calls[0]["nbs"] == 32
    assert train_calls[0]["save_plots"] is True
    assert train_calls[0]["seed"] == 7
    assert "optimizer" not in train_calls[0]
    assert results["model_path"].endswith("trial/weights/best.pt")
    assert results["checkpoint_path"] == results["model_path"]


def test_libreyolo_training_rejects_loss_clip() -> None:
    with pytest.raises(MLXUserError, match="loss-clip is not supported"):
        TrainLibreYOLOObjectDetection({"loss_clip": 1.0}).execute()


def test_libreyolo_training_auto_resumes_existing_last_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: {0: thing}\n", encoding="utf-8")
    project = tmp_path / "runs"
    weights = project / "trial" / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.touch()
    load_calls = []
    train_calls = []

    class FakeLoadedModel:
        def train(self, **kwargs):
            train_calls.append(kwargs)
            return {"save_dir": str(project / "trial"), "last_checkpoint": str(last)}

    def fake_factory(model_path, **kwargs):
        load_calls.append((model_path, kwargs))
        return FakeLoadedModel()

    _install_fake_libreyolo(
        monkeypatch,
        LibreYOLO=fake_factory,
        LibreYOLO9=lambda **kwargs: pytest.fail("scratch model was not expected"),
    )
    results = TrainLibreYOLOObjectDetection(
        {
            "model": "yolo9-t",
            "dataset_path": str(dataset),
            "output_path": str(project),
            "run_name": "trial",
            "device": "cpu",
            "use_best": False,
        }
    ).execute()

    assert load_calls == [(str(last.resolve()), {"device": "cpu", "task": "detect"})]
    assert train_calls[0]["resume"] is True
    assert "pretrained" not in train_calls[0]
    assert results["checkpoint_path"] == str(last.resolve())


def test_libreyolo_training_rejects_non_pt_warm_start(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "model.onnx"
    checkpoint.touch()
    _install_fake_libreyolo(
        monkeypatch,
        LibreYOLO=lambda *args, **kwargs: None,
        LibreYOLO9=lambda *args, **kwargs: None,
    )

    with pytest.raises(MLXUserError, match="PyTorch checkpoint"):
        TrainLibreYOLOObjectDetection(
            {"model": "yolo9-t", "model_path": str(checkpoint)}
        ).execute()


def test_libreyolo_conversion_moves_export_to_requested_target(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    generated = tmp_path / "generated.onnx"
    destination = tmp_path / "exports" / "detector.onnx"
    export_calls = []

    class FakeModel:
        def export(self, **kwargs):
            export_calls.append(kwargs)
            generated.touch()
            return str(generated)

    _install_fake_libreyolo(
        monkeypatch,
        LibreYOLO=lambda *args, **kwargs: FakeModel(),
    )
    result = ConvertLibreYOLOObjectDetectionModel(
        {
            "model_path": str(checkpoint),
            "output_path": str(destination),
            "height": 640,
            "width": 640,
            "device": "cpu",
        }
    ).execute()

    assert result == destination.resolve()
    assert destination.exists()
    assert not generated.exists()
    assert export_calls == [{"format": "onnx", "imgsz": 640, "device": "cpu"}]


def test_libreyolo_listing_builds_canonical_yolo9_configurations(monkeypatch) -> None:
    calls = []

    class FakeYOLO9:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.model = nn.Linear(3, 2)

    _install_fake_libreyolo(
        monkeypatch,
        LibreYOLO=lambda *args, **kwargs: None,
        LibreYOLO9=FakeYOLO9,
    )

    summaries = ListLibreYOLOModels().execute()

    assert summaries == [
        ModelParameterSummary("yolo9-t", 8),
        ModelParameterSummary("yolo9-s", 8),
        ModelParameterSummary("yolo9-m", 8),
        ModelParameterSummary("yolo9-c", 8),
        ModelParameterSummary("yolo9-s-drax-b5", 8),
    ]
    assert [call["size"] for call in calls] == ["t", "s", "m", "c", "s"]
    assert all("drax_config" not in call for call in calls[:-1])
    drax_config = calls[-1]["drax_config"]
    assert drax_config.enabled is True
    assert drax_config.stages == ("b5",)
    assert drax_config.use_attention is True
    assert drax_config.efficient is True
    assert drax_config.fusion_mode == "average"
    assert drax_config.drop_path == 0.0


def test_libreyolo_training_builds_documented_drax_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: {0: thing}\n", encoding="utf-8")
    constructor_calls = []

    class FakeYOLO9:
        def __init__(self, **kwargs):
            constructor_calls.append(kwargs)

        def train(self, **kwargs):
            return {}

    _install_fake_libreyolo(
        monkeypatch,
        LibreYOLO=lambda *args, **kwargs: pytest.fail("checkpoint loader was not expected"),
        LibreYOLO9=FakeYOLO9,
    )

    TrainLibreYOLOObjectDetection(
        {
            "model": "yolo9-s-drax-b5",
            "dataset_path": str(dataset),
            "output_path": str(tmp_path / "runs"),
            "device": "cpu",
        }
    ).execute()

    assert len(constructor_calls) == 1
    call = constructor_calls[0]
    assert call["size"] == "s"
    drax_config = call["drax_config"]
    assert drax_config.enabled is True
    assert drax_config.stages == ("b5",)
    assert drax_config.use_attention is True
    assert drax_config.efficient is True
    assert drax_config.fusion_mode == "average"
    assert drax_config.drop_path == 0.0


def test_installed_libreyolo_can_construct_scratch_yolo9_t() -> None:
    libreyolo = pytest.importorskip("libreyolo")

    model = libreyolo.LibreYOLO9(
        model_path=None,
        size="t",
        device="cpu",
        task="detect",
    )

    assert sum(parameter.numel() for parameter in model.model.parameters()) > 0
