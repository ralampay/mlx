"""Inference routing contracts for embedded adapter detector checkpoints."""

from pickle import UnpicklingError
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.commands import CreateObjectDetector
from mlx.modes.object_detection.libreyolo import adapter_checkpoint, adapter_loading
from mlx.modes.object_detection.libreyolo.detector_loading import LoadLibreYOLODetector
from mlx.modes.object_detection.requests import ObjectDetectionRequest


def test_cli_model_path_restores_embedded_adapter_and_predicts(tmp_path, monkeypatch):
    path = tmp_path / "arbitrary-name.pt"
    torch.save({"format": adapter_checkpoint.FORMAT}, path)
    calls = {}
    registry = object()

    class Loader:
        def __init__(self, checkpoint, **kwargs):
            calls.update(checkpoint=checkpoint, **kwargs)

        def execute(self):
            def predict(frame, **kwargs):
                calls["prediction"] = kwargs
                return [SimpleNamespace(names={0: "object"}, boxes=None)]
            return SimpleNamespace(predict=predict)

    def fail_provider(*args, **kwargs):
        pytest.fail("Embedded adapter must not reach the native LibreYOLO loader")

    monkeypatch.setattr(adapter_checkpoint, "LoadAdapterDetector", Loader)
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=fail_provider))
    parsed = build_parser().parse_args([
        "--mode", "object_detection", "--action", "infer-video",
        "--provider", "libreyolo", "--model-path", str(path), "--device", "cuda:0",
    ])
    from mlx.modes.object_detection.libreyolo.provider import LibreYOLOProvider

    detector = CreateObjectDetector(
        ObjectDetectionRequest.from_config(vars(parsed)),
        provider=LibreYOLOProvider(registry=registry),
    ).execute()
    result = detector.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert calls["checkpoint"] == path
    assert calls["device"] == "cuda:0"
    assert calls["registry"] is registry
    assert calls["payload"] == {"format": adapter_checkpoint.FORMAT}
    assert calls["prediction"]["device"] == "cuda:0"
    assert result.names == {0: "object"}
    assert result.detections == ()


@pytest.mark.parametrize("suffix", [".pt", ".onnx"])
def test_ordinary_models_keep_native_loading(tmp_path, monkeypatch, suffix):
    path = tmp_path / f"adapter-in-filename{suffix}"
    if suffix == ".pt":
        torch.save({"model": {"weight": torch.ones(1)}}, path)
    else:
        def fail_inspection(*args, **kwargs):
            pytest.fail("ONNX inference must not inspect a PyTorch checkpoint")
        monkeypatch.setattr(torch, "load", fail_inspection)
    model = object()
    calls = []

    def native(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=native))
    assert LoadLibreYOLODetector(path, device="cpu").execute() is model
    assert calls == [(str(path), {"device": "cpu", "task": "detect"})]


def test_explicit_adapter_keeps_foundation_loading(tmp_path, monkeypatch):
    path = tmp_path / "foundation.pt"
    torch.save({"model": {}}, path)
    calls = {}

    def separate(checkpoint, adapter_path, **kwargs):
        calls.update(checkpoint=checkpoint, adapter_path=adapter_path, **kwargs)
        return SimpleNamespace(execute=lambda: "adapted")

    monkeypatch.setattr(adapter_loading, "LoadAdaptedYOLOX", separate)
    result = LoadLibreYOLODetector(path, adapter_path="adapter.pt").execute()
    assert result == "adapted"
    assert calls == dict(checkpoint=path, adapter_path="adapter.pt", device="cpu")


@pytest.mark.parametrize("payload,adapter_path,message", [
    ({"format": adapter_checkpoint.FORMAT}, "adapter.pt", "Omit --adapter"),
    ({"format": "mlx-yolox-adapter-detector-v99"}, None, "Unsupported adapter detector format"),
    ({"format": adapter_checkpoint.FORMAT}, None, "Cannot load adapter detector"),
])
def test_invalid_embedded_requests_are_actionable(tmp_path, payload, adapter_path, message):
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    with pytest.raises(MLXUserError, match=message):
        CreateObjectDetector(ObjectDetectionRequest(
            provider="libreyolo", model_path=str(path), adapter=adapter_path,
        )).execute()


@pytest.mark.parametrize("exception", [EOFError, UnpicklingError, RuntimeError])
def test_inspection_failure_delegates_native_compatibility(tmp_path, monkeypatch, exception):
    def fail_read(*args, **kwargs):
        raise exception("provider-specific checkpoint")

    monkeypatch.setattr(torch, "load", fail_read)
    model = object()
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **kw: model))
    assert LoadLibreYOLODetector(tmp_path / "legacy.pt").execute() is model


@pytest.mark.parametrize("exception", [EOFError, UnpicklingError])
def test_direct_embedded_read_failure_preserves_cause(tmp_path, monkeypatch, exception):
    def fail_read(*args, **kwargs):
        raise exception("corrupt checkpoint")

    monkeypatch.setattr(torch, "load", fail_read)
    with pytest.raises(MLXUserError, match="Cannot load adapter detector") as error:
        adapter_checkpoint.LoadAdapterDetector(tmp_path / "model.pt").execute()
    assert isinstance(error.value.__cause__, exception)


@pytest.mark.parametrize("change,message", [
    ({"names": "object"}, "list of class labels"),
    ({"names": []}, "list of class labels"),
    ({"names": [None]}, "list of class labels"),
    ({"config": []}, "config must be a mapping"),
    ({"config": {"train_head": "False"}}, "train_head must be a boolean"),
])
def test_embedded_metadata_validation(tmp_path, change, message):
    payload = {"format": adapter_checkpoint.FORMAT, "names": ["object"], "config": {}}
    payload.update(change)
    with pytest.raises(MLXUserError, match=message):
        adapter_checkpoint.LoadAdapterDetector(tmp_path / "model.pt", payload=payload).execute()


@pytest.mark.parametrize("setting", ["adapter_reduction", "adapter_rank"])
def test_zero_adapter_settings_are_rejected_instead_of_defaulted(tmp_path, setting):
    payload = {
        "format": adapter_checkpoint.FORMAT, "size": "n", "names": ["object"],
        "model": {}, "config": {"method": "lora", setting: 0},
    }
    with pytest.raises(MLXUserError, match="must be positive"):
        adapter_checkpoint.LoadAdapterDetector(tmp_path / "model.pt", payload=payload).execute()


def test_provider_load_failure_remains_actionable(tmp_path, monkeypatch):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"invalid")

    def native(*args, **kwargs):
        raise RuntimeError("invalid native checkpoint")

    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=native))
    with pytest.raises(MLXUserError, match="Failed to load LibreYOLO detection model") as error:
        CreateObjectDetector(ObjectDetectionRequest(provider="libreyolo", model_path=str(path))).execute()
    assert isinstance(error.value.__cause__, RuntimeError)
