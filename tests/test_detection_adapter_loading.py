from __future__ import annotations

import copy
import sys
from pickle import UnpicklingError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.commands import CreateObjectDetector
from mlx.modes.object_detection import feature_adapters
from mlx.modes.object_detection.libreyolo import adapter_loading, adapter_targets
from mlx.modes.object_detection.requests import ObjectDetectionRequest


@pytest.fixture
def loading(monkeypatch, tmp_path):
    foundation = tmp_path / "foundation.pt"
    foundation.touch()
    adapter = tmp_path / "adapter.pt"
    adapter.touch()
    artifact = {
        "config": {
            "model": "yolox-l", "method": "lora", "checkpoint_sha256": "a" * 64,
            "injected_modules": ["0"], "adapter_rank": 4, "adapter_alpha": 2.0,
        },
        "state": {"adapter.weight": torch.ones(1)},
    }
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1))
    calls = {}

    def inject(raw, method, targets, **kwargs):
        calls["injection"] = (raw, method, targets, kwargs)

    def restore(raw, state):
        calls["state"] = state

    def verify(name, path):
        calls["foundation"] = (name, path)
        return SimpleNamespace(execute=lambda: (model, {"sha256": "a" * 64}))

    def wrap(raw, info, device):
        calls["device"] = device
        return SimpleNamespace(model=raw)

    monkeypatch.setattr(feature_adapters, "inject_adapters", inject)
    monkeypatch.setattr(feature_adapters, "load_adapter_state_dict", restore)
    monkeypatch.setattr(adapter_targets, "yolox_targets", lambda *args, **kwargs: {"0": 3})
    monkeypatch.setitem(sys.modules, "libreyolo.utils.serialization", SimpleNamespace(
        load_untrusted_torch_file=lambda *args, **kwargs: artifact,
    ))
    monkeypatch.setattr(adapter_loading, "VerifyFoundationCheckpoint", verify)
    monkeypatch.setattr(adapter_loading, "build_experimental_yolox", wrap)
    return SimpleNamespace(foundation=foundation, adapter=adapter, artifact=artifact,
                           model=model, calls=calls)


def test_standalone_adapter_restores_saved_configuration(loading):
    original = copy.deepcopy(loading.artifact["config"])
    wrapper = adapter_loading.LoadAdaptedYOLOX(
        loading.foundation, loading.adapter, device="cpu"
    ).execute()
    assert wrapper.model is loading.model
    assert not wrapper.model.training
    assert loading.calls["foundation"] == ("yolox-l", loading.foundation)
    raw, method, targets, kwargs = loading.calls["injection"]
    assert raw is loading.model
    assert method == "lora"
    assert targets == {"0": 3}
    assert kwargs == dict(reduction=8, rank=4, alpha=2.0, train_head=False, registry=None)
    assert loading.calls["state"] is loading.artifact["state"]
    assert loading.artifact["config"] == original


@pytest.mark.parametrize("change, message", [
    ({"checkpoint_sha256": "b" * 64}, "checksum does not match"),
    ({"checkpoint_sha256": None}, "checksum"),
    ({"model": "yolox-drax-csp-m"}, "standard YOLOX"),
    ({"model": []}, "standard YOLOX"),
    ({"method": None}, "method"),
    ({"injected_modules": ["missing"]}, "injection path"),
    ({"injected_modules": ["0", "0"]}, "unique"),
    ({"adapter_rank": 0}, "positive"),
    ({"adapter_alpha": float("nan")}, "finite"),
    ({"train_head": "False"}, "boolean"),
    ({"checkpoint_sha256": "z" * 64}, "checksum"),
])
def test_invalid_adapter_fails_before_tensor_restoration(loading, change, message):
    loading.artifact["config"].update(change)
    with pytest.raises(MLXUserError, match=message):
        adapter_loading.LoadAdaptedYOLOX(loading.foundation, loading.adapter).execute()
    assert "state" not in loading.calls


@pytest.mark.parametrize("key", ["config", "state"])
def test_missing_artifact_sections_are_actionable(loading, key):
    del loading.artifact[key]
    with pytest.raises(MLXUserError, match="config and state"):
        adapter_loading.LoadAdaptedYOLOX(loading.foundation, loading.adapter).execute()


def test_missing_adapter_and_onnx_foundation_fail_clearly(loading):
    with pytest.raises(MLXUserError, match="not found"):
        adapter_loading.LoadAdaptedYOLOX(loading.foundation, loading.adapter.with_name("missing.pt")).execute()
    with pytest.raises(MLXUserError, match="foundation .pt"):
        adapter_loading.LoadAdaptedYOLOX(loading.foundation.with_suffix(".onnx"), loading.adapter).execute()


@pytest.mark.parametrize("exception", [ValueError, EOFError, UnpicklingError])
def test_corrupt_adapter_is_wrapped(loading, monkeypatch, exception):
    def fail(*args, **kwargs):
        raise exception("invalid checkpoint")
    monkeypatch.setattr(sys.modules["libreyolo.utils.serialization"], "load_untrusted_torch_file", fail)
    with pytest.raises(MLXUserError, match="Cannot read adapter") as error:
        adapter_loading.LoadAdaptedYOLOX(loading.foundation, loading.adapter).execute()
    assert isinstance(error.value.__cause__, exception)


@pytest.mark.parametrize("method", ["bottleneck", "conv-adapter", "convpass", "drax", "ssf", "drax-hybrid", "drax-residual-fusion"])
def test_other_methods_restore_without_cli_hyperparameters(loading, method):
    loading.artifact["config"].update(method=method, train_head=True, adapter_reduction=16)
    adapter_loading.LoadAdaptedYOLOX(loading.foundation, loading.adapter).execute()
    _, actual_method, targets, kwargs = loading.calls["injection"]
    assert actual_method == method
    assert targets == {"0": 3}
    assert kwargs["train_head"] is True
    assert kwargs["reduction"] == 16


def test_changed_feature_topology_is_rejected(loading):
    loading.artifact["config"].update(method="drax", injected_modules=["changed"])
    with pytest.raises(MLXUserError, match="target policy"):
        adapter_loading.LoadAdaptedYOLOX(loading.foundation, loading.adapter).execute()
    assert "state" not in loading.calls


def test_strict_state_failure_preserves_cause(loading, monkeypatch):
    def fail(*args):
        raise ValueError("Adapter state mismatch")
    monkeypatch.setattr(feature_adapters, "load_adapter_state_dict", fail)
    with pytest.raises(MLXUserError, match="Adapter state mismatch") as error:
        adapter_loading.LoadAdaptedYOLOX(loading.foundation, loading.adapter).execute()
    assert isinstance(error.value.__cause__, ValueError)


def test_cli_adapter_reaches_frame_prediction(loading, monkeypatch):
    received = {}

    class LoadModel:
        def __init__(self, model_path, adapter_path, *, device):
            received.update(model_path=model_path, adapter_path=adapter_path, device=device)

        def execute(self):
            return SimpleNamespace(predict=lambda *args, **kwargs: SimpleNamespace(names={0: "car"}, boxes=None))

    monkeypatch.setattr(adapter_loading, "LoadAdaptedYOLOX", LoadModel)
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=None))
    parsed = build_parser().parse_args([
        "--mode", "object_detection", "--action", "infer-video", "--provider", "libreyolo",
        "--model-path", str(loading.foundation), "--adapter", str(loading.adapter),
    ])
    request = ObjectDetectionRequest.from_config(vars(parsed))
    detector = CreateObjectDetector(request).execute()
    assert received == dict(model_path=loading.foundation, adapter_path=str(loading.adapter), device="cpu")
    result = detector.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.names == {0: "car"}
    assert result.detections == ()


def test_ultralytics_rejects_separate_adapter():
    with pytest.raises(MLXUserError, match="--provider libreyolo"):
        CreateObjectDetector(ObjectDetectionRequest(adapter="adapter.pt")).execute()
