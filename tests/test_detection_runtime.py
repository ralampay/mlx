"""Portable GPU startup and scoped provider-diagnostic contracts."""

import logging
from types import SimpleNamespace

import pytest

from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.libreyolo.dependencies import quiet_unused_attention_imports
from mlx.modes.object_detection.runtime import ReportInferenceDevice


@pytest.mark.parametrize("device,hip,cuda,expected", [
    ("cuda:0", "7.0", None, "ROCm"),
    ("cuda", None, "12.0", "CUDA"),
    ("0", "7.0", None, "ROCm"),
])
def test_startup_reports_requested_gpu_and_runtime(device, hip, cuda, expected):
    events = []
    requested = []
    torch = SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda d: requested.append(d) or "Test GPU"),
    )
    result = ReportInferenceDevice(device, reporter=CallbackWorkflowReporter(events.append), torch_module=torch).execute()
    assert requested == ["cuda:0"]
    assert result["requested_device"] == "cuda:0"
    assert result["backend"] == expected
    assert result["gpu_name"] == "Test GPU"
    assert result["runtime_version"] == (hip or cuda)
    assert events[0].payload == result


def test_unavailable_gpu_is_reported_without_claiming_acceleration():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    result = ReportInferenceDevice("cuda:0", torch_module=torch).execute()
    assert not result["available"]
    assert "gpu_name" not in result


def test_cpu_diagnostics_do_not_touch_pytorch():
    assert ReportInferenceDevice("cpu", torch_module=object()).execute() is None


def test_invalid_gpu_index_is_actionable():
    def fail(device):
        raise AssertionError("bad index")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=fail))
    with pytest.raises(MLXUserError, match="Check --device") as error:
        ReportInferenceDevice("cuda:99", torch_module=torch).execute()
    assert isinstance(error.value.__cause__, AssertionError)


def test_gpu_startup_presentation_identifies_rocm(monkeypatch):
    from mlx.core.commands import WorkflowEvent
    from mlx.modes.object_detection import presentation

    messages = []
    monkeypatch.setattr(presentation, "print_info", messages.append)
    presentation.RichWorkflowReporter().emit(WorkflowEvent("info", "device", payload={
        "event": "inference_device", "requested_device": "cuda:0", "available": True,
        "backend": "ROCm", "runtime_version": "7.0", "gpu_name": "Test GPU",
    }))
    assert messages == ["Requested inference device: cuda:0 | PyTorch backend: ROCm 7.0 | GPU: Test GPU"]


@pytest.mark.parametrize("fail", [False, True])
def test_optional_import_notice_filter_is_precise_and_restored(caplog, fail):
    logger = logging.getLogger("dinov2")
    original = list(logger.filters)
    try:
        with quiet_unused_attention_imports():
            logger.warning("xFormers not available")
            logger.warning("Other diagnostic")
            logging.getLogger("other-provider").warning("xFormers not available")
            if fail:
                raise RuntimeError("load failed")
    except RuntimeError:
        assert fail
    assert logger.filters == original
    assert [record.message for record in caplog.records] == ["Other diagnostic", "xFormers not available"]
    logger.warning("xFormers not available")
    assert caplog.records[-1].name == "dinov2"
