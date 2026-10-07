"""Inference startup diagnostics at the PyTorch integration boundary."""

from mlx.core.commands import NullWorkflowReporter, emit
from mlx.core.exceptions import MLXUserError


class ReportInferenceDevice:
    def __init__(self, device, *, reporter=None, torch_module=None):
        self.device = str(device)
        self.reporter = reporter or NullWorkflowReporter()
        self.torch_module = torch_module

    def execute(self):
        normalized = self.device.strip().lower()
        if not (normalized.startswith("cuda") or normalized.isdecimal()):
            return None
        device = "cuda:0" if normalized == "cuda" else (
            f"cuda:{normalized}" if normalized.isdecimal() else normalized
        )
        torch = self.torch_module
        if torch is None:
            import torch
        details = {"event": "inference_device", "requested_device": device,
                   "available": torch.cuda.is_available()}
        if details["available"]:
            try:
                details["gpu_name"] = torch.cuda.get_device_name(device)
            except (ValueError, RuntimeError, AssertionError) as exc:
                raise MLXUserError(f"GPU device {self.device!r} is unavailable. Check --device and the installed PyTorch runtime.") from exc
            hip = torch.version.hip
            details["backend"] = "ROCm" if hip else "CUDA"
            details["runtime_version"] = hip or torch.version.cuda
        emit(self.reporter, "info", "Requested inference GPU device.", payload=details)
        return details
