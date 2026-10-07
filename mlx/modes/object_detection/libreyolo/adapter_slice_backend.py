"""LibreYOLO model reconstruction and prediction cache integration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.evaluation import normalize_detection_metrics
from mlx.modes.object_detection.libreyolo.adapter_backend import (
    VerifyFoundationCheckpoint,
    build_experimental_yolox,
)
from mlx.modes.object_detection.libreyolo.adapter_loading import (
    ApplyYOLOXAdapter,
    resolve_adapter_targets,
)


class LibreYOLOAdapterPredictionWriter:
    """Reconstruct one study checkpoint and cache native COCO predictions."""

    def __init__(self, request: Any, foundation: Mapping[str, Any], device: torch.device, *, registry=None):
        self.registry = registry
        self.request = request
        self.foundation = foundation
        self.device = device

    def __call__(self, run: Any, split: str, destination: Path) -> Mapping[str, Any]:
        wrapper = ReconstructAdapterModel(
            self.request.checkpoint, self.foundation, self.device, registry=self.registry
        ).execute(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="slice-eval-", dir=self.request.analysis_root
        ) as temporary:
            native = wrapper.val(
                data=str(self.request.dataset / "data.yaml"), split=split,
                imgsz=self.request.image_size, batch=self.request.batch_size,
                device=str(self.device), workers=self.request.workers,
                conf=0.001, iou=0.6, verbose=False, save_json=True,
                save_plots=False, save_dir=temporary,
            )
            generated = Path(temporary) / "predictions.json"
            if not generated.is_file():
                matches = list(Path(temporary).rglob("predictions.json"))
                if len(matches) != 1:
                    raise MLXUserError("LibreYOLO did not produce exactly one predictions.json")
                generated = matches[0]
            try:
                predictions = json.loads(generated.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MLXUserError(f"Cannot read LibreYOLO predictions at {generated}: {exc}") from exc
            write_json_atomic(destination, predictions)
        del wrapper
        torch.cuda.empty_cache()
        return normalize_detection_metrics(native)

    @staticmethod
    def _adapter_targets(model, config, method):
        return ReconstructAdapterModel._adapter_targets(model, config, method)

    @staticmethod
    def _restore_dense_checkpoint(model, run, loader):
        return ReconstructAdapterModel._restore_dense_checkpoint(model, run, loader)


class ReconstructAdapterModel:
    """Strict foundation-first restoration shared by post-training evaluations."""

    def __init__(self, checkpoint, foundation, device, *, registry=None):
        self.registry = registry
        self.checkpoint = Path(checkpoint)
        self.foundation = foundation
        self.device = device

    def execute(self, run):
        from libreyolo.utils.serialization import load_untrusted_torch_file

        model_name = f"yolox-{self.foundation['size']}"
        model, info = VerifyFoundationCheckpoint(model_name, self.checkpoint).execute()
        if info["sha256"] != self.foundation["sha256"]:
            raise MLXUserError("Foundation checkpoint changed during slice prediction generation")
        method = run.method
        if method not in {"frozen", "head-only", "full-finetune"}:
            config = run.config
            artifact = load_untrusted_torch_file(
                str(run.directory / "adapter" / "checkpoint.pt"), map_location="cpu"
            )
            ApplyYOLOXAdapter(model, {**config, "method": method}, artifact["state"], registry=self.registry).execute()
        elif method in {"head-only", "full-finetune"}:
            self._restore_dense_checkpoint(model, run, load_untrusted_torch_file)

        model.to(self.device).eval()
        wrapper = build_experimental_yolox(model, info, str(self.device))
        return wrapper

    @staticmethod
    def _adapter_targets(model, config, method):
        return resolve_adapter_targets(model, config, method)

    @staticmethod
    def _restore_dense_checkpoint(model, run, loader) -> None:
        selected = Path(str(run.config.get("selected_checkpoint_path") or ""))
        checkpoint = loader(str(selected), map_location="cpu")
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise MLXUserError(f"Selected checkpoint has no model state: {selected}")
        if run.method == "full-finetune":
            loaded = model.load_state_dict(state, strict=True)
            if loaded.missing_keys or loaded.unexpected_keys:
                raise MLXUserError(f"Full-finetune checkpoint did not load strictly: {selected}")
            return
        foundation_state = model.state_dict()
        head_names = {name for name in foundation_state if name.startswith("head.")}
        if not head_names or any(name not in state for name in head_names):
            raise MLXUserError(f"Head-only checkpoint is incomplete: {selected}")
        selected_state = {
            name: state[name] if name in head_names else value
            for name, value in foundation_state.items()
        }
        loaded = model.load_state_dict(selected_state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise MLXUserError(f"Head-only checkpoint did not load strictly: {selected}")


__all__ = ["LibreYOLOAdapterPredictionWriter"]
