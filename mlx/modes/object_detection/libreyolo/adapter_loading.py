"""Foundation-first restoration of standalone YOLOX feature adapters."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from pickle import UnpicklingError

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.libreyolo.adapter_backend import (
    MODEL_SIZES,
    VerifyFoundationCheckpoint,
    build_experimental_yolox,
)


def resolve_adapter_targets(model, config, method):
    from mlx.modes.object_detection.libreyolo.adapter_targets import yolox_targets

    paths = config.get("injected_modules")
    if "injected_modules" in config:
        if (not isinstance(paths, list) or not paths
                or any(not isinstance(path, str) or not path for path in paths)
                or len(set(paths)) != len(paths)):
            raise MLXUserError(f"{method.title()} checkpoint requires unique, nonempty injected_modules paths")
    # Older hybrids used three convolutions rather than the current LoRA policy.
    if method in {"drax-hybrid", "drax-residual-fusion", "lora"} and paths is not None:
        try:
            return {path: model.get_submodule(path).out_channels for path in paths}
        except AttributeError as exc:
            raise MLXUserError(f"Invalid {method.title()} checkpoint injection path: {exc}") from exc
    targets = yolox_targets(model, str(config.get("adapter_target") or "neck"), method)
    if paths is not None and set(paths) != set(targets):
        raise MLXUserError("Recorded adapter injection paths do not match the YOLOX target policy")
    return targets


class ApplyYOLOXAdapter:
    """Attach and strictly restore adapter tensors on an already verified model."""

    def __init__(self, model, config, state):
        self.model = model
        self.config = config
        self.state = state

    def execute(self):
        from mlx.modes.object_detection.feature_adapters import inject_adapters, load_adapter_state_dict

        try:
            method = self.config["method"]
            targets = resolve_adapter_targets(self.model, self.config, method)
            reduction = self.config.get("adapter_reduction")
            rank = self.config.get("adapter_rank")
            reduction = int(8 if reduction is None else reduction)
            rank = int(8 if rank is None else rank)
            alpha = float(self.config.get("adapter_alpha", 1.0))
            train_head = self.config.get("train_head", False)
            if not isinstance(train_head, bool):
                raise ValueError("train_head must be a boolean")
            if reduction < 1 or rank < 1 or not math.isfinite(alpha):
                raise ValueError("adapter reduction/rank must be positive and alpha finite")
            if not isinstance(self.state, Mapping):
                raise ValueError("adapter state must be a tensor mapping")
            inject_adapters(
                self.model, method, targets, reduction=reduction, rank=rank,
                alpha=alpha, train_head=train_head,
            )
            load_adapter_state_dict(self.model, self.state)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise MLXUserError(f"Cannot restore YOLOX adapter: {exc}. Inspect its configuration and tensor state.") from exc
        return self.model


class LoadAdaptedYOLOX:
    """Load a standalone adapter with its exact foundation, independent of a study directory."""

    def __init__(self, model_path, adapter_path, *, device="cpu"):
        self.model_path = Path(model_path).expanduser()
        self.adapter_path = Path(adapter_path).expanduser()
        self.device = device

    def execute(self):
        if self.model_path.suffix.lower() != ".pt":
            raise MLXUserError("Detection --adapter requires a YOLOX foundation .pt checkpoint, not an ONNX model.")
        artifact = self._read_artifact()
        config = self._validate_config(artifact)
        model, info = VerifyFoundationCheckpoint(config["model"], self.model_path).execute()
        if info["sha256"] != config["checkpoint_sha256"]:
            raise MLXUserError("Adapter foundation checksum does not match --model-path. Supply the exact foundation used to train this adapter.")
        ApplyYOLOXAdapter(model, config, artifact["state"]).execute()
        model.eval()
        return build_experimental_yolox(model, info, self.device)

    def _read_artifact(self):
        from libreyolo.utils.serialization import load_untrusted_torch_file

        if not self.adapter_path.is_file():
            raise MLXUserError(f"Adapter checkpoint not found: {self.adapter_path}")
        try:
            return load_untrusted_torch_file(str(self.adapter_path), map_location="cpu")
        except (OSError, ValueError, RuntimeError, TypeError, EOFError, UnpicklingError) as exc:
            raise MLXUserError(f"Cannot read adapter checkpoint {self.adapter_path}: {exc}") from exc

    @staticmethod
    def _validate_config(artifact):
        config = artifact.get("config") if isinstance(artifact, Mapping) else None
        if not isinstance(config, Mapping) or "state" not in artifact:
            raise MLXUserError("Adapter checkpoint must contain config and state mappings from a YOLOX adapter experiment.")
        model_name = config.get("model")
        if not isinstance(model_name, str) or model_name not in MODEL_SIZES:
            raise MLXUserError("Adapter config must identify a standard YOLOX model (for example yolox-l).")
        if not isinstance(config.get("method"), str) or not config["method"]:
            raise MLXUserError("Adapter config is missing its method.")
        digest = config.get("checkpoint_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise MLXUserError("Adapter config requires a valid foundation checkpoint_sha256 checksum.")
        return config
