"""LibreYOLO construction and trainer boundary for adapter experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

from torch import nn

from mlx.core.exceptions import MLXUserError


MODEL_SIZES = {f"yolox-{size}": size for size in "ntsmlx"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerifyFoundationCheckpoint:
    def __init__(self, model_name: str, checkpoint_path: Path):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)

    def execute(self):
        from libreyolo.models.yolox.nn import LibreYOLOXModel
        from libreyolo.utils.serialization import load_untrusted_torch_file, validate_checkpoint_metadata

        path = self.checkpoint_path
        if not path.is_file():
            raise MLXUserError(f"Foundation checkpoint not found: {path}")
        try:
            checkpoint = load_untrusted_torch_file(str(path), map_location="cpu")
            validate_checkpoint_metadata(checkpoint)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            raise MLXUserError(f"Cannot read YOLOX foundation checkpoint {path}: {exc}. "
                               "Inspect the file and its LibreYOLO metadata.") from exc
        size = MODEL_SIZES[self.model_name]
        if checkpoint.get("model_family") != "yolox" or checkpoint.get("size") != size or checkpoint.get("task") != "detect":
            raise MLXUserError(f"Checkpoint metadata must identify {self.model_name} detection")
        model = LibreYOLOXModel(config=size, nb_classes=checkpoint["nc"])
        try:
            result = model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise MLXUserError(f"Foundation checkpoint is incompatible with standard {self.model_name}: {exc}") from exc
        info = {"checkpoint": str(path), "sha256": _sha256(path), "size": size,
                "classes": checkpoint["names"], "nc": checkpoint["nc"],
                "image_size": checkpoint["imgsz"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}
        return model, info


class _AdapterYOLOXTrainerMixin:
    """Keep frozen BN statistics and trainable counts fixed during setup."""

    def _apply_freeze_config(self):
        super()._apply_freeze_config()
        expected = getattr(self.wrapper_model, "_adapter_expected_trainable", None)
        actual = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if expected is not None and actual != expected:
            raise MLXUserError(f"Adapter trainable parameter count changed during trainer setup: "
                               f"expected {expected}, found {actual}. Check checkpoint loading and freeze settings.")
        self._frozen_bn_modules = tuple(module for module in self.model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and all(not p.requires_grad for p in module.parameters()))


def build_experimental_yolox(raw, info, device):
    from libreyolo.models.yolox.model import LibreYOLOX
    from libreyolo.models.yolox.trainer import YOLOXTrainer

    class AdapterYOLOXTrainer(_AdapterYOLOXTrainerMixin, YOLOXTrainer):
        pass

    class ExperimentalYOLOX(LibreYOLOX):
        def _trainer_class(self):
            return AdapterYOLOXTrainer

    wrapper = ExperimentalYOLOX(model_path=None, size=info["size"], nb_classes=info["nc"], device=device)
    wrapper.model = raw.to(wrapper.device)
    wrapper.names = {int(key): value for key, value in info["classes"].items()}
    wrapper.model_path = str(info["checkpoint"])
    return wrapper
