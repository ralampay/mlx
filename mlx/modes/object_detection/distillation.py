"""Validation of the local object-detection distillation request."""

import math
from pathlib import Path
from typing import Any, Mapping

from mlx.core.exceptions import MLXUserError


def validate_distillation_options(config: Mapping[str, Any]) -> None:
    keys = ("distiller", "distill_loss", "distill_weight", "distill_temperature", "distill_mask_ratio")
    if not any(config.get(key) is not None for key in keys):
        return
    if (
        str(config.get("mode", "object_detection")).replace("-", "_") != "object_detection"
        or config.get("platform", "local") != "local"
        or config.get("provider", "ultralytics") != "libreyolo"
        or config.get("action", "train") not in {"train", "fine-tune"}
    ):
        raise MLXUserError("Distillation supports only local LibreYOLO object-detection train/fine-tune.")
    if not config.get("distiller"):
        raise MLXUserError("Distillation options require --distiller pointing to a teacher .pt checkpoint.")
    teacher = Path(config["distiller"]).expanduser()
    if teacher.suffix.lower() != ".pt" or not teacher.is_file():
        raise MLXUserError(f"Teacher checkpoint must be an existing local .pt file: {teacher}")
    loss = config.get("distill_loss") or "cwd"
    if loss not in {"cwd", "mgd"}:
        raise MLXUserError("--distill-loss must be cwd or mgd.")
    for key in ("distill_weight", "distill_temperature", "distill_mask_ratio"):
        raw = config.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise MLXUserError(f"{key} must be a finite number.") from exc
        valid = 0 <= value < 1 if key == "distill_mask_ratio" else value > 0
        if not math.isfinite(value) or not valid:
            raise MLXUserError(f"{key} must be finite and {'in [0, 1)' if key == 'distill_mask_ratio' else 'positive'}.")
    if loss == "cwd" and config.get("distill_mask_ratio") is not None:
        raise MLXUserError("--distill-mask-ratio applies only to MGD.")
    if loss == "mgd" and config.get("distill_temperature") is not None:
        raise MLXUserError("--distill-temperature applies only to CWD.")
