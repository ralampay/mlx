from __future__ import annotations

from pathlib import Path

from mlx.core.exceptions import MLXUserError


def object_detection_dataset_root(extracted_path: Path) -> Path:
    """Locate the single YOLO dataset root in an extracted archive."""

    candidates = sorted(path for path in extracted_path.rglob("data.yaml") if path.is_file())
    if len(candidates) != 1:
        raise MLXUserError(
            "The extracted object-detection dataset must contain exactly one data.yaml; "
            f"found {len(candidates)}."
        )
    return candidates[0].parent


__all__ = ["object_detection_dataset_root"]
