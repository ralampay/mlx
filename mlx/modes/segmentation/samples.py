from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.data import SegmentationEvaluationDataset
from mlx.modes.segmentation.utils import load_checkpoint_bundle
from mlx.modes.segmentation.visualization import (
    blend_overlay,
    colorize_mask,
    compose_segmentation_sample_panel,
)


class GenerateSegmentationSamples:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        checkpoint_path: str | Path,
        test_split_path: str | Path,
        output_dir: str | Path,
        sample_limit: int = 16,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.config = dict(config)
        self.checkpoint_path = Path(checkpoint_path)
        self.test_split_path = Path(test_split_path)
        self.output_dir = Path(output_dir)
        self.sample_limit = int(sample_limit)
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> list[Path]:
        if self.sample_limit < 1:
            raise MLXUserError("Segmentation sample limit must be at least 1.")

        checkpoint_config = {**self.config, "model_path": str(self.checkpoint_path)}
        model, metadata = load_checkpoint_bundle(checkpoint_config)
        device = str(self.config.get("device", "cpu"))
        model = model.to(device)
        model.eval()
        dataset = SegmentationEvaluationDataset(
            self.test_split_path,
            input_size=tuple(metadata["input_size"]),
            num_classes=int(metadata["num_classes"]),
            colored=bool(metadata["colored"]),
            transform=str(metadata.get("transform", "resize")),
        )
        indices = evenly_spaced_sample_indices(len(dataset), self.sample_limit)

        samples_root = self.output_dir / "samples"
        temporary_root: Path | None = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temporary_root = Path(
                tempfile.mkdtemp(prefix=".samples-", dir=str(self.output_dir))
            )
            panel_paths = self._render_samples(
                model,
                metadata,
                dataset,
                indices,
                temporary_root,
                device,
            )
            if samples_root.exists():
                shutil.rmtree(samples_root)
            temporary_root.replace(samples_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise MLXUserError(f"Unable to generate segmentation samples: {exc}") from exc
        finally:
            if temporary_root is not None and temporary_root.exists():
                shutil.rmtree(temporary_root, ignore_errors=True)

        final_paths = [samples_root / "panels" / path.name for path in panel_paths]
        emit(
            self.reporter,
            "success",
            f"Wrote {len(final_paths)} segmentation test samples to {samples_root}",
            payload={
                "event": "segmentation_samples",
                "count": len(final_paths),
                "checkpoint": str(self.checkpoint_path),
                "output_dir": str(samples_root),
            },
        )
        return final_paths

    def _render_samples(
        self,
        model,
        metadata: dict[str, Any],
        dataset: SegmentationEvaluationDataset,
        indices: list[int],
        destination: Path,
        device: str,
    ) -> list[Path]:
        directories = {
            name: destination / name
            for name in ("original", "ground_truth", "prediction", "overlay", "panels")
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        palette = metadata["palette"]
        num_classes = int(metadata["num_classes"])
        threshold = float(metadata.get("mask_threshold", 0.5))
        alpha = float(self.config.get("overlay_alpha", 0.45))
        panel_paths: list[Path] = []
        with torch.no_grad():
            for index in indices:
                image_tensor, target_tensor = dataset[index]
                logits = model(image_tensor.unsqueeze(0).to(device))
                if num_classes == 2:
                    probability = torch.softmax(logits, dim=1)[0, 1]
                    prediction = (probability >= threshold).long().cpu().numpy()
                else:
                    prediction = logits.argmax(dim=1)[0].cpu().numpy()

                original = _tensor_to_rgb_image(image_tensor)
                target = target_tensor.numpy()
                ground_truth = colorize_mask(target, palette)
                predicted = colorize_mask(prediction, palette)
                overlay = blend_overlay(original, predicted, alpha)
                panel = compose_segmentation_sample_panel(
                    original,
                    ground_truth,
                    predicted,
                    overlay,
                )
                stem = dataset.samples[index][0].stem
                for name, image in (
                    ("original", original),
                    ("ground_truth", ground_truth),
                    ("prediction", predicted),
                    ("overlay", overlay),
                    ("panels", panel),
                ):
                    path = directories[name] / f"{stem}.png"
                    _write_rgb_image(path, image)
                    if name == "panels":
                        panel_paths.append(path)
        return panel_paths


def evenly_spaced_sample_indices(sample_count: int, limit: int) -> list[int]:
    if sample_count <= 0 or limit <= 0:
        return []
    selected_count = min(sample_count, limit)
    if selected_count == sample_count:
        return list(range(sample_count))
    return np.linspace(0, sample_count - 1, num=selected_count, dtype=int).tolist()


def _tensor_to_rgb_image(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return array


def _write_rgb_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise MLXUserError(f"Unable to write segmentation sample image: {path}")


__all__ = ["GenerateSegmentationSamples", "evenly_spaced_sample_indices"]
