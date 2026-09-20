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
from mlx.modes.saliency_mapping.checkpoints import load_checkpoint_bundle
from mlx.modes.saliency_mapping.data import SaliencyDataset
from mlx.modes.saliency_mapping.visualization import (
    compose_saliency_panel,
    probability_to_gray,
    probability_to_heatmap,
    saliency_overlay,
    tensor_to_rgb,
    write_rgb,
)
from mlx.modes.saliency_mapping.compatibility import evenly_spaced_sample_indices


class GenerateSaliencySamples:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        checkpoint_path: str | Path,
        split_path: str | Path,
        output_dir: str | Path,
        sample_limit: int = 16,
        reporter: WorkflowReporter | None = None,
        model_registry=None,
    ) -> None:
        self.model_registry = model_registry
        self.config = dict(config)
        self.checkpoint_path = Path(checkpoint_path)
        self.split_path = Path(split_path)
        self.output_dir = Path(output_dir)
        self.sample_limit = int(sample_limit)
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> list[Path]:
        if self.sample_limit < 1:
            raise MLXUserError("Saliency sample limit must be at least 1.")
        model, metadata = load_checkpoint_bundle(
            {**self.config, "model_path": str(self.checkpoint_path)},
            **({"model_registry": self.model_registry} if self.model_registry is not None else {}),
        )
        device = str(self.config.get("device", "cpu"))
        model = model.to(device).eval()
        dataset = SaliencyDataset(
            self.split_path,
            split="test",
            input_size=tuple(metadata["input_size"]),
            colored=bool(metadata["colored"]),
            transform=str(metadata["transform"]),
        )
        indices = evenly_spaced_sample_indices(len(dataset), self.sample_limit)
        final_root = self.output_dir / "samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".saliency-samples-", dir=self.output_dir))
        try:
            paths = self._render(model, dataset, indices, temporary, device)
            if final_root.exists():
                shutil.rmtree(final_root)
            temporary.replace(final_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise MLXUserError(f"Unable to generate saliency samples: {exc}") from exc
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        final = [final_root / "panels" / path.name for path in paths]
        emit(self.reporter, "success", f"Wrote {len(final)} saliency sample panels to {final_root}")
        return final

    def _render(self, model, dataset, indices, root: Path, device: str) -> list[Path]:
        names = ("original", "ground_truth", "prediction", "heatmap", "overlay", "panels")
        directories = {name: root / name for name in names}
        for path in directories.values():
            path.mkdir(parents=True)
        panels = []
        with torch.inference_mode():
            for index in indices:
                image, target = dataset[index]
                probability = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                target_array = target[0].numpy()
                original = tensor_to_rgb(image)
                overlay = saliency_overlay(
                    original,
                    probability,
                    float(self.config.get("overlay_alpha", 0.45)),
                )
                panel = compose_saliency_panel(original, target_array, probability, overlay)
                stem = dataset.samples[index][0].stem
                write_rgb(directories["original"] / f"{stem}.png", original)
                cv2.imwrite(str(directories["ground_truth"] / f"{stem}.png"), probability_to_gray(target_array))
                cv2.imwrite(str(directories["prediction"] / f"{stem}.png"), probability_to_gray(probability))
                write_rgb(directories["heatmap"] / f"{stem}.png", probability_to_heatmap(probability))
                write_rgb(directories["overlay"] / f"{stem}.png", overlay)
                panel_path = directories["panels"] / f"{stem}.png"
                write_rgb(panel_path, panel)
                panels.append(panel_path)
        return panels


__all__ = ["GenerateSaliencySamples"]
