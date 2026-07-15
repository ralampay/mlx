from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import print_warning
from mlx.modes.image_classification.data import iter_dataset_images, load_image_tensor
from mlx.modes.image_classification.presentation import (
    display_classification_predictions,
    display_similarity_matches,
)
from mlx.modes.image_classification.utils import load_checkpoint_bundle


def infer_image_classification(config: dict[str, Any]) -> dict[str, Any]:
    model, metadata = load_checkpoint_bundle(config)
    device = config.get("device", "cpu")
    model = model.to(device)
    model.eval()

    if metadata["family"] == "one-shot":
        return _infer_one_shot(model, metadata, config, device)
    return _infer_standard(model, metadata, config, device)


def _infer_one_shot(model, metadata: dict[str, Any], config: dict[str, Any], device: str) -> dict[str, Any]:
    return RankOneShotReferences(
        model=model,
        metadata=metadata,
        input_image=Path(config["input_img"]),
        dataset_path=Path(config["dataset_path"]),
        device=device,
    ).execute()


class RankOneShotReferences:
    def __init__(self, *, model, metadata, input_image: Path, dataset_path: Path, device: str) -> None:
        self.model = model
        self.metadata = metadata
        self.input_image = input_image
        self.dataset_path = dataset_path
        self.device = device

    def execute(self) -> dict[str, Any]:
        if not self.dataset_path.exists():
            raise MLXUserError(f"Dataset path not found: {self.dataset_path}")

        query = self._load_tensor(self.input_image)
        matches: list[tuple[str, Path, float]] = []
        with torch.no_grad():
            for reference_path in iter_dataset_images(self.dataset_path):
                try:
                    reference = self._load_tensor(reference_path)
                except MLXUserError as exc:
                    print_warning(f"Skipping {reference_path}: {exc}")
                    continue

                similarity = float(self.model(query, reference).reshape(-1)[0].item())
                label = (
                    reference_path.parent.name
                    if reference_path.parent != self.dataset_path
                    else reference_path.stem
                )
                matches.append((label, reference_path, similarity))

        matches.sort(key=lambda item: item[2], reverse=True)
        best_match = matches[0] if matches else None
        result = {
            "input_image": self.input_image,
            "best_match_label": best_match[0] if best_match else None,
            "best_match_path": best_match[1] if best_match else None,
            "similarity_score": best_match[2] if best_match else None,
            "top_matches": matches[:10],
        }
        display_similarity_matches(result)
        return result

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        tensor = load_image_tensor(
            image_path,
            input_size=self.metadata["input_size"],
            colored=self.metadata["colored"],
        )
        return tensor.unsqueeze(0).to(self.device)


def _infer_standard(model, metadata: dict[str, Any], config: dict[str, Any], device: str) -> dict[str, Any]:
    input_img_path = Path(config["input_img"])
    classes = metadata["classes"]
    if not classes:
        raise MLXUserError(
            "The checkpoint does not contain class labels, so standard infer-image cannot run."
        )

    with torch.no_grad():
        image = load_image_tensor(
            input_img_path,
            input_size=metadata["input_size"],
            colored=metadata["colored"],
        )
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_k = min(5, len(classes))
    scores, indices = torch.topk(probabilities, k=top_k)
    top_predictions = [(classes[index], float(score)) for score, index in zip(scores.tolist(), indices.tolist())]
    result = {
        "input_image": input_img_path,
        "predicted_label": top_predictions[0][0] if top_predictions else None,
        "top_predictions": top_predictions,
    }
    display_classification_predictions(result)
    return result
