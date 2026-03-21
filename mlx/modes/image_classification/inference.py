from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
    input_img_path = Path(config["input_img"])
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise MLXUserError(f"Dataset path not found: {dataset_path}")

    def embedding_for(image_path: Path) -> torch.Tensor:
        with torch.no_grad():
            tensor = load_image_tensor(
                image_path,
                input_size=metadata["input_size"],
                colored=metadata["colored"],
            )
            tensor = tensor.unsqueeze(0).to(device)
            return model.embedding(tensor)

    input_embedding = embedding_for(input_img_path)
    best_match = None
    min_distance = float("inf")
    all_scores = []

    for reference_path in iter_dataset_images(dataset_path):
        try:
            reference_embedding = embedding_for(reference_path)
        except MLXUserError as exc:
            print_warning(f"Skipping {reference_path}: {exc}")
            continue

        distance = F.pairwise_distance(input_embedding, reference_embedding).item()
        label = reference_path.parent.name if reference_path.parent != dataset_path else reference_path.stem
        all_scores.append((label, reference_path, distance))

        if distance < min_distance:
            min_distance = distance
            best_match = (label, reference_path)

    all_scores.sort(key=lambda item: item[2])
    result = {
        "input_image": input_img_path,
        "best_match_label": best_match[0] if best_match else None,
        "best_match_path": best_match[1] if best_match else None,
        "distance": min_distance,
        "top_matches": all_scores[:10],
    }
    display_similarity_matches(result)
    return result


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
