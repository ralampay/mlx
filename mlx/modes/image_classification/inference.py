from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.data import iter_dataset_images, load_image_tensor
from mlx.modes.image_classification.utils import load_checkpoint_bundle
from mlx.modes.image_classification.models.joint_svdd import JointDeepSVDDClassifier
from mlx.modes.image_classification.requests import ImageClassificationRequest


class InferImageClassification:
    def __init__(
        self,
        request: ImageClassificationRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> dict[str, Any]:
        return _run_inference(self.request.to_config(), reporter=self.reporter)


def infer_image_classification(config: dict[str, Any]) -> dict[str, Any]:
    compatibility_config = {"display": True, **config}
    result = InferImageClassification(
        ImageClassificationRequest.from_config(compatibility_config)
    ).execute()
    if compatibility_config.get("display", True):
        _display_inference_result(result)
    return result


def _run_inference(
    config: dict[str, Any],
    *,
    reporter: WorkflowReporter | None = None,
) -> dict[str, Any]:
    reporter = reporter or NullWorkflowReporter()
    model, metadata = load_checkpoint_bundle(config)
    device = config.get("device", "cpu")
    model = model.to(device)
    model.eval()

    if metadata["family"] == "one-shot":
        return _infer_one_shot(model, metadata, config, device, reporter=reporter)
    return _infer_standard(model, metadata, config, device)


def _infer_one_shot(
    model,
    metadata: dict[str, Any],
    config: dict[str, Any],
    device: str,
    *,
    reporter: WorkflowReporter,
) -> dict[str, Any]:
    return RankOneShotReferences(
        model=model,
        metadata=metadata,
        input_image=Path(config["input_img"]),
        dataset_path=Path(config["dataset_path"]),
        device=device,
        reporter=reporter,
    ).execute()


class RankOneShotReferences:
    def __init__(
        self,
        *,
        model,
        metadata,
        input_image: Path,
        dataset_path: Path,
        device: str,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.input_image = input_image
        self.dataset_path = dataset_path
        self.device = device
        self.reporter = reporter or NullWorkflowReporter()

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
                    emit(
                        self.reporter,
                        "warning",
                        f"Skipping {reference_path}: {exc}",
                    )
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
        return result

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        tensor = load_image_tensor(
            image_path,
            input_size=self.metadata["input_size"],
            colored=self.metadata["colored"],
        )
        return tensor.unsqueeze(0).to(self.device)


def _display_inference_result(result: dict[str, Any]) -> None:
    if "top_matches" in result:
        display_similarity_matches(result)
    else:
        display_classification_predictions(result)


def display_similarity_matches(result: dict[str, Any]) -> None:
    """Compatibility presentation entrypoint retained outside command execution."""

    from mlx.modes.image_classification.presentation import display_similarity_matches as display

    display(result)


def display_classification_predictions(result: dict[str, Any]) -> None:
    """Compatibility presentation entrypoint retained outside command execution."""

    from mlx.modes.image_classification.presentation import (
        display_classification_predictions as display,
    )

    display(result)


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
        output = model(image.unsqueeze(0).to(device))
        if isinstance(model, JointDeepSVDDClassifier):
            score = float(model.compute_svdd_score(output.svdd_embedding)[0].item())
            accepted = bool(model.is_in_distribution(output.svdd_embedding)[0].item())
            logits = output.logits
        else:
            score = None
            accepted = True
            logits = output
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_k = min(5, len(classes))
    scores, indices = torch.topk(probabilities, k=top_k)
    top_predictions = [(classes[index], float(score)) for score, index in zip(scores.tolist(), indices.tolist())]
    if isinstance(model, JointDeepSVDDClassifier):
        threshold = float(model.svdd_threshold.item())
        result = {
            "input_image": input_img_path,
            "accepted": accepted,
            "predicted_label": top_predictions[0][0] if accepted and top_predictions else None,
            "confidence": top_predictions[0][1] if accepted and top_predictions else None,
            "top_predictions": top_predictions if accepted else [],
            "ood_score": score,
            "ood_threshold": threshold,
            "rejection_reason": None if accepted else "out_of_distribution",
        }
        return result
    result = {
        "input_image": input_img_path,
        "predicted_label": top_predictions[0][0] if top_predictions else None,
        "top_predictions": top_predictions,
    }
    return result
