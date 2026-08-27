from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch import nn

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.data import (
    load_image_tensor,
    load_standard_classification_directory,
    resolve_evaluation_dir,
)
from mlx.modes.image_classification.evaluation import build_one_shot_benchmark_pairs
from mlx.modes.image_classification.utils import load_checkpoint_bundle
from mlx.modes.image_classification.requests import ImageClassificationRequest


@dataclass(frozen=True)
class CamResult:
    source_path: Path
    output_path: Path | None
    label: str | None
    predicted_label: str | None
    target_index: int
    method: str
    visualization: np.ndarray
    grayscale_cam: np.ndarray


class _ClassifierTarget:
    def __init__(self, category: int) -> None:
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class _SiameseOutputTarget:
    def __init__(self, index: int = 0) -> None:
        self.index = index

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        output = model_output.reshape(model_output.shape[0], -1)
        return output[:, self.index]


class _SiameseBranchWrapper(nn.Module):
    def __init__(self, model: nn.Module, fixed_image: torch.Tensor, *, explain_first: bool) -> None:
        super().__init__()
        self.model = model
        self.explain_first = explain_first
        self.register_buffer("fixed_image", fixed_image.detach())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        fixed = self.fixed_image.expand(image.shape[0], -1, -1, -1)
        if self.explain_first:
            return self.model(image, fixed)
        return self.model(fixed, image)


class GenerateImageClassificationCams:
    def __init__(
        self,
        request: ImageClassificationRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> list[CamResult]:
        results = _generate_cams(self.request.to_config())
        emit(
            self.reporter,
            "success",
            f"Generated {len(results)} CAM image(s)",
            payload={"event": "cam_completed", "count": len(results)},
        )
        return results


def generate_image_classification_cams(config: dict[str, Any]) -> list[CamResult]:
    config = {"display": True, **config}
    results = GenerateImageClassificationCams(
        ImageClassificationRequest.from_config(config)
    ).execute()
    if config.get("display", True):
        display_cam_results(results, delay=int(config.get("window_delay", 0)))
    return results


def _generate_cams(config: dict[str, Any]) -> list[CamResult]:
    model, metadata = load_checkpoint_bundle(config)
    device = config.get("device", "cpu")
    model = model.to(device)
    model.eval()

    method = str(config.get("cam_method") or "gradcam").lower()
    if metadata["family"] == "one-shot":
        results = generate_one_shot_cams(model, metadata, config, device=device, method=method)
    else:
        results = generate_standard_cams(model, metadata, config, device=device, method=method)

    return results


def generate_standard_cams(
    model: nn.Module,
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    device: str,
    method: str,
) -> list[CamResult]:
    classes = metadata.get("classes") or []
    if not classes:
        raise MLXUserError("CAM generation for standard classifiers requires checkpoint class labels.")

    eval_dir = resolve_evaluation_dir(config["dataset_path"])
    dataset = load_standard_classification_directory(
        eval_dir,
        label_names=classes,
        input_size=metadata["input_size"],
        colored=metadata["colored"],
    )
    samples = dataset.samples
    max_samples = config.get("max_samples")
    if max_samples is not None:
        samples = samples[: max(0, int(max_samples))]

    target_layers = [_resolve_target_layer(model, config.get("target_layer"), metadata["model_name"])]
    output_dir = _resolve_output_dir(config)
    cam_class = _resolve_cam_class(method)
    results: list[CamResult] = []

    with cam_class(model=model, target_layers=target_layers) as cam:
        for image_path, label_index in samples:
            input_tensor, rgb_image = _load_cam_input(
                image_path,
                input_size=metadata["input_size"],
                colored=metadata["colored"],
                device=device,
            )
            target_index = _resolve_standard_target_index(
                model,
                input_tensor,
                config.get("target_index"),
            )
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[_ClassifierTarget(target_index)],
                aug_smooth=bool(config.get("aug_smooth", False)),
                eigen_smooth=bool(config.get("eigen_smooth", False)),
            )[0]
            visualization = overlay_cam(rgb_image, grayscale_cam)
            predicted_label = classes[target_index] if target_index < len(classes) else str(target_index)
            label = classes[label_index] if label_index < len(classes) else str(label_index)
            output_path = _save_cam_image(
                visualization,
                output_dir,
                image_path,
                prefix=f"{method}-{predicted_label}",
                enabled=bool(config.get("save_images", True)),
            )
            results.append(
                CamResult(
                    source_path=image_path,
                    output_path=output_path,
                    label=label,
                    predicted_label=predicted_label,
                    target_index=target_index,
                    method=method,
                    visualization=visualization,
                    grayscale_cam=grayscale_cam,
                )
            )
    return results


def generate_one_shot_cams(
    model: nn.Module,
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    device: str,
    method: str,
) -> list[CamResult]:
    test_path = resolve_evaluation_dir(config["dataset_path"])
    pairs = build_one_shot_benchmark_pairs(
        test_path,
        pairs_per_class=int(config.get("num_pairs", 100)),
        random_seed=config.get("random_seed"),
    )
    max_samples = config.get("max_samples")
    if max_samples is not None:
        pairs = pairs[: max(0, int(max_samples))]

    target_layer_path = config.get("target_layer") or "embedding.3"
    output_dir = _resolve_output_dir(config)
    cam_class = _resolve_cam_class(method)
    results: list[CamResult] = []

    for pair_index, pair in enumerate(pairs, start=1):
        first_tensor, first_rgb = _load_cam_input(
            pair.image_one,
            input_size=metadata["input_size"],
            colored=metadata["colored"],
            device=device,
        )
        second_tensor, second_rgb = _load_cam_input(
            pair.image_two,
            input_size=metadata["input_size"],
            colored=metadata["colored"],
            device=device,
        )
        for side, image_path, rgb_image, moving_tensor, fixed_tensor, explain_first in (
            ("a", pair.image_one, first_rgb, first_tensor, second_tensor, True),
            ("b", pair.image_two, second_rgb, second_tensor, first_tensor, False),
        ):
            wrapper = _SiameseBranchWrapper(model, fixed_tensor, explain_first=explain_first).to(device)
            target_layer = _resolve_target_layer(wrapper.model, target_layer_path, metadata["model_name"])
            with cam_class(model=wrapper, target_layers=[target_layer]) as cam:
                target_index = int(config.get("target_index") or 0)
                grayscale_cam = cam(
                    input_tensor=moving_tensor,
                    targets=[_SiameseOutputTarget(target_index)],
                    aug_smooth=bool(config.get("aug_smooth", False)),
                    eigen_smooth=bool(config.get("eigen_smooth", False)),
                )[0]
            visualization = overlay_cam(rgb_image, grayscale_cam)
            predicted_label = "same" if pair.target == 1 else "different"
            output_path = _save_cam_image(
                visualization,
                output_dir,
                image_path,
                prefix=f"{method}-pair{pair_index:04d}-{side}-{predicted_label}",
                enabled=bool(config.get("save_images", True)),
            )
            results.append(
                CamResult(
                    source_path=image_path,
                    output_path=output_path,
                    label=image_path.parent.name,
                    predicted_label=predicted_label,
                    target_index=target_index,
                    method=method,
                    visualization=visualization,
                    grayscale_cam=grayscale_cam,
                )
            )
    return results


def overlay_cam(rgb_image: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise MLXUserError("Install the optional Grad-CAM dependency with 'pip install grad-cam'.") from exc

    return show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)


def display_cam_results(results: Iterable[CamResult], *, delay: int = 0) -> None:
    from mlx.modes.image_classification.presentation import display_cam_results as display

    display(results, delay=delay)


def _resolve_cam_class(method: str):
    try:
        from pytorch_grad_cam import AblationCAM, GradCAM, ScoreCAM
    except ImportError as exc:
        raise MLXUserError("Install the optional Grad-CAM dependency with 'pip install grad-cam'.") from exc

    classes = {
        "gradcam": GradCAM,
        "ablationcam": AblationCAM,
        "scorecam": ScoreCAM,
    }
    try:
        return classes[method]
    except KeyError as exc:
        raise MLXUserError("Unsupported CAM method. Choose gradcam, ablationcam, or scorecam.") from exc


def _resolve_standard_target_index(model: nn.Module, input_tensor: torch.Tensor, target_index: int | None) -> int:
    if target_index is not None:
        return int(target_index)
    with torch.no_grad():
        logits = model(input_tensor)
    return int(logits.argmax(dim=1).item())


def _load_cam_input(
    image_path: Path,
    *,
    input_size: tuple[int, int],
    colored: bool,
    device: str,
) -> tuple[torch.Tensor, np.ndarray]:
    tensor = load_image_tensor(image_path, input_size=input_size, colored=colored)
    rgb_image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if not colored:
        rgb_image = np.repeat(rgb_image, 3, axis=2)
    rgb_image = np.clip(rgb_image, 0.0, 1.0).astype(np.float32)
    return tensor.unsqueeze(0).to(device), rgb_image


def _resolve_target_layer(model: nn.Module, requested_layer: str | None, model_name: str) -> nn.Module:
    layer_path = requested_layer or _default_target_layer_path(model, model_name)
    try:
        return _resolve_module_path(model, layer_path)
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise MLXUserError(f"Could not resolve CAM target layer '{layer_path}'.") from exc


def _default_target_layer_path(model: nn.Module, model_name: str) -> str:
    if model_name == "siamese-le-net" and hasattr(model, "embedding"):
        return "embedding.3"
    if model_name.startswith("siamese-") and hasattr(model, "embedding"):
        if model_name == "siamese-drax_mobilenet_v3_large":
            return "embedding.adapter_up"
        if hasattr(model.embedding, "layer4"):
            return "embedding.layer4.-1"
        if hasattr(model.embedding, "features"):
            return "embedding.features.-1"
    if hasattr(model, "layer4"):
        return "layer4.-1"
    if hasattr(model, "features"):
        return "features.-1"
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return name
    raise MLXUserError("Could not infer a CAM target layer. Pass --target-layer.")


def _resolve_module_path(model: nn.Module, path: str) -> nn.Module:
    module: Any = model
    for part in path.split("."):
        if part.lstrip("-").isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Resolved object is not a torch module: {path}")
    return module


def _resolve_output_dir(config: dict[str, Any]) -> Path | None:
    output_path = config.get("output_path")
    if not output_path:
        return None
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_cam_image(
    visualization: np.ndarray,
    output_dir: Path | None,
    source_path: Path,
    *,
    prefix: str,
    enabled: bool,
) -> Path | None:
    if output_dir is None or not enabled:
        return None
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix)
    safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in source_path.parent.name)
    output_path = output_dir / f"{safe_label}-{source_path.stem}-{safe_prefix}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return output_path
