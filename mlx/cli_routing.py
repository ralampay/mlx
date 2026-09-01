from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping


class UnknownModeError(ValueError):
    """Raised when the selected mode is not registered."""


ModeRunner = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ModeDescriptor:
    name: str
    aliases: tuple[str, ...]
    runner: str
    default_action: str
    actions: tuple[str, ...]
    purpose: str


MODE_DESCRIPTORS: tuple[ModeDescriptor, ...] = (
    ModeDescriptor(
        name="object_detection",
        aliases=("object-detection",),
        runner="mlx.modes.object_detection.runner:run_object_detection",
        default_action="train",
        actions=("train", "fine-tune", "benchmark", "best-model", "resume", "status", "stop", "convert", "infer-camera", "infer-video", "ls-models"),
        purpose="Provider-backed detection training, evaluation, and inference",
    ),
    ModeDescriptor(
        name="track",
        aliases=("tracking",),
        runner="mlx.modes.object_detection.tracking.runner:run_tracking",
        default_action="run",
        actions=("run", "export-mot", "ls-trackers"),
        purpose="Provider-neutral video tracking and MOT benchmarking",
    ),
    ModeDescriptor(
        name="image_classification",
        aliases=("image-classification",),
        runner="mlx.modes.image_classification.runner:run_image_classification",
        default_action="test",
        actions=("train", "test", "benchmark", "resume", "status", "stop", "infer-image", "cam", "build-dataset", "ls-models"),
        purpose="One-shot and standard image-classification workflows",
    ),
    ModeDescriptor(
        name="image_recognition_oc",
        aliases=("image-recognition-oc",),
        runner="mlx.modes.image_recognition_oc.runner:run_image_recognition_oc",
        default_action="ls-models",
        actions=("train", "train-all", "infer-image", "benchmark", "resume", "status", "stop", "ls-models"),
        purpose="Normal-only still-image one-class recognition",
    ),
    ModeDescriptor(
        name="segmentation",
        aliases=(),
        runner="mlx.modes.segmentation.runner:run_segmentation",
        default_action="test",
        actions=("train", "test", "benchmark", "infer-image", "infer-camera", "infer-video", "build-dataset", "ls-models"),
        purpose="Semantic segmentation workflows for U-Net style models",
    ),
    ModeDescriptor(
        name="video_anomaly_detection",
        aliases=("video-anomaly-detection",),
        runner="mlx.modes.video_anomaly_detection.runner:run_video_anomaly_detection",
        default_action="ls-models",
        actions=("train", "train-all", "benchmark", "resume", "status", "infer-video", "ls-models"),
        purpose="Normal-only clip-level video anomaly detection",
    ),
    ModeDescriptor(
        name="nlp",
        aliases=(),
        runner="mlx.modes.nlp.runner:run_nlp",
        default_action="embed",
        actions=("embed",),
        purpose="Text embedding workflows for GGUF models and CSV data",
    ),
)

MODE_DESCRIPTOR_REGISTRY: Mapping[str, ModeDescriptor] = MappingProxyType({
    alias: descriptor
    for descriptor in MODE_DESCRIPTORS
    for alias in (descriptor.name, *descriptor.aliases)
})

# Compatibility mapping retained for callers that inspect the historic registry.
MODE_REGISTRY: Mapping[str, str] = MappingProxyType({
    alias: descriptor.runner for alias, descriptor in MODE_DESCRIPTOR_REGISTRY.items()
})


def resolve_mode_descriptor(mode: str) -> ModeDescriptor:
    descriptor = MODE_DESCRIPTOR_REGISTRY.get(mode)
    if descriptor is None:
        raise UnknownModeError(f"Unknown mode '{mode}'.")
    return descriptor


def resolve_mode_runner(mode: str) -> ModeRunner:
    dotted_path = resolve_mode_descriptor(mode).runner
    module_path, function_name = dotted_path.split(":")
    return getattr(import_module(module_path), function_name)


__all__ = [
    "MODE_DESCRIPTORS",
    "MODE_DESCRIPTOR_REGISTRY",
    "MODE_REGISTRY",
    "ModeDescriptor",
    "ModeRunner",
    "UnknownModeError",
    "resolve_mode_descriptor",
    "resolve_mode_runner",
]
