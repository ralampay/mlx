"""Lazy explanation methods; third-party imports happen for the selected method."""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference


@dataclass(frozen=True)
class CamRegistry:
    entries: Mapping[str, str] = field(default_factory=lambda: {
        "gradcam": "pytorch_grad_cam:GradCAM",
        "ablationcam": "pytorch_grad_cam:AblationCAM",
        "scorecam": "pytorch_grad_cam:ScoreCAM",
    })

    def __post_init__(self):
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def register(self, name, reference):
        if not name.strip():
            raise ValueError("CAM method name cannot be empty.")
        return CamRegistry({**self.entries, name.strip().lower(): reference})

    def resolve(self, name):
        reference = name if ":" in name else self.entries.get(name)
        if reference is None:
            raise MLXUserError(f"Unsupported CAM method '{name}'. Available: {', '.join(sorted(self.entries))}.")
        return load_reference(reference, kind="CAM method (install image-explainability extra)")


DEFAULT_CAM_REGISTRY = CamRegistry()
