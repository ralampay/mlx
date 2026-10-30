"""Provider boundary for caching predictions from completed adapter studies."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter_slices import AdapterSliceRequest, StudyRun


class SlicePredictionWriter(Protocol):
    def __call__(self, run: StudyRun, split: str, destination: Path) -> Mapping[str, Any]: ...


class AdapterSliceBackend(Protocol):
    def prepare(self, request: AdapterSliceRequest, model_name: str | None) -> tuple[Mapping[str, Any], Any]:
        """Return verified foundation metadata (classes, nc, sha256) and a device handle."""
        ...

    def create_writer(self, request: AdapterSliceRequest, foundation: Mapping[str, Any],
                      device: Any, *, registry=None) -> SlicePredictionWriter: ...
