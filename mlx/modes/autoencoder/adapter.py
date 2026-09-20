from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mlx.core.artifacts import sha256_file
from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.artifacts import build_checkpoint_model, load_checkpoint
from mlx.modes.autoencoder.data import l2_normalize_tensor
from mlx.modes.autoencoder.models import AutoencoderRegistry, DEFAULT_AUTOENCODER_REGISTRY


class AutoencoderRepresentationTransformer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        registry: AutoencoderRegistry = DEFAULT_AUTOENCODER_REGISTRY,
    ) -> None:
        path, checkpoint = load_checkpoint(checkpoint_path)
        self._path = path
        self._checkpoint = checkpoint
        self._device = device
        self._model = build_checkpoint_model(checkpoint, registry=registry, device=device)

    @property
    def input_dimensions(self) -> int:
        return int(self._checkpoint["input_dimensions"])

    @property
    def output_dimensions(self) -> int:
        return int(self._checkpoint["bottleneck_dimensions"])

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "type": "autoencoder",
            "path": self._path.name,
            "sha256": sha256_file(self._path),
            "architecture": self._checkpoint["architecture"],
            "architecture_path": self._checkpoint["architecture_path"],
            "input_dimensions": self.input_dimensions,
            "output_dimensions": self.output_dimensions,
            "expects_l2_normalized_input": bool(
                self._checkpoint["expects_l2_normalized_input"]
            ),
            "loss": self._checkpoint.get("loss"),
        }

    @torch.no_grad()
    def transform(self, vectors: Sequence[Sequence[float]]) -> list[list[float]]:
        if not vectors:
            return []
        try:
            values = torch.tensor(vectors, dtype=torch.float32, device=self._device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise MLXUserError(f"Unable to convert vectors for autoencoder encoding: {exc}") from exc
        if values.ndim != 2 or values.shape[1] != self.input_dimensions:
            actual = int(values.shape[1]) if values.ndim == 2 else "non-matrix"
            raise MLXUserError(
                f"Autoencoder input dimension mismatch: expected {self.input_dimensions}, got {actual}."
            )
        if not torch.all(torch.isfinite(values)):
            raise MLXUserError("Autoencoder input contains non-finite values.")
        if bool(self._checkpoint["expects_l2_normalized_input"]):
            values = l2_normalize_tensor(values)
        try:
            encoded = self._model.encode(values)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise MLXUserError(f"Autoencoder encoding failed: {exc}") from exc
        if (
            encoded.ndim != 2
            or encoded.shape != (values.shape[0], self.output_dimensions)
            or not torch.all(torch.isfinite(encoded))
        ):
            raise MLXUserError("Autoencoder produced an invalid or non-finite bottleneck representation.")
        result = encoded.detach().cpu().tolist()
        if any(not math.isfinite(float(value)) for row in result for value in row):
            raise MLXUserError("Autoencoder produced a non-finite bottleneck representation.")
        return [[float(value) for value in row] for row in result]


__all__ = ["AutoencoderRepresentationTransformer"]
