from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import BatchSampler, Dataset

from mlx.core.exceptions import MLXUserError


@dataclass(frozen=True)
class EmbeddingCsvTable:
    path: Path
    fieldnames: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]
    vectors: tuple[tuple[float, ...], ...]
    dimensions: int
    source_normalized: bool


class EmbeddingCsvLoader:
    def load(self, path: str | Path) -> EmbeddingCsvTable:
        source_path = Path(path).expanduser()
        if not source_path.is_file():
            raise MLXUserError(f"Embedding CSV not found: {source_path}")
        rows: list[Mapping[str, str]] = []
        vectors: list[tuple[float, ...]] = []
        dimensions: int | None = None
        identifiers: set[str] = set()
        try:
            with source_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                fieldnames = tuple(reader.fieldnames or ())
                if "embedding" not in fieldnames:
                    raise MLXUserError(
                        f"Embedding CSV '{source_path}' must contain an 'embedding' column."
                    )
                for line, row in enumerate(reader, start=2):
                    vector = self._parse_vector(row.get("embedding"), source_path, line)
                    if dimensions is None:
                        dimensions = len(vector)
                    elif len(vector) != dimensions:
                        raise MLXUserError(
                            f"Embedding dimension changed in {source_path} at line {line}: "
                            f"expected {dimensions}, got {len(vector)}."
                        )
                    identifier = row.get("id")
                    if identifier is not None:
                        if not identifier or identifier in identifiers:
                            raise MLXUserError(
                                f"Embedding CSV has an empty or duplicate ID at line {line}: {identifier!r}."
                            )
                        identifiers.add(identifier)
                    rows.append(dict(row))
                    vectors.append(vector)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise MLXUserError(f"Unable to read embedding CSV '{source_path}': {exc}") from exc
        if not vectors or dimensions is None:
            raise MLXUserError(f"Embedding CSV contains no vector rows: {source_path}")
        source_normalized = self._source_normalized(source_path, dimensions)
        return EmbeddingCsvTable(
            source_path,
            fieldnames,
            tuple(rows),
            tuple(vectors),
            dimensions,
            source_normalized,
        )

    @staticmethod
    def _parse_vector(value: str | None, path: Path, line: int) -> tuple[float, ...]:
        try:
            raw = json.loads(value or "")
        except json.JSONDecodeError as exc:
            raise MLXUserError(f"Malformed embedding JSON in {path} at line {line}.") from exc
        if not isinstance(raw, list) or not raw:
            raise MLXUserError(f"Embedding in {path} at line {line} must be a non-empty JSON array.")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
            raise MLXUserError(f"Embedding in {path} at line {line} contains non-numeric values.")
        vector = tuple(float(item) for item in raw)
        if any(not math.isfinite(item) for item in vector):
            raise MLXUserError(f"Embedding in {path} at line {line} contains non-finite values.")
        return vector

    @staticmethod
    def _source_normalized(path: Path, dimensions: int) -> bool:
        own_manifest = path.with_suffix(".manifest.json")
        manifest_path = (
            own_manifest
            if own_manifest.is_file()
            else path.parent / "embedding_manifest.json"
        )
        if (
            manifest_path.name == "embedding_manifest.json"
            and path.name not in {"corpus_embeddings.csv", "query_embeddings.csv"}
        ):
            return False
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path == own_manifest:
                manifest_dimensions = int(manifest["output_dimensions"])
                normalized = manifest["normalized"]
            else:
                embedding = manifest["embedding"]
                manifest_dimensions = int(embedding["dimensions"])
                normalized = embedding["normalized"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise MLXUserError(
                f"Unable to use sibling embedding manifest '{manifest_path}': {exc}"
            ) from exc
        if manifest_dimensions != dimensions:
            raise MLXUserError(
                f"Sibling embedding manifest dimensions ({manifest_dimensions}) do not match "
                f"CSV dimensions ({dimensions})."
            )
        if not isinstance(normalized, bool):
            raise MLXUserError(
                f"Embedding normalization provenance must be Boolean in '{manifest_path}'."
            )
        return normalized


class VectorDataset(Dataset):
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.values = torch.tensor(vectors, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


class MergeSingletonBatchSampler(BatchSampler):
    """Retain every sample, merging a trailing singleton into the previous batch."""

    def __init__(self, sampler, batch_size: int) -> None:
        if batch_size < 2 or len(sampler) < 2:
            raise ValueError("Similarity batching requires batch size and partition size >= 2.")
        super().__init__(sampler, batch_size, drop_last=False)

    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    yield pending + batch
                    return
                yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        count = super().__len__()
        return count - int(len(self.sampler) % self.batch_size == 1)


def l2_normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if torch.any(norms == 0):
        raise MLXUserError("Cannot L2-normalize a zero input vector.")
    return values / norms


def load_json_object(path, *, purpose: str) -> dict[str, Any]:
    from mlx.core.configuration import load_component_options
    return load_component_options(path, purpose=purpose)


__all__ = [
    "EmbeddingCsvLoader",
    "EmbeddingCsvTable",
    "VectorDataset",
    "MergeSingletonBatchSampler",
    "l2_normalize_tensor",
    "load_json_object",
]
