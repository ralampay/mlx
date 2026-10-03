"""Small persistent exact cosine index, independent of approximate-search libraries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.vector_store.protocol import VectorSearchResult


class ExactCosineVectorStore:
    def __init__(self, path, *, collection="corpus", create=True):
        self.root = Path(path)
        if collection != "corpus":
            raise MLXUserError("The exact index supports the corpus collection only.")
        self.ids = []
        self._seen = set()
        self._dimensions = None
        self._matrix = None
        self._output = None
        self._writable = create
        try:
            if create:
                self.root.mkdir(parents=True, exist_ok=True)
                if any(self.root.iterdir()):
                    raise MLXUserError(f"Exact index directory must be empty: {self.root}")
                self._output = (self.root / "vectors.f32").open("wb")
            else:
                manifest = json.loads((self.root / "index.json").read_text())
                self.ids = manifest["ids"]
                self._dimensions = manifest["dimensions"]
                if (manifest.get("schema_version") != 1 or manifest.get("similarity") != "cosine"
                        or type(self._dimensions) is not int or self._dimensions < 1
                        or not isinstance(self.ids, list) or not self.ids
                        or any(not isinstance(i, str) or not i for i in self.ids)
                        or len(set(self.ids)) != len(self.ids)):
                    raise ValueError("Invalid exact index manifest")
                if (self.root / "vectors.f32").stat().st_size != len(self.ids) * self._dimensions * 4:
                    raise ValueError("Exact index vector size does not match its manifest")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MLXUserError(f"Unable to open exact cosine index {self.root}: {exc}") from exc

    def add(self, records):
        if not self._writable or self._output is None:
            raise MLXUserError("Exact index is read-only or closed.")
        if not records:
            return
        try:
            values = np.asarray([r.vector for r in records], dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise MLXUserError("Exact index requires rectangular numeric vectors.") from exc
        if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
            raise MLXUserError("Exact index requires finite nonempty vectors.")
        if self._dimensions is not None and values.shape[1] != self._dimensions:
            raise MLXUserError("Exact index embedding dimension mismatch.")
        ids = [r.id for r in records]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids) or self._seen.intersection(ids):
            raise MLXUserError("Exact index requires unique nonempty document IDs.")
        self._dimensions = values.shape[1]
        values /= np.maximum(np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True), 1e-12)
        self._output.write(values.astype("<f4").tobytes())
        self.ids.extend(ids)
        self._seen.update(ids)
        self._matrix = None

    def query(self, vector, *, k):
        if type(k) is not int or k < 1 or not self.ids:
            raise MLXUserError("Exact search requires a positive depth and a nonempty index.")
        try:
            values = np.array(vector, dtype=np.float32, copy=True)
        except (TypeError, ValueError) as exc:
            raise MLXUserError("Exact search query must contain numeric values.") from exc
        if values.shape != (self._dimensions,) or not np.isfinite(values).all():
            raise MLXUserError("Exact search query has invalid dimensions or non-finite values.")
        if self._output is not None:
            self._output.flush()
        if self._matrix is None:
            self._matrix = np.memmap(self.root / "vectors.f32", dtype="<f4", mode="r",
                                     shape=(len(self.ids), self._dimensions))
        values /= max(float(np.linalg.norm(values.astype(np.float64))), 1e-12)
        # Bound score computation even when callers use corpora larger than NanoBEIR.
        candidates = []
        for start in range(0, len(self.ids), 4096):
            scores = self._matrix[start:start + 4096] @ values
            if not np.isfinite(scores).all():
                raise MLXUserError("Exact index contains non-finite vectors; rebuild it.")
            candidates.extend((float(score), self.ids[start + j]) for j, score in enumerate(scores))
            candidates = sorted(candidates, key=lambda item: (-item[0], item[1]))[:k]
        return [VectorSearchResult(identifier, score) for score, identifier in candidates]

    def close(self):
        if self._output is not None:
            self._output.close()
            self._output = None
            write_json_atomic(self.root / "index.json", {
                "schema_version": 1, "similarity": "cosine",
                "dimensions": self._dimensions, "ids": self.ids,
            })
        self._matrix = None
