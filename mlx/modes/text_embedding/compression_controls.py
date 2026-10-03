"""Fitted and deterministic controls for saved vector representations."""
from pathlib import Path

import numpy as np

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError


class TruncateVectors:
    def __init__(self, input_dimensions, output_dimensions):
        if not 1 <= output_dimensions < input_dimensions:
            raise MLXUserError("Truncation width must be positive and smaller than source dimensions.")
        self.input_dimensions, self.output_dimensions = input_dimensions, output_dimensions

    @property
    def provenance(self):
        return {"type": "truncate", "input_dimensions": self.input_dimensions, "output_dimensions": self.output_dimensions}

    def transform(self, vectors):
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dimensions or not np.isfinite(values).all():
            raise MLXUserError("Truncation requires finite vectors of the declared dimension.")
        return values[:, :self.output_dimensions].tolist()


class FitLinearProjection:
    def __init__(self, vectors, training_rows, dimensions, output, *, split_hash, centered=True):
        self.vectors, self.rows = vectors, training_rows
        self.dimensions, self.output, self.split_hash = dimensions, Path(output), split_hash
        self.centered = centered

    def execute(self):
        values = np.asarray(self.vectors, dtype=np.float64)[self.rows]
        if self.dimensions > min(values.shape) or not np.isfinite(values).all():
            raise MLXUserError("Linear projection needs finite training data and at least as many training rows as requested dimensions.")
        try:
            if self.centered:
                from sklearn.decomposition import PCA
                fitted = PCA(n_components=self.dimensions, whiten=False, svd_solver="full").fit(values)
                mean, components = fitted.mean_, fitted.components_
            else:
                _, _, vh = np.linalg.svd(values, full_matrices=False)
                mean, components = np.zeros(values.shape[1]), vh[:self.dimensions]
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise MLXUserError(f"Unable to fit linear projection on training documents: {exc}") from exc
        self.output.mkdir(parents=True, exist_ok=True)
        kind = "pca" if self.centered else "svd"
        np.savez(self.output / f"{kind}.npz", mean=mean, components=components)
        write_json_atomic(self.output / f"{kind}.json", {"split_hash": self.split_hash, "training_rows": len(self.rows),
                                                   "dimensions": self.dimensions, "whiten": False, "solver": "full", "centered": self.centered})
        return self.output


class LinearProjectionVectors:
    def __init__(self, path, output_dimensions, *, kind="pca"):
        if kind not in ("pca", "svd"):
            raise MLXUserError("Projection kind must be pca or svd.")
        self.kind = kind
        self.path = Path(path)
        try:
            with np.load(self.path, allow_pickle=False) as data:
                self.mean, self.components = data["mean"].copy(), data["components"].copy()
        except (OSError, ValueError, KeyError) as exc:
            raise MLXUserError(f"Unable to read projection artifact {path}: {exc}") from exc
        if (self.mean.ndim != 1 or self.components.ndim != 2 or self.components.shape[1] != len(self.mean)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.components).all()
                or not 1 <= output_dimensions <= len(self.components)):
            raise MLXUserError("Projection artifact or requested prefix has invalid dimensions or values.")
        self.input_dimensions, self.output_dimensions = len(self.mean), output_dimensions

    @property
    def provenance(self):
        return {"type": self.kind, "centered": self.kind == "pca", "sha256": sha256_file(self.path), "output_dimensions": self.output_dimensions,
                "input_dimensions": self.input_dimensions, "whiten": False}

    def transform(self, vectors):
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_dimensions or not np.isfinite(values).all():
            raise MLXUserError("Projection requires finite vectors with the fitted input width.")
        return ((values - self.mean) @ self.components[:self.output_dimensions].T).tolist()


class FitPcaVectors(FitLinearProjection):
    """Backward-compatible centered PCA fitting command."""
    def __init__(self, vectors, training_rows, dimensions, output, *, split_hash):
        super().__init__(vectors, training_rows, dimensions, output, split_hash=split_hash, centered=True)


class PcaVectors(LinearProjectionVectors):
    """Backward-compatible centered PCA adapter."""
    def __init__(self, path, output_dimensions):
        super().__init__(path, output_dimensions, kind="pca")
