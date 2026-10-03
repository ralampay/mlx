"""Validated named variants and reproducible v2 experiment expansion."""
from dataclasses import dataclass, replace, field
import math
import re

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.experiment_artifacts import read_json


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    kind: str
    model: str
    loss: str
    dimensions: tuple[int, ...]
    evaluation_dimensions: tuple[int, ...]
    model_config: dict
    loss_config: dict

    def runs(self, seeds):
        if self.kind == "truncate":
            return [(None, d, (d,)) for d in self.dimensions]
        return [(seed, width, self.evaluation_dimensions or (width,))
                for seed in seeds for width in self.dimensions]


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    variants: tuple[ExperimentVariant, ...]
    secondary: tuple[tuple[str, str], ...]
    training: dict
    selection: dict = field(default_factory=dict)

    def counts(self):
        neural = fits = svd_fits = evaluations = 0
        for v in self.variants:
            runs = v.runs(self.seeds)
            neural += len(runs) if v.kind == "autoencoder" else 0
            fits += len(runs) if v.kind == "pca" else 0
            svd_fits += len(runs) if v.kind == "svd" else 0
            evaluations += sum(len(dims) for _, _, dims in runs)
        n = len(self.datasets)
        return {"datasets": n, "training_runs": n * neural, "pca_fits": n * fits, "svd_fits": n * svd_fits,
                "retrieval_evaluations": n * (evaluations + 1) if self.phase == "confirmation" else 0}

    def resolve_request(self, request):
        # CLI-explicit runtime settings win; variant settings are never overridden by legacy flags.
        explicit = request.extras.get("explicit_options", ())
        settings = {k: v for k, v in self.training.items() if k not in explicit}
        return replace(request, **settings, seeds=self.seeds)


def _dimensions(values, label):
    if (not isinstance(values, list) or not values
            or any(type(d) is not int or d < 1 for d in values) or len(set(values)) != len(values)):
        raise MLXUserError(f"{label} must contain distinct positive integer dimensions.")
    return tuple(values)


def load_experiment_config(path):
    data = read_json(path)
    allowed = {"schema_version", "phase", "datasets", "seeds", "training", "variants", "secondary", "selection"}
    if set(data) - allowed or data.get("schema_version") != 2 or data.get("phase") not in ("pilot", "confirmation"):
        raise MLXUserError("Experiment config requires schema_version 2 and phase pilot or confirmation; unknown fields are rejected.")
    datasets, seeds = data.get("datasets"), data.get("seeds")
    if (not isinstance(datasets, list) or len(datasets) < 2
            or any(not isinstance(d, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", d) for d in datasets)
            or len(set(datasets)) != len(datasets)):
        raise MLXUserError("Experiment requires at least two distinct dataset directory names.")
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise MLXUserError("Experiment requires distinct integer seeds (at least two).")
    training = data.get("training", {})
    allowed_training = {"hidden_dim", "epochs", "batch_size", "lr", "val_ratio", "device"}
    if not isinstance(training, dict) or set(training) - allowed_training:
        raise MLXUserError("Unsupported experiment training setting.")
    for key in ("hidden_dim", "epochs", "batch_size"):
        if key in training and (type(training[key]) is not int or training[key] < 1):
            raise MLXUserError(f"Training {key} must be a positive integer.")
    for key in ("lr", "val_ratio"):
        if key in training and (type(training[key]) not in (int, float) or not math.isfinite(training[key]) or training[key] <= 0):
            raise MLXUserError(f"Training {key} must be finite and positive.")
    variants = []
    if not isinstance(data.get("variants"), list) or not data["variants"]:
        raise MLXUserError("Provide at least one experiment variant.")
    for item in data["variants"]:
        if not isinstance(item, dict) or set(item) - {"name", "kind", "model", "loss", "dimensions", "evaluation_dimensions", "model_config", "loss_config"}:
            raise MLXUserError("Invalid experiment variant fields.")
        name, kind = item.get("name"), item.get("kind", "autoencoder")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or kind not in ("autoencoder", "pca", "svd", "truncate"):
            raise MLXUserError("Variant requires a safe unique name and kind autoencoder, pca, svd, or truncate.")
        dims = _dimensions(item.get("dimensions"), name)
        evaluation = _dimensions(item["evaluation_dimensions"], name) if "evaluation_dimensions" in item else ()
        if evaluation and (len(dims) != 1 or max(evaluation) > dims[0] or kind == "truncate"):
            raise MLXUserError("Shared evaluation dimensions require one training width and cannot exceed it.")
        for key in ("model_config", "loss_config"):
            if not isinstance(item.get(key, {}), dict):
                raise MLXUserError(f"{name}: {key} must be an object.")
        if any(not isinstance(item.get(key, default), str) or not item.get(key, default).strip()
               for key, default in (("model", "simple"), ("loss", "mse"))):
            raise MLXUserError("Variant model and loss must be nonempty registry references.")
        if kind != "autoencoder" and any(key in item for key in ("model", "loss", "model_config", "loss_config")):
            raise MLXUserError("Projection and truncation controls do not accept model or loss options.")
        if data["phase"] == "pilot" and kind != "autoencoder":
            raise MLXUserError("Pilot configuration accepts autoencoders only; controls run during confirmation.")
        variants.append(ExperimentVariant(name, kind, item.get("model", "simple"), item.get("loss", "mse"),
                                         dims, evaluation, item.get("model_config", {}), item.get("loss_config", {})))
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise MLXUserError("Variant names must be unique.")
    secondary = data.get("secondary", [])
    if not isinstance(secondary, list) or any(not isinstance(pair, list) or len(pair) != 2
            or any(n not in names for n in pair) or pair[0] == pair[1] for pair in secondary):
        raise MLXUserError("Secondary comparisons require two distinct existing variant names.")
    if len({tuple(pair) for pair in secondary}) != len(secondary):
        raise MLXUserError("Secondary comparisons must be unique.")
    dimensions_by_name = {v.name: set(v.evaluation_dimensions or v.dimensions) for v in variants}
    for candidate, control in secondary:
        if dimensions_by_name[candidate] != dimensions_by_name[control]:
            raise MLXUserError("Secondary comparisons require identical evaluation dimensions.")
    if not isinstance(data.get("selection", {}), dict):
        raise MLXUserError("Experiment selection metadata must be an object.")
    return ExperimentConfig(data["phase"], tuple(datasets), tuple(seeds), tuple(variants),
                            tuple(tuple(p) for p in secondary), training, data.get("selection", {}))


def run_name(variant, width, seed):
    return f"{variant.name}-{width}/seed-{seed}" if seed is not None else f"{variant.name}-{width}"
