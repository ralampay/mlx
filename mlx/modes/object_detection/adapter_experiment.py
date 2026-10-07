"""Provider-independent ordering and artifacts for adapter experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


from mlx.core.exceptions import MLXUserError
from mlx.core.artifacts import write_json_atomic as _write_json
from .adapter_contracts import AdapterExperimentBackend
from mlx.modes.object_detection.adapter_baselines import LoadAdapterBaseline, baseline_root, baseline_provenance
from mlx.modes.object_detection.feature_adapters import DEFAULT_FEATURE_ADAPTER_REGISTRY
from mlx.modes.object_detection.prepared_adapter_data import load_prepared_adapter_dataset
from mlx.modes.object_detection.adapter_tensors import _flatten_tensors
from mlx.modes.object_detection.adapter_configuration import FOUNDATION, DEFAULT_DATASET, DEFAULT_OUTPUT


TRAINING_STRATEGIES = ("frozen", "head-only", "full-finetune")
# Historical public ordering; validation uses the injected registry below.
METHODS = (
    *TRAINING_STRATEGIES, "bottleneck", "ssf", "lora", "convpass", "conv-adapter",
    "drax", "drax-hybrid", "drax-spatial", "drax-residual-fusion",
)



@dataclass(frozen=True)
class AdapterExperimentRequest:
    model: str
    checkpoint: Path
    dataset: Path
    output: Path
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    epochs: int = 12
    batch_size: int = 1
    gradient_accumulation: int = 1
    image_size: int = 640
    device: str = "cuda"
    workers: int = 0
    amp: bool = True
    reduction: int = 8
    rank: int = 8
    alpha: float = 1.0
    target: str = "neck"
    train_head: bool = False
    lr: float = 0.0001
    baseline_study: Path | None = None
    head_policy: str = "preserve"
    condition_id: str | None = None
    max_labels: int = 50
    max_detections: int = 300
    seed_adapter_initialization: bool = False

    def method_id(self, method: str) -> str:
        return self.condition_id or method

    @classmethod
    def from_config(cls, config: dict, *, registry=None) -> "AdapterExperimentRequest":
        explicit = set(config.get("_explicit_options") or ())

        def value(name, default):
            return config.get(name, default) if ("_explicit_options" not in config or name in explicit) else default

        model = config.get("model") or "yolox-l"
        checkpoint = config.get("model_path") or (FOUNDATION if model == "yolox-l" else None)
        if checkpoint is None:
            raise MLXUserError("Provide --checkpoint for this model")
        methods_text = config.get("methods") or config.get("adapter") or "drax"
        methods = tuple(dict.fromkeys(item.strip() for item in methods_text.split(",") if item.strip()))
        seeds_text = config.get("experiment_seeds") or str(
            config.get("random_seed") if config.get("random_seed") is not None else 42
        )
        try:
            seeds = tuple(dict.fromkeys(int(item.strip()) for item in str(seeds_text).split(",")))
        except ValueError as exc:
            raise MLXUserError("--experiment-seeds must be comma-separated integers") from exc
        request = cls(
            model=model,
            checkpoint=Path(checkpoint).expanduser().resolve(),
            dataset=Path(value("dataset_path", DEFAULT_DATASET) or DEFAULT_DATASET).expanduser().resolve(),
            output=Path(config.get("output_path") or DEFAULT_OUTPUT).expanduser().resolve(),
            methods=methods,
            seeds=seeds,
            epochs=int(value("epochs", 12)),
            batch_size=int(value("batch_size", 1)),
            gradient_accumulation=int(config.get("gradient_accumulation") or 1),
            image_size=int(value("height", 640)),
            device=str(config.get("device") or "cuda"),
            workers=int(value("workers", 0)),
            amp=bool(value("amp", True)),
            reduction=int(config.get("adapter_reduction", 8)),
            rank=int(config.get("adapter_rank", 8)),
            alpha=float(config.get("adapter_alpha", 1.0)),
            target=str(config.get("adapter_target", "neck")),
            train_head=bool(config.get("train_head", False)),
            lr=float(config.get("lr0") or 0.0001),
            baseline_study=Path(config["baseline_study"]).expanduser().resolve() if config.get("baseline_study") else None,
            head_policy=str(config.get("head_policy") or "preserve"),
            condition_id=config.get("condition_id"),
        )
        request.validate(width=int(value("width", 640)), registry=registry)
        return request

    def validate(self, *, width: int | None = None, registry=None) -> None:
        if self.max_labels < 1 or self.max_detections < 1:
            raise MLXUserError("Label and detection capacities must be positive")
        if self.condition_id is not None:
            import re
            if len(self.methods) != 1 or self.head_policy != "reset-classifiers" or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.condition_id):
                raise MLXUserError("A condition ID requires one taxonomy-transfer method and a safe lowercase slug")
        if self.head_policy not in {"preserve", "reset-classifiers"}:
            raise MLXUserError("Unknown head policy")
        if self.head_policy == "reset-classifiers" and (not self.train_head or "frozen" in self.methods or self.baseline_study):
            raise MLXUserError("Taxonomy transfer requires --train-head, no frozen method and no reused baseline")
        methods = (*TRAINING_STRATEGIES, *(registry or DEFAULT_FEATURE_ADAPTER_REGISTRY).names())
        if not self.methods or set(self.methods) - set(methods):
            raise MLXUserError(f"--methods must use {', '.join(methods)}")
        positive = (self.epochs, self.batch_size, self.gradient_accumulation, self.reduction, self.rank)
        if not self.seeds or any(item < 1 for item in positive):
            raise MLXUserError("Seeds must be nonempty and numeric experiment settings must be positive")
        if not math.isfinite(self.alpha) or not math.isfinite(self.lr) or self.lr <= 0:
            raise MLXUserError("Adapter alpha must be finite and learning rate must be positive")
        if self.image_size < 1:
            raise MLXUserError("Image size must be positive")
        if width is not None and width != self.image_size:
            raise MLXUserError("Adapter experiments require --width equal to --height")


class RunAdapterExperiment:
    """Run one or more independently stored adapter conditions."""

    def __init__(self, request: AdapterExperimentRequest, *, backend: AdapterExperimentBackend | None = None, registry=None,
                 dataset_loader=load_prepared_adapter_dataset):
        self.request = request
        self.registry = registry or DEFAULT_FEATURE_ADAPTER_REGISTRY
        if backend is None:
            from .adapter_composition import create_experiment_backend
            backend = create_experiment_backend()
        self.backend = backend
        self.dataset_loader = dataset_loader

    def execute(self) -> list[dict]:
        request = self.request
        request.validate(registry=self.registry)
        self.backend.validate(request)
        device = self.backend.resolve_device(request.device)
        foundation_model, foundation = self.backend.verify_foundation(request)
        transfer = request.head_policy == "reset-classifiers"
        dataset = self.dataset_loader(request.dataset)
        expected_classes = [foundation["classes"][index] for index in range(foundation["nc"])]
        if not transfer and dataset["classes"] != expected_classes:
            raise MLXUserError("Dataset class order differs from foundation checkpoint")
        request.output.mkdir(parents=True, exist_ok=True)
        environment = self.backend.collect_environment(request, device)
        dataset_summary = {key: value for key, value in dataset.items() if key != "selected_images"}
        self._write_once(request.output / "dataset.json", dataset_summary)
        foundation_summary = {**foundation, "classes": expected_classes}
        self._write_once(request.output / "foundation.json", foundation_summary)
        study = {
            **self.backend.study_metadata(request),
            "model": request.model,
            "foundation_checkpoint": str(request.checkpoint),
            "foundation_sha256": foundation["sha256"],
            "dataset": dataset["dataset"],
            "dataset_selection_sha256": dataset["selection_sha256"],
            "available_methods": list(dict.fromkeys((request.method_id(m) for m in request.methods) if transfer else ("frozen", *request.methods))),
            "planned_seeds": list(request.seeds),
        }
        previous_study = request.output / "study.json"
        if previous_study.is_file():
            declared = json.loads(previous_study.read_text())
            if (set(study["available_methods"]) <= set(declared.get("available_methods", []))
                    and set(request.seeds) <= set(declared.get("planned_seeds", []))):
                study["available_methods"] = declared["available_methods"]
                study["planned_seeds"] = declared["planned_seeds"]
        self._write_once(request.output / "study.json", study)

        reused = {}
        baseline = baseline_root(request.output, request.baseline_study)
        if baseline:
            from mlx.modes.object_detection.adapter_baselines import read_artifact
            previous_environment = read_artifact(baseline / "environment.json")
            for field in ("gpu_name", "pytorch_version", "pytorch_cuda_version", "cudnn_version"):
                if environment[field] != previous_environment.get(field):
                    raise MLXUserError(f"Baseline environment mismatch for {field}")
            expected = {
                "model": request.model, "checkpoint_sha256": foundation["sha256"],
                "dataset_selection_sha256": dataset["selection_sha256"],
                "image_size": request.image_size, "physical_batch_size": request.batch_size,
                "effective_batch_size": request.batch_size * request.gradient_accumulation,
                "gradient_accumulation": request.gradient_accumulation, "amp": request.amp,
                "device": str(device), "epochs": request.epochs,
                "optimizer": "adamw", "learning_rate": request.lr,
                **{f"{split}_images": dataset["splits"][split]["images"] for split in ("train", "val", "test")},
            }
            loader = LoadAdapterBaseline(baseline, expected, request.seeds)
            prior = loader.execute()
            self._write_once(request.output / "baseline.json", baseline_provenance(baseline, prior, expected, request.seeds))
            reused = {(run.method, run.seed): run.metrics for run in prior if run.method == "frozen"}

        rows = []
        for seed in request.seeds:
            methods = tuple(dict.fromkeys(request.methods if transfer else ("frozen", *request.methods)))
            for method in methods:
                if (method, seed) in reused:
                    rows.append(dict(reused[(method, seed)]))
                    continue
                existing = request.output / request.method_id(method) / f"seed-{seed}" / "metrics.json"
                if existing.exists():
                    metrics = json.loads(existing.read_text())
                    expected = {
                        "checkpoint_sha256": foundation["sha256"],
                        "dataset_selection_sha256": dataset["selection_sha256"],
                        "physical_batch_size": request.batch_size,
                        "effective_batch_size": request.batch_size * request.gradient_accumulation,
                        "image_size": request.image_size,
                        "device": str(device),
                        "amp": request.amp,
                        "epochs": 0 if method == "frozen" else request.epochs,
                        "method": request.method_id(method),
                        "seed": seed,
                        "learning_rate": request.lr,
                        "adapter_rank": request.rank if self._uses(method, "rank") else None,
                        "adapter_reduction": request.reduction if self._uses(method, "reduction") else None,
                        "adapter_alpha": request.alpha if method not in {"frozen", "head-only", "full-finetune"} else None,
                        "train_head": request.train_head,
                        "adapter_target": request.target if method not in {"frozen", "head-only", "full-finetune"} else None,
                    }
                    if transfer:
                        expected["head_policy"] = request.head_policy
                        expected["target_classes"] = dataset["classes"]
                        if request.condition_id:
                            expected["adapter_method"] = method
                        if request.max_labels != 50 or request.max_detections != 300:
                            expected.update(max_labels=request.max_labels,max_detections=request.max_detections)
                    if request.seed_adapter_initialization:
                        expected["seed_adapter_initialization"] = True
                        expected["adapter_initialization_seed"] = seed
                    definition = self.registry.entries.get(method)
                    if definition and definition.verify_recorded_targets:
                        expected["injected_modules"] = list(self.backend.targets(
                            foundation_model, request, method, self.registry))
                    if metrics.get("status") != "completed" or any(
                        metrics.get(key) != value for key, value in expected.items()
                    ):
                        raise MLXUserError(
                            f"Existing completed run is incompatible at {existing.parent}"
                        )
                    rows.append(metrics)
                    continue
                rows.append(self._run_one(method, seed, dataset, foundation, device))
                if transfer:
                    self.backend.release(device)
        return rows

    @staticmethod
    def _write_once(path: Path, value) -> None:
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise MLXUserError(f"Existing study metadata differs at {path}; use a new output root")
            return
        _write_json(path, value)

    def _uses(self, method, parameter):
        definition = self.registry.entries.get(method)
        return bool(definition and parameter in definition.parameters)

    def _run_one(self, method, seed, manifest, info, device):
        return self.backend.run_condition(self.request, method, seed, manifest, info, device, self.registry)


def __getattr__(name):
    # Historical helper imports remain lazy; new callers use provider boundaries.
    if name in {"_identity_error", "_restore_best_checkpoint", "_verify_training_checkpoint"}:
        from .libreyolo import adapter_execution
        return getattr(adapter_execution, name)
    if name in {"MODEL_SIZES", "CollectAdapterEnvironment", "VerifyFoundationCheckpoint",
                "build_experimental_yolox", "require_experiment_device"}:
        from .libreyolo import adapter_backend
        return getattr(adapter_backend, name)
    raise AttributeError(name)
