"""Reproducible local-CUDA YOLOX adapter experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_data import load_prepared_adapter_dataset
from mlx.modes.object_detection.adapter_metrics import measure_precision_recall
from mlx.modes.object_detection.adapter_baselines import LoadAdapterBaseline, baseline_root, baseline_provenance
from mlx.modes.object_detection.evaluation import normalize_detection_metrics
from mlx.modes.object_detection.libreyolo.adapter_backend import (
    MODEL_SIZES,
    CollectAdapterEnvironment,
    VerifyFoundationCheckpoint,
    build_experimental_yolox,
    require_experiment_device,
)


METHODS = (
    "frozen", "head-only", "full-finetune", "bottleneck", "ssf", "lora",
    "convpass", "conv-adapter", "drax", "drax-hybrid",
)
FOUNDATION = Path("~/Desktop/object-detection-models/foundational-yolox-l.pt")
DEFAULT_DATASET = Path("~/Desktop/datasets/object-detection/dawn/processed")
DEFAULT_OUTPUT = Path("~/Desktop/experiments/yolox-l-adapters")


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

    @classmethod
    def from_config(cls, config: dict) -> "AdapterExperimentRequest":
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
            dataset=Path(config.get("dataset_path") or DEFAULT_DATASET).expanduser().resolve(),
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
        )
        request.validate(width=int(value("width", 640)))
        return request

    def validate(self, *, width: int | None = None) -> None:
        if self.model not in MODEL_SIZES:
            raise MLXUserError(f"--model must be one of {', '.join(MODEL_SIZES)}")
        if not self.methods or set(self.methods) - set(METHODS):
            raise MLXUserError(f"--methods must use {', '.join(METHODS)}")
        positive = (self.epochs, self.batch_size, self.gradient_accumulation, self.reduction, self.rank)
        if not self.seeds or any(item < 1 for item in positive):
            raise MLXUserError("Seeds must be nonempty and numeric experiment settings must be positive")
        if not math.isfinite(self.alpha) or not math.isfinite(self.lr) or self.lr <= 0:
            raise MLXUserError("Adapter alpha must be finite and learning rate must be positive")
        if self.target not in {"backbone", "neck", "backbone+neck"}:
            raise MLXUserError("--adapter-target must be backbone, neck, or backbone+neck")
        if self.image_size < 32 or self.image_size % 32:
            raise MLXUserError("--height must be a multiple of 32 and at least 32")
        if width is not None and width != self.image_size:
            raise MLXUserError("YOLOX adapter experiments require --width equal to --height")


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _flatten_tensors(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _flatten_tensors(item)]
    return []


def _identity_error(model: torch.nn.Module, inject) -> float:
    model.eval()
    device = next(model.parameters()).device
    sample = torch.randn(1, 3, 128, 128, device=device)
    with torch.inference_mode():
        expected = _flatten_tensors(model(sample))
    # Modules created under inference_mode contain inference tensors that
    # optimizers cannot update. Injection must happen in normal mode.
    inject()
    with torch.inference_mode():
        actual = _flatten_tensors(model(sample))
    if len(expected) != len(actual):
        raise MLXUserError("Adapter injection changed the YOLOX output structure")
    return max(
        (float((left - right).abs().max().item()) for left, right in zip(expected, actual)),
        default=0.0,
    )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _restore_best_checkpoint(model: torch.nn.Module, result: dict) -> tuple[Path, dict]:
    """Restore the validation-selected weights before test evaluation/export."""
    from libreyolo.utils.serialization import load_untrusted_torch_file

    path = Path(result.get("best_checkpoint") or "")
    if not path.is_file():
        raise MLXUserError("Training completed without a best validation checkpoint")
    checkpoint = load_untrusted_torch_file(str(path), map_location="cpu")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise MLXUserError(f"Best checkpoint has no model state dict: {path}")
    try:
        loaded = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise MLXUserError(f"Cannot restore validation-selected checkpoint {path}: {exc}") from exc
    if loaded.missing_keys or loaded.unexpected_keys:
        raise MLXUserError(
            f"Best checkpoint restoration was not strict: missing={loaded.missing_keys}, "
            f"unexpected={loaded.unexpected_keys}"
        )
    selection = {
        "selected_checkpoint": "best",
        "selected_checkpoint_path": str(path),
        "selected_epoch": checkpoint.get("best_epoch"),
        "selected_validation_mAP50": checkpoint.get("best_mAP50"),
        "selected_validation_mAP50_95": checkpoint.get("best_mAP50_95"),
    }
    del checkpoint
    return path, selection


def _verify_training_checkpoint(
    result: dict,
    frozen_before: dict[str, torch.Tensor],
    trainable_before: dict[str, torch.Tensor],
) -> tuple[list[tuple[str, float]], bool]:
    """Verify optimizer effects against the non-EMA training state."""
    from libreyolo.utils.serialization import load_untrusted_torch_file

    path = Path(result.get("last_checkpoint") or "")
    if not path.is_file():
        raise MLXUserError("Training completed without a final training checkpoint")
    checkpoint = load_untrusted_torch_file(str(path), map_location="cpu")
    state = checkpoint.get("train_model") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise MLXUserError(f"Final checkpoint has no training state dict: {path}")
    changed_frozen = [
        (name, float((value - state[name]).abs().max().item()))
        for name, value in frozen_before.items()
        if not torch.equal(value, state[name])
    ]
    trainable_changed = any(
        not torch.equal(value, state[name])
        for name, value in trainable_before.items()
    )
    del checkpoint
    return changed_frozen, trainable_changed


class RunAdapterExperiment:
    """Run one or more independently stored adapter conditions."""

    def __init__(self, request: AdapterExperimentRequest):
        self.request = request

    def execute(self) -> list[dict]:
        request = self.request
        request.validate()
        device = require_experiment_device(request.device)
        _, foundation = VerifyFoundationCheckpoint(request.model, request.checkpoint).execute()
        dataset = load_prepared_adapter_dataset(request.dataset)
        expected_classes = [foundation["classes"][index] for index in range(foundation["nc"])]
        if dataset["classes"] != expected_classes:
            raise MLXUserError("Dataset class order differs from foundation checkpoint")
        request.output.mkdir(parents=True, exist_ok=True)
        environment = CollectAdapterEnvironment(
            request.output, device=str(device), amp=request.amp,
            mlx_root=Path(__file__).resolve().parents[3],
            libreyolo_root=Path(__import__("libreyolo").__file__).resolve().parents[1],
        ).execute()
        dataset_summary = {key: value for key, value in dataset.items() if key != "selected_images"}
        self._write_once(request.output / "dataset.json", dataset_summary)
        foundation_summary = {**foundation, "classes": expected_classes}
        self._write_once(request.output / "foundation.json", foundation_summary)
        study = {
            "research_question": (
                "Can parameter-efficient adapters approach full fine-tuning on a shifted "
                "YOLOX-L target domain, and does DraxAdapter improve the performance-efficiency tradeoff?"
            ),
            "model": request.model,
            "foundation_checkpoint": str(request.checkpoint),
            "foundation_sha256": foundation["sha256"],
            "dataset": dataset["dataset"],
            "dataset_selection_sha256": dataset["selection_sha256"],
            "available_methods": list(dict.fromkeys(("frozen", *request.methods))),
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
            methods = tuple(dict.fromkeys(("frozen", *request.methods)))
            for method in methods:
                if (method, seed) in reused:
                    rows.append(dict(reused[(method, seed)]))
                    continue
                existing = request.output / method / f"seed-{seed}" / "metrics.json"
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
                        "method": method,
                        "seed": seed,
                        "learning_rate": request.lr,
                        "adapter_rank": request.rank if method in {"lora", "drax-hybrid"} else None,
                        "adapter_reduction": request.reduction if method in {"bottleneck", "convpass", "conv-adapter", "drax", "drax-hybrid"} else None,
                        "adapter_alpha": request.alpha if method not in {"frozen", "head-only", "full-finetune"} else None,
                        "train_head": request.train_head,
                        "adapter_target": request.target if method not in {"frozen", "head-only", "full-finetune"} else None,
                    }
                    if metrics.get("status") != "completed" or any(
                        metrics.get(key) != value for key, value in expected.items()
                    ):
                        raise MLXUserError(
                            f"Existing completed run is incompatible at {existing.parent}"
                        )
                    rows.append(metrics)
                    continue
                rows.append(self._run_one(method, seed, dataset, foundation, device))
        return rows

    @staticmethod
    def _write_once(path: Path, value) -> None:
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise MLXUserError(f"Existing study metadata differs at {path}; use a new output root")
            return
        _write_json(path, value)

    def _run_one(self, method, seed, manifest, info, device):
        from libreyolo.adapters import adapter_state_dict, count_parameters, inject_adapters, yolox_targets
        if method == "drax-hybrid":
            from mlx.core.random import seed_everything
            seed_everything(seed)

        run_dir = self.request.output / method / f"seed-{seed}"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise MLXUserError(f"Run directory already contains data: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        model, current = VerifyFoundationCheckpoint(self.request.model, self.request.checkpoint).execute()
        if current["sha256"] != info["sha256"]:
            raise MLXUserError("Foundation checkpoint changed during the experiment")
        model.to(device)
        targets = {}
        identity_error = None
        if method == "frozen":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        elif method == "head-only":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in model.head.parameters():
                parameter.requires_grad_(True)
        elif method != "full-finetune":
            targets = yolox_targets(model, self.request.target, method)
            identity_error = _identity_error(
                model,
                lambda: inject_adapters(
                    model, method, targets, reduction=self.request.reduction,
                    rank=self.request.rank, alpha=self.request.alpha,
                    train_head=self.request.train_head,
                ),
            )
            if identity_error != 0.0:
                raise MLXUserError(
                    f"Identity verification failed for {method}: max error {identity_error}"
                )
        counts = count_parameters(model)
        wrapper = build_experimental_yolox(model, info, str(device))
        wrapper._adapter_expected_trainable = counts["trainable"]
        frozen_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        frozen_buffers = {
            f"{name}.{buffer_name}": value.detach().cpu().clone()
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and all(not parameter.requires_grad for parameter in module.parameters())
            for buffer_name, value in module.named_buffers(recurse=False)
        }
        frozen_before.update(frozen_buffers)
        trainable_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        split_counts = manifest["splits"]
        effective_batch = self.request.batch_size * self.request.gradient_accumulation
        config = {
            "experiment_id": f"{method}-seed-{seed}",
            "method": method,
            "seed": seed,
            "model": self.request.model,
            "adapter_initialization_seed": seed if method == "drax-hybrid" else None,
            "foundation_checkpoint": info["checkpoint"],
            "checkpoint_sha256": info["sha256"],
            "dataset": manifest["dataset"],
            "dataset_path": str(self.request.dataset),
            "dataset_selection_sha256": manifest["selection_sha256"],
            "train_images": split_counts["train"]["images"],
            "val_images": split_counts["val"]["images"],
            "test_images": split_counts["test"]["images"],
            "epochs": 0 if method == "frozen" else self.request.epochs,
            "physical_batch_size": self.request.batch_size,
            "gradient_accumulation": self.request.gradient_accumulation,
            "effective_batch_size": effective_batch,
            "image_size": self.request.image_size,
            "device": str(device),
            "amp": self.request.amp,
            "optimizer": "adamw",
            "learning_rate": self.request.lr,
            "adapter_target": self.request.target if method not in {"frozen", "head-only", "full-finetune"} else None,
            "adapter_reduction": self.request.reduction if method in {"bottleneck", "convpass", "conv-adapter", "drax", "drax-hybrid"} else None,
            "adapter_rank": self.request.rank if method in {"lora", "drax-hybrid"} else None,
            "adapter_alpha": self.request.alpha if method not in {"frozen", "head-only", "full-finetune"} else None,
            "train_head": self.request.train_head,
            "injected_modules": list(targets),
            "identity_max_abs_error": identity_error,
            "trainable_params": counts["trainable"],
            "frozen_params": counts["frozen"],
            "total_params": counts["total"],
            "trainable_percent": counts["trainable_percent"],
        }
        _write_json(run_dir / "config.json", config)

        gpu = device.type == "cuda"
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        training_seconds = 0.0
        checkpoint_size = 0
        result = None
        observed_gradients = set()
        hooks = []
        if method == "drax-hybrid":
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    hooks.append(parameter.register_hook(
                        lambda gradient, key=name: observed_gradients.add(key)
                    ))
        try:
            if method != "frozen":
                started = time.perf_counter()
                result = wrapper.train(
                    data=str(self.request.dataset / "data.yaml"),
                    epochs=self.request.epochs,
                    batch=self.request.batch_size,
                    nbs=effective_batch,
                    imgsz=self.request.image_size,
                    device=str(device),
                    workers=self.request.workers,
                    seed=seed,
                    lr0=self.request.lr,
                    optimizer="adamw",
                    project=str(run_dir),
                    name="training",
                    exist_ok=True,
                    pretrained=True,
                    resume=False,
                    amp=self.request.amp,
                    eval_interval=1,
                    no_aug_epochs=0,
                    mosaic_prob=0.0,
                    mixup_prob=0.0,
                    flip_prob=0.5,
                    hsv_prob=1.0,
                )
                if gpu:
                    torch.cuda.synchronize(device)
                training_seconds = time.perf_counter() - started
                epoch_metrics = result.get("epoch_metrics") or []
                if any(not math.isfinite(float(event["train_loss"])) for event in epoch_metrics):
                    raise MLXUserError(f"Nonfinite training loss during {method}")
                if method == "drax-hybrid" and not observed_gradients:
                    raise MLXUserError("Hybrid training produced no adapter gradients")
                changed_frozen, trainable_parameters_changed = _verify_training_checkpoint(
                    result, frozen_before, trainable_before
                )
                foundation_unchanged_before_selection = not changed_frozen
                if not foundation_unchanged_before_selection:
                    raise MLXUserError(
                        f"Frozen foundation parameters changed during {method} training: "
                        f"{changed_frozen[:10]} ({len(changed_frozen)} tensors total)"
                    )
                if not trainable_parameters_changed:
                    raise MLXUserError(f"No trainable parameter changed during {method} training")

                artifact, selection = _restore_best_checkpoint(model, result)
                if method != "full-finetune":
                    with torch.no_grad():
                        current_parameters = model.state_dict()
                        for name, value in frozen_before.items():
                            current_parameters[name].copy_(value.to(current_parameters[name].device))
                current_parameters = model.state_dict()
                foundation_parameters_unchanged = all(
                    torch.equal(value, current_parameters[name].detach().cpu())
                    for name, value in frozen_before.items()
                )
                selected_trainable_parameters_changed = any(
                    not torch.equal(value, current_parameters[name].detach().cpu())
                    for name, value in trainable_before.items()
                )
                if not foundation_parameters_unchanged:
                    raise MLXUserError(
                        f"Frozen foundation parameters differ after selecting {method} checkpoint"
                    )
                if not selected_trainable_parameters_changed:
                    raise MLXUserError(
                        f"Validation-selected {method} checkpoint has no trainable update"
                    )
                config.update(selection)
                config.update(
                    {
                        "foundation_parameters_unchanged": foundation_parameters_unchanged,
                        "frozen_buffers_verified": len(frozen_buffers),
                        "adapter_gradient_tensors": len(observed_gradients) if method == "drax-hybrid" else None,
                        "trainable_parameters_changed": trainable_parameters_changed,
                        "selected_trainable_parameters_changed": selected_trainable_parameters_changed,
                    }
                )
                _write_json(run_dir / "config.json", config)
                if method not in {"head-only", "full-finetune"}:
                    artifact_dir = run_dir / "adapter"
                    artifact_dir.mkdir()
                    artifact = artifact_dir / "checkpoint.pt"
                    torch.save({"state": adapter_state_dict(model), "config": config}, artifact)
                    weights = run_dir / "training" / "weights"
                    if weights.exists():
                        for redundant in weights.glob("*.pt"):
                            redundant.unlink()
                checkpoint_size = artifact.stat().st_size if artifact.is_file() else 0
                self._write_training_csv(run_dir / "training.csv", result.get("epoch_metrics") or [], gpu)

            native = wrapper.val(
                data=str(self.request.dataset / "data.yaml"), split="test",
                imgsz=self.request.image_size, batch=self.request.batch_size,
                device=str(device), workers=self.request.workers, conf=0.001, iou=0.6,
                verbose=False, save_json=False, save_plots=False,
                save_dir=str(run_dir / "evaluation"),
            )
            metrics = normalize_detection_metrics(native)
            metrics.update(
                measure_precision_recall(
                    wrapper, self.request.dataset, image_size=self.request.image_size
                )
            )
            latency = self._forward_latency_ms(wrapper, device)
            peak_allocated = torch.cuda.max_memory_allocated(device) if gpu else None
            peak_reserved = torch.cuda.max_memory_reserved(device) if gpu else None
            seconds_per_epoch = training_seconds / self.request.epochs if method != "frozen" else 0.0
            images_per_second = (
                split_counts["train"]["images"] * self.request.epochs / training_seconds
                if training_seconds else 0.0
            )
            metrics.update(
                {
                    **config,
                    "mAP50": metrics.pop("map_50"),
                    "mAP50_95": metrics.pop("map_50_95"),
                    "training_seconds": training_seconds,
                    "seconds_per_epoch": seconds_per_epoch,
                    "images_per_second": images_per_second,
                    "peak_cuda_memory_mb": peak_allocated / 2**20 if peak_allocated else None,
                    "peak_cuda_reserved_mb": peak_reserved / 2**20 if peak_reserved else None,
                    "checkpoint_size_mb": checkpoint_size / 2**20,
                    "inference_latency_ms": latency,
                    "status": "completed",
                }
            )
            self._write_metrics(run_dir, metrics)
            return metrics
        except (RuntimeError, MLXUserError) as exc:
            failure = {**config, "status": "failed", "error": str(exc)}
            if "out of memory" in str(exc).lower():
                failure["failure_type"] = "cuda_oom"
                if gpu:
                    torch.cuda.empty_cache()
            _write_json(run_dir / "metrics.json", failure)
            if isinstance(exc, MLXUserError):
                raise
            raise MLXUserError(
                f"{method} run failed for seed {seed} without changing its configuration: {exc}"
            ) from exc
        finally:
            for hook in hooks:
                hook.remove()

    def _write_training_csv(self, path: Path, events: list[dict], gpu: bool) -> None:
        fields = (
            "epoch", "train_loss", "validation_loss", "mAP50", "mAP50_95",
            "precision", "recall", "learning_rate", "epoch_seconds",
            "peak_cuda_memory_mb",
        )
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for event in events:
                val = event.get("val_metrics") or {}
                lr = event.get("lr") or {}
                writer.writerow(
                    {
                        "epoch": event.get("epoch"),
                        "train_loss": event.get("train_loss"),
                        "validation_loss": val.get("metrics/loss"),
                        "mAP50": val.get("metrics/mAP50"),
                        "mAP50_95": val.get("metrics/mAP50-95"),
                        # LibreYOLO's historical validation aliases expose AP
                        # and AR under precision/recall. Leave these blank here
                        # rather than mislabel them as fixed-threshold metrics.
                        "precision": None,
                        "recall": None,
                        "learning_rate": next(iter(lr.values()), None),
                        "epoch_seconds": event.get("epoch_seconds"),
                        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if gpu else None,
                    }
                )

    @staticmethod
    def _write_metrics(run_dir: Path, metrics: dict) -> None:
        _write_json(run_dir / "metrics.json", metrics)
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=metrics)
            writer.writeheader()
            writer.writerow(metrics)
        (run_dir / "summary.md").write_text(
            f"# {metrics['method']}, seed {metrics['seed']}\n\n"
            f"- mAP50: {metrics['mAP50']:.4f}\n"
            f"- mAP50-95: {metrics['mAP50_95']:.4f}\n"
            f"- Trainable parameters: {metrics['trainable_params']:,} / {metrics['total_params']:,}\n"
            f"- Status: {metrics['status']}\n",
            encoding="utf-8",
        )

    def _forward_latency_ms(self, wrapper, device: torch.device) -> float:
        model = wrapper.model.eval()
        sample = torch.zeros(1, 3, self.request.image_size, self.request.image_size, device=device)
        with torch.inference_mode():
            for _ in range(5):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(20):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return 1000 * (time.perf_counter() - started) / 20
