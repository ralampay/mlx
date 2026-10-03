"""Reproducible YOLOX adapter experiments using LibreYOLO as the model library."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import platform
from pathlib import Path
import time

import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_data import prepare_adapter_dataset
from mlx.modes.object_detection.adapter_metrics import measure_precision_recall
from mlx.modes.object_detection.evaluation import normalize_detection_metrics
from mlx.modes.object_detection.libreyolo.adapter_backend import (
    MODEL_SIZES, VerifyFoundationCheckpoint, build_experimental_yolox,
)


METHODS = ("frozen", "head-only", "full-finetune", "bottleneck", "ssf", "lora",
           "convpass", "conv-adapter", "drax")
FOUNDATION = Path("~/Desktop/object-detection-models/foundational-yolox-l.pt")


@dataclass(frozen=True)
class AdapterExperimentRequest:
    model: str
    checkpoint: Path
    dataset: Path
    output: Path
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    epochs: int = 12
    batch_size: int = 2
    image_size: int = 640
    device: str = "cpu"
    workers: int = 0
    reduction: int = 8
    rank: int = 8
    alpha: float = 1.0
    target: str = "neck"
    train_head: bool = False
    lr: float = 0.0001

    @classmethod
    def from_config(cls, config: dict) -> "AdapterExperimentRequest":
        explicit = set(config.get("_explicit_options") or ())

        def value(name, default):
            return config.get(name, default) if ("_explicit_options" not in config or name in explicit) else default

        model = config.get("model") or "yolox-l"
        checkpoint = config.get("model_path") or (FOUNDATION if model == "yolox-l" else None)
        if checkpoint is None:
            raise MLXUserError("Provide --model-path for this model")
        methods_text = config.get("methods") or config.get("adapter") or "drax"
        methods = tuple(dict.fromkeys(item.strip() for item in methods_text.split(",") if item.strip()))
        seeds_text = config.get("experiment_seeds") or str(config.get("random_seed") if config.get("random_seed") is not None else 42)
        try:
            seeds = tuple(dict.fromkeys(int(value.strip()) for value in str(seeds_text).split(",")))
        except ValueError as exc:
            raise MLXUserError("--experiment-seeds must be comma-separated integers") from exc
        request = cls(model=model, checkpoint=Path(checkpoint).expanduser().resolve(),
                      dataset=Path(config.get("dataset_path") or "").expanduser().resolve(),
                      output=Path(config.get("output_path") or "results/yolox-l-adapters").expanduser().resolve(),
                      methods=methods, seeds=seeds, epochs=int(value("epochs", 12)),
                      batch_size=int(value("batch_size", 2)),
                      image_size=int(value("height", 640)), device=str(config.get("device") or "cpu"),
                      workers=int(value("workers", 0)), reduction=int(config.get("adapter_reduction", 8)),
                      rank=int(config.get("adapter_rank", 8)), alpha=float(config.get("adapter_alpha", 1.0)),
                      target=str(config.get("adapter_target", "neck")),
                      train_head=bool(config.get("train_head", False)),
                      lr=float(config.get("lr0") or 0.0001))
        if request.model not in MODEL_SIZES:
            raise MLXUserError(f"--model must be one of {', '.join(MODEL_SIZES)}")
        if not request.methods or set(request.methods) - set(METHODS):
            raise MLXUserError(f"--methods must use {', '.join(METHODS)}")
        if not request.seeds or request.epochs < 1 or request.batch_size < 1 or request.reduction < 1 or request.rank < 1:
            raise MLXUserError("Seeds must be nonempty; epochs, batch size, reduction and rank must be positive")
        if not math.isfinite(request.alpha) or not math.isfinite(request.lr) or request.lr <= 0:
            raise MLXUserError("Adapter alpha must be finite and learning rate must be positive")
        if request.target not in {"backbone", "neck", "backbone+neck"}:
            raise MLXUserError("--adapter-target must be backbone, neck, or backbone+neck")
        if request.image_size < 32 or request.image_size % 32:
            raise MLXUserError("--height must be a multiple of 32 and at least 32")
        if int(value("width", 640)) != request.image_size:
            raise MLXUserError("YOLOX adapter experiments require --width equal to --height")
        return request


class RunAdapterExperiment:
    def __init__(self, request: AdapterExperimentRequest):
        self.request = request

    def execute(self) -> list[dict]:
        request = self.request
        _, foundation_info = VerifyFoundationCheckpoint(request.model, request.checkpoint).execute()
        dataset_dir = request.output / "dataset"
        manifest = prepare_adapter_dataset(request.dataset, dataset_dir, seed=42)
        if manifest["classes"] != [foundation_info["classes"][i] for i in range(foundation_info["nc"])]:
            raise MLXUserError("Dataset class order differs from foundation checkpoint; remap explicitly before training")
        rows = []
        for seed in request.seeds:
            baseline_path = request.output / "frozen" / f"seed-{seed}"
            if (baseline_path / "metrics.json").exists():
                baseline_config = json.loads((baseline_path / "config.json").read_text())
                expected = {"checkpoint_sha256": foundation_info["sha256"],
                            "selection_sha256": manifest["selection_sha256"], "seed": seed,
                            "batch_size": request.batch_size, "image_size": request.image_size,
                            "device": request.device}
                if any(baseline_config.get(key) != value for key, value in expected.items()):
                    raise MLXUserError(f"Frozen baseline settings differ at {baseline_path}; use a new output directory")
                baseline = json.loads((baseline_path / "metrics.json").read_text())
            else:
                baseline = self._run_one("frozen", seed, dataset_dir, manifest, foundation_info)
            rows.append(baseline)
            for method in request.methods:
                if method != "frozen":
                    rows.append(self._run_one(method, seed, dataset_dir, manifest, foundation_info))
        return rows

    def _run_one(self, method, seed, dataset_dir, manifest, info):
        from libreyolo.adapters import adapter_state_dict, count_parameters, inject_adapters, yolox_targets
        raw, current = VerifyFoundationCheckpoint(self.request.model, self.request.checkpoint).execute()
        if current["sha256"] != info["sha256"]:
            raise MLXUserError("Foundation checkpoint changed during the experiment")
        wrapper = build_experimental_yolox(raw, info, self.request.device)
        if method == "frozen":
            for param in raw.parameters():
                param.requires_grad_(False)
        elif method == "head-only":
            for param in raw.parameters():
                param.requires_grad_(False)
            for param in raw.head.parameters():
                param.requires_grad_(True)
        elif method != "full-finetune":
            targets = yolox_targets(raw, self.request.target, method)
            inject_adapters(raw, method, targets, reduction=self.request.reduction,
                            rank=self.request.rank, alpha=self.request.alpha,
                            train_head=self.request.train_head)
        counts = count_parameters(raw)
        wrapper._adapter_expected_trainable = counts["trainable"]
        run_dir = self.request.output / method / f"seed-{seed}"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise MLXUserError(f"Run directory already contains data: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {"model": self.request.model, "checkpoint": info["checkpoint"],
                  "checkpoint_sha256": info["sha256"], "method": method, "seed": seed,
                  "epochs": 0 if method == "frozen" else self.request.epochs,
                  "batch_size": self.request.batch_size, "image_size": self.request.image_size,
                  "optimizer": "adamw", "learning_rate": self.request.lr,
                  "augmentations": {"mosaic_prob": 0.0, "mixup_prob": 0.0,
                                    "flip_prob": 0.5, "hsv_prob": 1.0},
                  "dataset": str(dataset_dir), "source_dataset": str(self.request.dataset),
                  "selection_sha256": manifest["selection_sha256"], "class_mapping": manifest["class_mapping"],
                  "classes": manifest["classes"], "placement": self.request.target,
                  "reduction": self.request.reduction, "rank": self.request.rank,
                  "alpha": self.request.alpha,
                  "train_head": self.request.train_head, "injected_modules": tuple(targets) if method not in {"frozen", "head-only", "full-finetune"} else (),
                  "device": self.request.device, "torch": torch.__version__, "python": platform.python_version(),
                  "platform": platform.platform(), **counts}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        training_seconds = 0.0
        checkpoint_size = 0
        gpu = torch.cuda.is_available() and self.request.device != "cpu"
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        if method != "frozen":
            started = time.perf_counter()
            try:
                result = wrapper.train(data=str(dataset_dir / "data.yaml"), epochs=self.request.epochs,
                    batch=self.request.batch_size, imgsz=self.request.image_size, device=self.request.device,
                    workers=self.request.workers, seed=seed, lr0=self.request.lr, optimizer="adamw",
                    project=str(run_dir), name="training", exist_ok=True, pretrained=True, resume=False,
                    amp=self.request.device != "cpu", no_aug_epochs=0, mosaic_prob=0.0,
                    mixup_prob=0.0, flip_prob=0.5, hsv_prob=1.0)
            except (OSError, ValueError, RuntimeError) as exc:
                raise MLXUserError(f"{method} training failed for seed {seed}: {exc}. "
                                   "Inspect the dataset, device memory, and training log.") from exc
            training_seconds = time.perf_counter() - started
            if method not in {"head-only", "full-finetune"}:
                artifact = run_dir / "adapter.pt"
                torch.save({"state": adapter_state_dict(wrapper.model), "config": config}, artifact)
                for redundant in (run_dir / "training" / "weights").glob("*.pt"):
                    redundant.unlink()
            else:
                artifact = Path(result.get("best_checkpoint") or run_dir / "training" / "weights" / "last.pt")
            checkpoint_size = artifact.stat().st_size if artifact.is_file() else 0
            events = result.get("epoch_metrics") or []
            if events:
                with (run_dir / "training.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted(events[0]))
                    writer.writeheader()
                    writer.writerows(events)
        try:
            native = wrapper.val(data=str(dataset_dir / "data.yaml"), split="test",
                imgsz=self.request.image_size, batch=self.request.batch_size, device=self.request.device,
                workers=self.request.workers, conf=0.001, iou=0.6, verbose=False,
                save_json=False, save_plots=False, save_dir=str(run_dir / "evaluation"))
        except (OSError, ValueError, RuntimeError) as exc:
            raise MLXUserError(f"{method} evaluation failed for seed {seed}: {exc}. "
                               "Inspect the prepared test split and device.") from exc
        metrics = normalize_detection_metrics(native)
        metrics.update(measure_precision_recall(wrapper, dataset_dir, image_size=self.request.image_size))
        denominator = metrics["precision"] + metrics["recall"]
        metrics["f1"] = (2 * metrics["precision"] * metrics["recall"] / denominator
                         if denominator else 0.0)
        latency = self._forward_latency_ms(wrapper)
        peak = torch.cuda.max_memory_allocated() if gpu else None
        metrics.update({"method": method, "seed": seed, "trainable_params": counts["trainable"],
                        "total_params": counts["total"], "trainable_percent": counts["trainable_percent"],
                        "training_seconds": training_seconds, "peak_memory_bytes": peak,
                        "checkpoint_size_bytes": checkpoint_size,
                        "inference_latency_ms": latency})
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        with (run_dir / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        (run_dir / "summary.md").write_text(f"# {method}, seed {seed}\n\n"
            f"mAP50: {metrics['map_50']:.4f}; mAP50-95: {metrics['map_50_95']:.4f}; "
            f"trainable: {counts['trainable']:,} / {counts['total']:,}.\n")
        return metrics

    def _forward_latency_ms(self, wrapper) -> float:
        """Batch-1 raw-model latency, excluding image decode and NMS."""
        model = wrapper.model.eval()
        device = next(model.parameters()).device
        sample = torch.zeros(1, 3, self.request.image_size, self.request.image_size, device=device)
        with torch.inference_mode():
            for _ in range(2):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(10):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return 1000 * (time.perf_counter() - started) / 10
