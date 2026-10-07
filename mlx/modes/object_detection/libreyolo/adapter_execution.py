"""LibreYOLO execution of one adapter condition; workflow ordering is MLX-owned."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import time
import torch
from mlx.core.exceptions import MLXUserError
from mlx.core.artifacts import write_json_atomic as _write_json
from mlx.modes.object_detection.adapter_tensors import _flatten_tensors
from mlx.modes.object_detection.adapter_metrics import measure_precision_recall
from mlx.modes.object_detection.evaluation import normalize_detection_metrics
from mlx.modes.object_detection.feature_adapters import DEFAULT_FEATURE_ADAPTER_REGISTRY
from .adapter_backend import (MODEL_SIZES, CollectAdapterEnvironment,
    VerifyFoundationCheckpoint, build_experimental_yolox, require_experiment_device)

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


class LibreYOLOExperimentBackend:
    """Provider operations used by the generic experiment coordinator."""
    def validate(self, request):
        if request.model not in MODEL_SIZES:
            raise MLXUserError(f"--model must be one of {', '.join(MODEL_SIZES)}")
        if request.target not in {"backbone", "neck", "backbone+neck"}:
            raise MLXUserError("--adapter-target must be backbone, neck, or backbone+neck")
        if request.image_size < 32 or request.image_size % 32:
            raise MLXUserError("--height must be a multiple of 32 and at least 32")

    def study_metadata(self, request):
        return {
            "research_question": (
                "Can parameter-efficient adapters approach full fine-tuning on a shifted "
                "YOLOX-L target domain, and does DraxAdapter improve the performance-efficiency tradeoff?"
            ),
        }

    def resolve_device(self, requested):
        return require_experiment_device(requested)

    def verify_foundation(self, request):
        return VerifyFoundationCheckpoint(request.model, request.checkpoint).execute()

    def collect_environment(self, request, device):
        import libreyolo
        return CollectAdapterEnvironment(request.output, device=str(device), amp=request.amp,
            mlx_root=Path(__file__).resolve().parents[4],
            libreyolo_root=Path(libreyolo.__file__).resolve().parents[1]).execute()

    def targets(self, model, request, method, registry):
        from .adapter_targets import yolox_targets
        return yolox_targets(model, request.target, method, registry=registry)

    def run_condition(self, request, method, seed, dataset, foundation, device, registry):
        try:
            return RunLibreYOLOAdapterCondition(
                request, method, seed, dataset, foundation, device, registry=registry,
            ).execute()
        except MLXUserError:
            raise
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise MLXUserError(
                f"Cannot execute adapter {method!r} for seed {seed}: {exc}. "
                "Inspect the adapter definition, model targets, and provider dependencies."
            ) from exc

    def release(self, device):
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


class RunLibreYOLOAdapterCondition:
    def __init__(self, request, method, seed, dataset, foundation, device, *, registry=None):
        self.condition = (method, seed, dataset, foundation, device)
        self.request = request
        self.registry = registry or DEFAULT_FEATURE_ADAPTER_REGISTRY

    def _capability(self, method, name):
        definition = self.registry.entries.get(method)
        return bool(definition and getattr(definition, name))

    def _uses(self, method, parameter):
        definition = self.registry.entries.get(method)
        return bool(definition and parameter in definition.parameters)

    def execute(self):
        method, seed, manifest, info, device = self.condition
        from mlx.modes.object_detection.feature_adapters import adapter_state_dict
        transfer = self.request.head_policy == "reset-classifiers"
        if self.request.seed_adapter_initialization or self._capability(method, "seed_initialization") or transfer:
            from mlx.core.random import seed_everything
            seed_everything(seed)

        run_dir = self.request.output / self.request.method_id(method) / f"seed-{seed}"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise MLXUserError(f"Run directory already contains data: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        model, info, targets, identity_error, counts, wrapper, frozen_before, frozen_buffers, trainable_before = self._prepare_model(
            method, seed, manifest, info, device, transfer)

        split_counts = manifest["splits"]
        effective_batch = self.request.batch_size * self.request.gradient_accumulation
        config = self._configuration(method, seed, manifest, info, device, transfer,
                                     targets, identity_error, counts, model)
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
        if self._capability(method, "verify_gradients") or transfer:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (not transfer or not name.startswith("head.")):
                    hooks.append(parameter.register_hook(
                        lambda gradient, key=name: observed_gradients.add(key)
                    ))
        try:
            if method != "frozen":
                if transfer:
                    from mlx.core.random import seed_everything
                    seed_everything(seed)
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
                    hsv_prob=0.0 if transfer else 1.0,
                    **({"warmup_epochs": min(5, max(0,self.request.epochs-1)), "patience": 0,
                        "save_period": 0,
                        "max_labels": self.request.max_labels, "eval_max_det": self.request.max_detections,
                        "weight_decay": 0.0005, "degrees": 0.0, "translate": 0.0,
                        "shear": 0.0, "mosaic_scale": (1.0,1.0), "mixup_scale": (1.0,1.0)} if transfer else {}),
                )
                if gpu:
                    torch.cuda.synchronize(device)
                training_seconds = time.perf_counter() - started
                epoch_metrics = result.get("epoch_metrics") or []
                if transfer:
                    measurements = {row["epoch"]:row for row in wrapper._adapter_epoch_cuda}
                    for event in epoch_metrics:
                        event.update(measurements[event["epoch"]])
                    _write_json(run_dir / "cuda-epochs.json", wrapper._adapter_epoch_cuda)
                if transfer and len(epoch_metrics) != self.request.epochs:
                    raise MLXUserError(f"Expected {self.request.epochs} epochs, recorded {len(epoch_metrics)}")
                if any(not math.isfinite(float(event["train_loss"])) for event in epoch_metrics):
                    raise MLXUserError(f"Nonfinite training loss during {method}")
                if (self._capability(method, "verify_gradients") or (transfer and targets)) and not observed_gradients:
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
                if transfer and targets:
                    _, adapter_changed = _verify_training_checkpoint(result, {},
                        {name:value for name,value in trainable_before.items() if not name.startswith("head.")})
                    if not adapter_changed:
                        raise MLXUserError("Training changed the head but not the adapter")

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
                if transfer and method in {"head-only", "full-finetune"}:
                    from libreyolo.utils.serialization import load_untrusted_torch_file
                    exported = load_untrusted_torch_file(str(artifact), map_location="cpu")
                    exported["model"] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    exported = {k:v for k,v in exported.items() if k not in {"optimizer","train_model","ema","scaler","rng_state"}}
                    artifact = run_dir / "checkpoint.pt"
                    torch.save(exported,artifact)
                    config["export_checkpoint_path"] = str(artifact)
                    del exported
                config.update(
                    {
                        "foundation_parameters_unchanged": foundation_parameters_unchanged,
                        "frozen_buffers_verified": len(frozen_buffers),
                        "adapter_gradient_tensors": len(observed_gradients) if self._capability(method, "verify_gradients") or (transfer and targets) else None,
                        "trainable_parameters_changed": trainable_parameters_changed,
                        "selected_trainable_parameters_changed": selected_trainable_parameters_changed,
                    }
                )
                _write_json(run_dir / "config.json", config)
                if method not in {"head-only", "full-finetune"}:
                    artifact_dir = run_dir / "adapter"
                    artifact_dir.mkdir()
                    artifact = artifact_dir / "checkpoint.pt"
                    if transfer:
                        from libreyolo.models.yolox.transfer import transfer_state_dict, load_transfer_state_dict
                        state = transfer_state_dict(model)
                        torch.save({"format": "yolox-taxonomy-transfer-v1", "state": state, "config": config}, artifact)
                        load_transfer_state_dict(model, torch.load(artifact, map_location="cpu", weights_only=True)["state"])
                    else:
                        torch.save({"state": adapter_state_dict(model), "config": config}, artifact)
                    weights = run_dir / "training" / "weights"
                    if weights.exists() and not transfer:
                        for redundant in weights.glob("*.pt"):
                            redundant.unlink()
                checkpoint_size = artifact.stat().st_size if artifact.is_file() else 0
                self._write_training_csv(run_dir / "training.csv", result.get("epoch_metrics") or [], gpu)

            native = wrapper.val(
                data=str(self.request.dataset / "data.yaml"), split="test",
                imgsz=self.request.image_size, batch=self.request.batch_size,
                device=str(device), workers=self.request.workers, conf=0.001, iou=0.6,
                verbose=False, save_json=transfer, save_plots=False,
                **({"max_det":self.request.max_detections} if transfer else {}),
                save_dir=str(run_dir / "evaluation"),
            )
            metrics = normalize_detection_metrics(native)
            metrics.update(
                measure_precision_recall(
                    wrapper, self.request.dataset, image_size=self.request.image_size,
                    **({"max_detections":self.request.max_detections} if transfer else {}),
                )
            )
            latency = self._forward_latency_ms(wrapper, device)
            peak_allocated = torch.cuda.max_memory_allocated(device) if gpu else None
            peak_reserved = torch.cuda.max_memory_reserved(device) if gpu else None
            if transfer and gpu:
                peak_allocated = max(row["peak_cuda_memory_mb"] for row in wrapper._adapter_epoch_cuda) * 2**20
                peak_reserved = max(row["peak_cuda_reserved_mb"] for row in wrapper._adapter_epoch_cuda) * 2**20
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

    def _prepare_model(self, method, seed, manifest, info, device, transfer):
        from .adapter_targets import yolox_targets
        from ..feature_adapters import count_parameters, inject_adapters
        model, current = VerifyFoundationCheckpoint(self.request.model, self.request.checkpoint).execute()
        if current["sha256"] != info["sha256"]:
            raise MLXUserError("Foundation checkpoint changed during the experiment")
        if transfer:
            from libreyolo.models.yolox.transfer import reset_classifiers
            initial = reset_classifiers(model, len(manifest["classes"]), seed=seed)
            initial_dir = self.request.output / "initial-heads"
            initial_dir.mkdir(exist_ok=True)
            initial_path = initial_dir / f"seed-{seed}.pt"
            if initial_path.exists():
                previous = torch.load(initial_path, map_location="cpu", weights_only=True)
                if set(previous) != set(initial) or any(not torch.equal(previous[k], v) for k,v in initial.items()):
                    raise MLXUserError("Paired classifier initialization differs")
            else:
                torch.save(initial, initial_path)
            info = {**info, "nc": len(manifest["classes"]), "classes": dict(enumerate(manifest["classes"]))}
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
            targets = yolox_targets(model, self.request.target, method, registry=self.registry)
            identity_error = _identity_error(
                model,
                lambda: inject_adapters(
                    model, method, targets, reduction=self.request.reduction,
                    rank=self.request.rank, alpha=self.request.alpha,
                    train_head=self.request.train_head, registry=self.registry,
                ),
            )
            if identity_error != 0.0:
                raise MLXUserError(
                    f"Identity verification failed for {method}: max error {identity_error}"
                )
        counts = count_parameters(model)
        wrapper = build_experimental_yolox(model, info, str(device))
        wrapper._adapter_expected_trainable = counts["trainable"]
        wrapper._adapter_measure_epochs = transfer
        wrapper._adapter_epoch_cuda = []
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

        return model, info, targets, identity_error, counts, wrapper, frozen_before, frozen_buffers, trainable_before

    def _configuration(self, method, seed, manifest, info, device, transfer,
                       targets, identity_error, counts, model):
        split_counts = manifest["splits"]
        effective_batch = self.request.batch_size * self.request.gradient_accumulation
        config = {
            "experiment_id": f"{self.request.method_id(method)}-seed-{seed}",
            "method": self.request.method_id(method),
            "seed": seed,
            "model": self.request.model,
            "adapter_initialization_seed": seed if self._capability(method, "seed_initialization") else None,
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
            "adapter_reduction": self.request.reduction if self._uses(method, "reduction") else None,
            "adapter_rank": self.request.rank if self._uses(method, "rank") else None,
            "adapter_alpha": self.request.alpha if method not in {"frozen", "head-only", "full-finetune"} else None,
            "train_head": self.request.train_head,
            "injected_modules": list(targets),
            "identity_max_abs_error": identity_error,
            "trainable_params": counts["trainable"],
            "frozen_params": counts["frozen"],
            "total_params": counts["total"],
            "trainable_percent": counts["trainable_percent"],
        }
        if transfer:
            head_count = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
            config.update(head_policy=self.request.head_policy, target_classes=manifest["classes"],
                adapter_method=method,
                max_labels=self.request.max_labels,max_detections=self.request.max_detections,
                head_trainable_params=head_count, adapter_trainable_params=(counts["trainable"]-head_count if targets else 0),
                adapter_initialization_seed=seed, weight_decay=0.0005,
                warmup_epochs=min(5, max(0,self.request.epochs-1)), patience=0,
                scheduler="yoloxwarmcos", evaluation_precision="float32", hsv_prob=0.0,
                flip_prob=0.5, mosaic_prob=0.0, mixup_prob=0.0,
                cuda_memory_scope="training epochs including validation; excludes final test and export")
        if self.request.seed_adapter_initialization:
            config["adapter_initialization_seed"] = seed
            config["seed_adapter_initialization"] = True
        return config

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
                        "peak_cuda_memory_mb": event.get("peak_cuda_memory_mb") if self.request.head_policy == "reset-classifiers" else (torch.cuda.max_memory_allocated() / 2**20 if gpu else None),
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


class CalibrateAdapterExperiment:
    def __init__(self, request, *, registry=None):
        self.request = request
        self.registry = registry or DEFAULT_FEATURE_ADAPTER_REGISTRY

    def execute(self):
        from .adapter_backend import CalibrateAdapterBatchSize
        request = self.request
        request.validate(registry=self.registry)
        backend = LibreYOLOExperimentBackend()
        backend.validate(request)
        backend.collect_environment(request, backend.resolve_device(request.device))
        profiles = request.methods if self.registry is not DEFAULT_FEATURE_ADAPTER_REGISTRY or (
            len(request.methods) == 1 and self.registry.entries.get(request.methods[0])
            and self.registry.resolve(request.methods[0]).verify_gradients
        ) else ("full-finetune", "drax")
        return CalibrateAdapterBatchSize(request.model, request.checkpoint, request.output,
            device=request.device, image_size=request.image_size, amp=request.amp,
            reduction=request.reduction, profiles=profiles, rank=request.rank,
            alpha=request.alpha, target=request.target, registry=self.registry).execute()
