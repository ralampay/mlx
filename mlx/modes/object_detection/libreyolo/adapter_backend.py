"""LibreYOLO construction and trainer boundary for adapter experiments."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess

import torch
from torch import nn

from mlx.core.exceptions import MLXUserError


MODEL_SIZES = {f"yolox-{size}": size for size in "ntsmlx"}


def require_experiment_device(requested: str) -> torch.device:
    """Resolve an explicit study device without ever falling back to CPU."""
    normalized = requested.strip().lower()
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise MLXUserError(
                "CUDA was requested but is not available. Adapter training was not started. "
                "Run nvidia-smi and the documented PyTorch CUDA diagnostic."
            )
        try:
            device = torch.device(normalized)
            torch.cuda.get_device_properties(device)
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise MLXUserError(
                f"CUDA device {requested!r} is unavailable. Adapter training was not started."
            ) from exc
        return device
    if normalized == "cpu":
        return torch.device("cpu")
    raise MLXUserError("Adapter experiments support explicit --device cpu or cuda[:index]")


def _git_commit(path: Path, *, distribution: str | None = None) -> str | None:
    snapshot = path / ".mlx-source.json"
    if snapshot.is_file():
        return json.loads(snapshot.read_text())["commit"]
    if not (path / ".git").exists():
        if distribution:
            from importlib.metadata import distribution as installed_distribution, PackageNotFoundError
            try:
                metadata = json.loads(installed_distribution(distribution).read_text("direct_url.json") or "{}")
                return metadata.get("vcs_info", {}).get("commit_id")
            except (PackageNotFoundError, ValueError):
                return None
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class CollectAdapterEnvironment:
    """Collect the reproducibility metadata for a local adapter study."""

    def __init__(self, output: Path, *, device: str, amp: bool, mlx_root: Path,
                 libreyolo_root: Path):
        self.output = Path(output)
        self.device = require_experiment_device(device)
        self.amp = bool(amp)
        self.mlx_root = Path(mlx_root)
        self.libreyolo_root = Path(libreyolo_root)

    def execute(self) -> dict:
        gpu_name = None
        gpu_vram = None
        driver = None
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            gpu_name = properties.name
            gpu_vram = properties.total_memory
            try:
                driver = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True, stderr=subprocess.DEVNULL,
                ).splitlines()[self.device.index or 0].strip()
            except (OSError, subprocess.CalledProcessError, IndexError):
                driver = None
        metadata = {
            "gpu_name": gpu_name,
            "gpu_vram_bytes": gpu_vram,
            "gpu_vram_gib": gpu_vram / 1024**3 if gpu_vram else None,
            "nvidia_driver": driver,
            "pytorch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "python_version": platform.python_version(),
            "libreyolo_git_commit": _git_commit(self.libreyolo_root, distribution="libreyolo"),
            "mlx_git_commit": _git_commit(self.mlx_root),
            "device": str(self.device),
            "amp": self.amp,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "environment.json"
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise MLXUserError(f"Environment metadata differs at {path}; use a new output root")
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerifyFoundationCheckpoint:
    def __init__(self, model_name: str, checkpoint_path: Path):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)

    def execute(self):
        from libreyolo.models.yolox.nn import LibreYOLOXModel
        from libreyolo.utils.serialization import load_untrusted_torch_file, validate_checkpoint_metadata

        path = self.checkpoint_path
        if not path.is_file():
            raise MLXUserError(f"Foundation checkpoint not found: {path}")
        try:
            checkpoint = load_untrusted_torch_file(str(path), map_location="cpu")
            validate_checkpoint_metadata(checkpoint)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            raise MLXUserError(f"Cannot read YOLOX foundation checkpoint {path}: {exc}. "
                               "Inspect the file and its LibreYOLO metadata.") from exc
        size = MODEL_SIZES[self.model_name]
        if checkpoint.get("model_family") != "yolox" or checkpoint.get("size") != size or checkpoint.get("task") != "detect":
            raise MLXUserError(f"Checkpoint metadata must identify {self.model_name} detection")
        model = LibreYOLOXModel(config=size, nb_classes=checkpoint["nc"])
        try:
            result = model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise MLXUserError(f"Foundation checkpoint is incompatible with standard {self.model_name}: {exc}") from exc
        info = {
            "checkpoint": str(path),
            "sha256": _sha256(path),
            "checkpoint_format": "PyTorch ZIP checkpoint mapping",
            "state_dict_key": "model",
            "state_dict_tensors": len(checkpoint["model"]),
            "schema_version": checkpoint.get("schema_version"),
            "libreyolo_version": checkpoint.get("libreyolo_version"),
            "family": checkpoint.get("model_family"),
            "size": size,
            "task": checkpoint.get("task"),
            "classes": checkpoint["names"],
            "nc": checkpoint["nc"],
            "image_size": checkpoint["imgsz"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "training_metadata": {
                "epoch": checkpoint.get("epoch"),
                "best_epoch": checkpoint.get("best_epoch"),
                "best_mAP50": checkpoint.get("best_mAP50"),
                "best_mAP50_95": checkpoint.get("best_mAP50_95"),
                "is_ema_weights": checkpoint.get("is_ema_weights"),
                "ema_updates": checkpoint.get("ema_updates"),
                "config": checkpoint.get("config"),
            },
        }
        return model, info


class CalibrateAdapterBatchSize:
    """Probe Drax and full-finetune memory, selecting their conservative minimum."""

    def __init__(self, model_name: str, checkpoint_path: Path, output: Path, *,
                 device: str = "cuda", image_size: int = 640, amp: bool = True,
                 reduction: int = 8, maximum_batch: int = 8,
                 profiles: tuple[str, ...] = ("full-finetune", "drax"),
                 rank: int = 8, alpha: float = 1.0, target: str = "neck", registry=None):
        self.registry = registry
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.output = Path(output)
        self.device = require_experiment_device(device)
        self.image_size = int(image_size)
        self.amp = bool(amp)
        self.reduction = int(reduction)
        self.maximum_batch = int(maximum_batch)
        self.profiles = profiles
        self.rank = rank
        self.alpha = alpha
        self.target = target

    def execute(self) -> dict:
        if self.device.type != "cuda":
            raise MLXUserError("Batch-size calibration for this study requires --device cuda")
        from mlx.modes.object_detection.libreyolo.adapter_targets import yolox_targets
        from mlx.modes.object_detection.feature_adapters import inject_adapters
        from libreyolo.training.autobatch import autobatch

        profiles = {}
        info = None
        if not self.profiles:
            raise MLXUserError("Calibration requires at least one training profile")
        for profile in self.profiles:
            model, info = VerifyFoundationCheckpoint(
                self.model_name, self.checkpoint_path
            ).execute()
            if profile != "full-finetune":
                inject_adapters(
                    model, profile, yolox_targets(model, self.target, profile, registry=self.registry),
                    reduction=self.reduction, rank=self.rank, alpha=self.alpha, registry=self.registry,
                )
            model.to(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            selected = autobatch(
                model, imgsz=self.image_size, amp=self.amp, fraction=0.60,
                default=1, max_probe=self.maximum_batch,
            )
            profiles[profile] = {
                "selected_batch_size": min(selected, self.maximum_batch),
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(self.device) / 2**20,
            }
            del model
            torch.cuda.empty_cache()
        chosen = min(item["selected_batch_size"] for item in profiles.values())
        result = {
            "model": self.model_name,
            "checkpoint_sha256": info["sha256"],
            "device": str(self.device),
            "image_size": self.image_size,
            "amp": self.amp,
            "profiles": profiles,
            "adapter_target": self.target,
            "adapter_reduction": self.reduction,
            "adapter_rank": self.rank,
            "adapter_alpha": self.alpha,
            "probed_batches": [value for value in (1, 2, 4, 8) if value <= self.maximum_batch],
            "selected_physical_batch_size": chosen,
            "target_vram_fraction": 0.60,
            "peak_cuda_memory_mb": max(item["peak_cuda_memory_mb"] for item in profiles.values()),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "calibration.json"
        if path.exists():
            raise MLXUserError(f"Calibration artifact already exists: {path}")
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        torch.cuda.empty_cache()
        return result


class _AdapterYOLOXTrainerMixin:
    """Keep frozen BN statistics and trainable counts fixed during setup."""

    def _train_epoch(self, epoch):
        if not getattr(self.wrapper_model, "_adapter_measure_epochs", False):
            return super()._train_epoch(epoch)
        import time
        gpu = self.device.type == "cuda"
        if gpu:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        result = super()._train_epoch(epoch)
        if gpu:
            torch.cuda.synchronize(self.device)
        self.wrapper_model._adapter_epoch_cuda.append({
            "epoch": epoch + 1,
            "epoch_seconds": time.perf_counter() - started,
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(self.device) / 2**20 if gpu else None,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(self.device) / 2**20 if gpu else None,
        })
        return result

    def _apply_freeze_config(self):
        super()._apply_freeze_config()
        expected = getattr(self.wrapper_model, "_adapter_expected_trainable", None)
        actual = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if expected is not None and actual != expected:
            raise MLXUserError(f"Adapter trainable parameter count changed during trainer setup: "
                               f"expected {expected}, found {actual}. Check checkpoint loading and freeze settings.")
        self._frozen_bn_modules = tuple(module for module in self.model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and all(not p.requires_grad for p in module.parameters()))


def build_experimental_yolox(raw, info, device):
    from libreyolo.models.yolox.model import LibreYOLOX
    from libreyolo.models.yolox.trainer import YOLOXTrainer

    class AdapterYOLOXTrainer(_AdapterYOLOXTrainerMixin, YOLOXTrainer):
        pass

    class ExperimentalYOLOX(LibreYOLOX):
        def _trainer_class(self):
            return AdapterYOLOXTrainer

    wrapper = ExperimentalYOLOX(model_path=None, size=info["size"], nb_classes=info["nc"], device=device)
    wrapper.model = raw.to(wrapper.device)
    wrapper.names = {int(key): value for key, value in info["classes"].items()}
    wrapper.model_path = str(info["checkpoint"])
    return wrapper
