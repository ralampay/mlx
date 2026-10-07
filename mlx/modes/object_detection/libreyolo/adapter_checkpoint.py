"""Self-contained MLX adapter-detector checkpoints, without pickled modules."""

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from pickle import UnpicklingError

import torch

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_tensors import _flatten_tensors
from mlx.modes.object_detection.adapter_slices import StudyRun
from mlx.modes.object_detection.feature_adapters import inject_adapters
from mlx.modes.object_detection.libreyolo.adapter_backend import (
    VerifyFoundationCheckpoint, build_experimental_yolox, require_experiment_device,
)
from mlx.modes.object_detection.libreyolo.adapter_slice_backend import ReconstructAdapterModel


FORMAT = "mlx-yolox-adapter-detector-v1"


class LoadAdapterDetector:
    """Load a complete adapter detector; no external foundation file is needed."""

    def __init__(self, checkpoint, *, device="cpu", registry=None, payload=None):
        self.registry = registry
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.payload = payload

    def execute(self):
        from libreyolo.models.yolox.nn import LibreYOLOXModel

        device = require_experiment_device(self.device)
        try:
            payload = self.payload
            if payload is None:
                payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or payload.get("format") != FORMAT:
                raise ValueError(f"Expected {FORMAT}")
            config = payload["config"]
            names = payload["names"]
            if not isinstance(config, Mapping):
                raise ValueError("config must be a mapping")
            if (not isinstance(names, list) or not names
                    or any(not isinstance(name, str) or not name for name in names)):
                raise ValueError("names must be a nonempty list of class labels")
            if not isinstance(config.get("train_head", False), bool):
                raise ValueError("train_head must be a boolean")
            model = LibreYOLOXModel(config=payload["size"], nb_classes=len(names))
            from .adapter_loading import resolve_adapter_targets
            targets = resolve_adapter_targets(model, config, config["method"], registry=self.registry)
            reduction = config.get("adapter_reduction")
            rank = config.get("adapter_rank")
            inject_adapters(model, config["method"], targets,
                            reduction=8 if reduction is None else reduction,
                            rank=8 if rank is None else rank,
                            alpha=config.get("adapter_alpha", 1.0),
                            train_head=config.get("train_head", False), registry=self.registry)
            model.load_state_dict(payload["model"], strict=True)
            model.to(device).eval()
            info = {"checkpoint": str(self.checkpoint), "size": payload["size"],
                    "nc": len(names), "classes": dict(enumerate(names))}
            return build_experimental_yolox(model, info, str(device))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, EOFError, UnpicklingError) as exc:
            raise MLXUserError(
                f"Cannot load adapter detector {self.checkpoint}: {exc}. "
                "Inspect its configuration, class labels, and tensor state."
            ) from exc


class ExportBestAdapterDetector:
    """Choose the best validation seed, verify reload, then publish atomically."""

    def __init__(self, study, destination, *, seeds, method="drax-residual-fusion", device="cuda", registry=None):
        self.registry = registry
        self.study, self.destination = Path(study), Path(destination)
        self.seeds, self.method, self.device = tuple(seeds), method, device

    def execute(self):
        candidates = []
        for seed in self.seeds:
            directory = self.study / self.method / f"seed-{seed}"
            metrics = json.loads((directory / "metrics.json").read_text())
            config = json.loads((directory / "config.json").read_text())
            if metrics.get("status") != "completed":
                raise MLXUserError(f"Incomplete export candidate: {directory}")
            candidates.append(StudyRun(self.method, seed, directory, metrics, config))
        chosen = select_validation_candidate(candidates)
        model, foundation = VerifyFoundationCheckpoint(chosen.config["model"], chosen.config["foundation_checkpoint"]).execute()
        del model
        device = require_experiment_device(self.device)
        wrapper = ReconstructAdapterModel(chosen.config["foundation_checkpoint"], foundation, device, registry=self.registry).execute(chosen)
        payload = {"format": FORMAT, "size": foundation["size"],
                   "names": [foundation["classes"][i] for i in range(foundation["nc"])],
                   "config": dict(chosen.config),
                   "model": {k: v.detach().cpu().clone() for k, v in wrapper.model.state_dict().items()}}
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.destination.parent, suffix=".pt", delete=False) as stream:
                temporary = Path(stream.name)
            torch.save(payload, temporary)
            restored = LoadAdapterDetector(temporary, device=str(device), registry=self.registry).execute()
            generator = torch.Generator(device=device).manual_seed(42)
            sample = torch.randn(1, 3, 128, 128, generator=generator, device=device)
            with torch.inference_mode():
                before = _flatten_tensors(wrapper.model(sample))
                after = _flatten_tensors(restored.model(sample))
            if len(before) != len(after):
                raise MLXUserError("Export changed model output structure")
            for left, right in zip(before, after):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            backup = None
            if self.destination.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                backup = self.destination.with_name(self.destination.name + f".backup-{stamp}")
                self.destination.replace(backup)
            temporary.replace(self.destination)
            receipt = {"path": str(self.destination), "sha256": sha256_file(self.destination),
                       "selected_seed": chosen.seed, "criterion": "validation mAP50-95; ties use lower seed",
                       "validation_mAP50_95": chosen.config["selected_validation_mAP50_95"],
                       "reload_exact": True, "backup": str(backup) if backup else None}
            write_json_atomic(self.study / "export.json", receipt)
            return receipt
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def select_validation_candidate(candidates):
    import math
    if not candidates or any(not math.isfinite(float(r.config["selected_validation_mAP50_95"])) for r in candidates):
        raise MLXUserError("Export requires finite validation scores for every candidate")
    return min(candidates, key=lambda r: (-float(r.config["selected_validation_mAP50_95"]), r.seed))
