"""Self-contained MLX adapter-detector checkpoints, without pickled modules."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import torch

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import _flatten_tensors
from mlx.modes.object_detection.adapter_slices import StudyRun
from mlx.modes.object_detection.feature_adapters import inject_adapters
from mlx.modes.object_detection.libreyolo.adapter_backend import (
    VerifyFoundationCheckpoint, build_experimental_yolox, require_experiment_device,
)
from mlx.modes.object_detection.libreyolo.adapter_slice_backend import ReconstructAdapterModel


FORMAT = "mlx-yolox-adapter-detector-v1"


class LoadAdapterDetector:
    """Load a complete adapter detector; no external foundation file is needed."""

    def __init__(self, checkpoint, *, device="cpu"):
        self.checkpoint = Path(checkpoint)
        self.device = device

    def execute(self):
        from libreyolo.models.yolox.nn import LibreYOLOXModel

        device = require_experiment_device(self.device)
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            if payload.get("format") != FORMAT:
                raise ValueError(f"Expected {FORMAT}")
            config = payload["config"]
            model = LibreYOLOXModel(config=payload["size"], nb_classes=len(payload["names"]))
            targets = {name: model.get_submodule(name).out_channels for name in config["injected_modules"]}
            inject_adapters(model, config["method"], targets,
                            reduction=config.get("adapter_reduction") or 8,
                            rank=config.get("adapter_rank") or 8,
                            alpha=config.get("adapter_alpha", 1.0),
                            train_head=config.get("train_head", False))
            model.load_state_dict(payload["model"], strict=True)
            model.to(device).eval()
            info = {"checkpoint": str(self.checkpoint), "size": payload["size"],
                    "nc": len(payload["names"]), "classes": dict(enumerate(payload["names"]))}
            return build_experimental_yolox(model, info, str(device))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            raise MLXUserError(f"Cannot load adapter detector {self.checkpoint}: {exc}") from exc


class ExportBestAdapterDetector:
    """Choose the best validation seed, verify reload, then publish atomically."""

    def __init__(self, study, destination, *, seeds, method="drax-residual-fusion", device="cuda"):
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
        wrapper = ReconstructAdapterModel(chosen.config["foundation_checkpoint"], foundation, device).execute(chosen)
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
            restored = LoadAdapterDetector(temporary, device=str(device)).execute()
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
