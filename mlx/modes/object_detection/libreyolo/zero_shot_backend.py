"""CUDA-only native validation and synchronized transfer-study benchmarks."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError
from .adapter_backend import require_experiment_device, VerifyFoundationCheckpoint
from .adapter_slice_backend import ReconstructAdapterModel
from ..zero_shot.snapshots import bundled_run


class LibreYOLOTransferEvaluator:
    def __init__(self, device="cuda", workers=4):
        if not str(device).startswith("cuda"):
            raise MLXUserError(
                "Zero-shot inference requires explicit CUDA; CPU fallback is disabled."
            )
        self.device = require_experiment_device(device)
        self.workers = workers

    def load(self, root, entry):
        foundation = Path(root) / "foundation.pt"
        raw, info = VerifyFoundationCheckpoint("yolox-l", foundation).execute()
        del raw
        wrapper = ReconstructAdapterModel(foundation, info, self.device).execute(
            bundled_run(root, entry)
        )
        if next(wrapper.model.parameters()).device != self.device:
            raise MLXUserError(
                "Reconstructed detector is not on the requested CUDA device"
            )
        return wrapper

    def calibrate(self, root, entry, output):
        wrapper = self.load(root, entry)
        trials, selected = [], None
        try:
            with torch.inference_mode():
                for batch in (1, 2, 4, 8):
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    try:
                        x = torch.zeros(batch, 3, 640, 640, device=self.device)
                        for _ in range(3):
                            prediction = wrapper._forward(x)
                            del prediction
                        torch.cuda.synchronize(self.device)
                        peak = torch.cuda.max_memory_reserved(self.device)
                        trials.append(
                            {"batch": batch, "peak_reserved_mib": peak / 2**20}
                        )
                        del x
                        if peak < total * 0.75 and free > total * 0.2:
                            selected = batch
                        else:
                            break
                    except torch.cuda.OutOfMemoryError:
                        trials.append({"batch": batch, "status": "oom"})
                        break
        finally:
            del wrapper
            torch.cuda.empty_cache()
        if selected is None:
            raise MLXUserError(
                "No CUDA batch passed calibration with the required VRAM headroom."
            )
        result = {
            "batch_size": selected,
            "trials": trials,
            "calibration_model": entry["id"],
            "headroom_fraction": 0.25,
        }
        write_json_atomic(output, result)
        return result

    def execute(self, root, entry, dataset, output, batch_size, limit=None):
        from libreyolo.validation.config import ValidationConfig
        from libreyolo.validation.detection_validator import DetectionValidator
        from torch.utils.data import DataLoader, Subset

        class TimedValidator(DetectionValidator):
            def _inference(self, images):
                images = images.to(self.device, non_blocking=True)
                torch.cuda.synchronize(self.device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                result = self.model._forward(images)
                end.record()
                end.synchronize()
                self.forward_ms += start.elapsed_time(end)
                return result

            def _warmup_model(self, n_warmup=10):
                with torch.inference_mode():
                    x = torch.zeros(1, 3, 640, 640, device=self.device)
                    for _ in range(n_warmup):
                        self.model._forward(x)
                    torch.cuda.synchronize(self.device)

            def _compute_metrics(self):
                # Scoring lives in the crowd-aware COCO boundary, once per run.
                write_json_atomic(
                    self.save_dir / "predictions.json", self.coco_evaluator.results
                )
                return {}

        output = Path(output)
        wrapper = self.load(root, entry)
        config = ValidationConfig(
            data=dataset["yaml"],
            split="val",
            batch_size=batch_size,
            imgsz=640,
            device=str(self.device),
            half=False,
            cuda_graph=False,
            conf_thres=0.001,
            iou_thres=0.6,
            max_det=300,
            save_dir=str(output),
            save_json=True,
            verbose=False,
            num_workers=self.workers,
            faster_coco_eval=False,
        )
        validator = TimedValidator(wrapper, config)
        validator.forward_ms = 0.0
        try:
            validator._setup()
            native_dataset = validator.dataloader.dataset
            ids = dataset["latency_image_ids"]
            positions = {int(i): n for n, i in enumerate(native_dataset.ids)}
            if set(positions) != {i["id"] for i in dataset["images"]}:
                raise MLXUserError(
                    "Native dataloader does not cover exactly the declared images"
                )
            indices = [positions[i] for i in ids]
            if limit is not None:
                indices = indices[:limit]
                validator.dataloader = DataLoader(
                    Subset(native_dataset, indices),
                    batch_size=batch_size,
                    collate_fn=validator.dataloader.collate_fn,
                    num_workers=self.workers,
                )
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            validator.run()
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
            memory = {
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(self.device)
                / 2**20,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(self.device)
                / 2**20,
            }
            pipeline_seconds = validator.speed["total"]
            forward_ms = validator.forward_ms
            # Decode/preprocess once outside latency timing; forward excludes H2D and NMS.
            latency = []
            collate = validator.dataloader.collate_fn
            with torch.inference_mode():
                for index in indices:
                    images, *_ = validator._preprocess_batch(
                        collate([native_dataset[index]])
                    )
                    images = images.to(self.device)
                    torch.cuda.synchronize(self.device)
                    for _ in range(3):
                        before = validator.forward_ms
                        prediction = validator._inference(images)
                        latency.append(validator.forward_ms - before)
                        del prediction
            return {
                **memory,
                "model": entry["id"],
                "dataset": dataset["name"],
                "images": validator.seen,
                "batch_size": batch_size,
                "device": str(self.device),
                "precision_mode": "float32",
                "amp": False,
                "pipeline_seconds": pipeline_seconds,
                "inference_wall_seconds": elapsed,
                "images_per_second": validator.seen / pipeline_seconds,
                "forward_ms_per_image": forward_ms / validator.seen,
                "latency_median_ms": float(np.median(latency)),
                "latency_p95_ms": float(np.percentile(latency, 95)),
                "latency_samples": len(latency),
                "latency_repetitions": 3,
                "warmup": 10,
                "total_params": sum(p.numel() for p in wrapper.model.parameters()),
                "trainable_params_at_training": entry["trainable_params"],
                "state_size_mib": entry.get(
                    "state_size_mib",
                    (Path(root) / entry["state"]).stat().st_size / 2**20,
                ),
                "evaluated_image_ids": (
                    [native_dataset.ids[i] for i in indices]
                    if limit
                    else sorted(positions)
                ),
            }
        finally:
            del validator, wrapper
            torch.cuda.empty_cache()
