"""Commands coordinating immutable inputs, local evaluation, and resumable artifacts."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
import random
import subprocess
import time
import hashlib

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from .data import InspectTransferDataset, read_json, preserve_identity
from .snapshots import SnapshotTransferModels
from .scoring import ScoreTransferPredictions


class RunTransferStudy:
    def __init__(self, specification, output, evaluator, verifier, phase="all"):
        self.spec, self.output = specification, Path(output).expanduser().resolve()
        self.evaluator, self.verifier, self.phase = evaluator, verifier, phase

    def execute(self):
        self.output.mkdir(parents=True, exist_ok=True)
        import fcntl

        with (self.output / ".study.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MLXUserError(
                    f"Another process owns this study: {self.output}"
                ) from exc
            return self._execute()

    def _execute(self):
        import shutil

        if (
            not (self.output / "models/manifest.json").exists()
            and shutil.disk_usage(self.output).free < 15 * 2**30
        ):
            raise MLXUserError(
                "At least 15 GiB free space is required for model snapshots and study artifacts."
            )
        preserve_identity(self.output / "study.json", self.spec)
        raw, foundation = self.verifier.execute()
        del raw
        models = SnapshotTransferModels(self.spec, self.output, foundation).execute()
        datasets = [
            InspectTransferDataset(d, foundation["classes"]).execute()
            for d in self.spec["datasets"]
        ]
        preserve_identity(self.output / "datasets.json", datasets)
        self._dataset_provenance(datasets)
        self._provenance()
        calibration_path = self.output / "calibration.json"
        calibration = (
            read_json(calibration_path)
            if calibration_path.exists()
            else self.evaluator.calibrate(
                self.output / "models",
                next(m for m in models if m["method"] == "drax-hybrid"),
                calibration_path,
            )
        )
        batch = calibration["batch_size"]
        if self.phase == "prepare":
            return {
                "status": "prepared",
                "output": str(self.output),
                "models": len(models),
                "batch": batch,
            }
        pilot = self.phase == "pilot"
        pairs = [
            (m, d)
            for m in models
            for d in datasets
            if not pilot
            or (
                m["method"] in {"frozen", "lora", "drax-hybrid"}
                and m["seed"] in {None, 1}
            )
        ]
        random.Random(42).shuffle(pairs)
        records = []
        for index, (model, dataset) in enumerate(pairs):
            destination = (
                self.output
                / ("pilot" if pilot else "runs")
                / dataset["name"]
                / model["id"]
            )
            write_json_atomic(
                self.output / "progress.json",
                {
                    "phase": self.phase,
                    "current": str(destination),
                    "index": index + 1,
                    "total": len(pairs),
                    "updated_unix": time.time(),
                },
            )
            result = self._evaluate(
                model, dataset, destination, batch, 24 if pilot else None
            )
            records.append(result)
        if pilot:
            estimate = (
                sum(
                    r["pipeline_seconds"]
                    / r["images"]
                    * next(
                        len(d["images"]) for d in datasets if d["name"] == r["dataset"]
                    )
                    for r in records
                )
                / 3
                * len(models)
            )
            result = {
                "status": "pilot_completed",
                "inference_only_estimate_hours": estimate / 3600,
                "note": "Excludes loading, scoring, latency repetitions, rendering; extrapolated from 24 images.",
                "results": records,
            }
            write_json_atomic(self.output / "pilot.json", result)
            return result
        write_json_atomic(
            self.output / "progress.json",
            {"phase": "inference_complete", "completed": len(records)},
        )
        return {
            "status": "inference_complete",
            "runs": len(records),
            "output": str(self.output),
        }

    def _dataset_provenance(self, datasets):
        from .snapshots import copy_verified

        rows = []
        for dataset in datasets:
            source = Path(dataset["annotations"]).parent.parent
            records = []
            for relative in (
                "manifest.json",
                "validation.json",
                "README.txt",
                "original/LICENSE",
                "original/License.pdf",
                "original/README.md",
                "original/source.json",
            ):
                path = source / relative
                if path.is_file():
                    digest = copy_verified(
                        path,
                        self.output / "dataset-provenance" / dataset["name"] / relative,
                    )
                    records.append({"file": relative, "sha256": digest})
            rows.append({"dataset": dataset["name"], "files": records})
        preserve_identity(self.output / "dataset-provenance/manifest.json", rows)

    def _evaluate(self, model, dataset, output, batch, limit):
        output.mkdir(parents=True, exist_ok=True)
        identity = {
            "model_sha256": model["sha256"],
            "foundation": sha256_file(self.output / "models/foundation.pt"),
            "annotation_sha256": dataset["annotation_sha256"],
            "yaml_sha256": dataset["yaml_sha256"],
            "batch": batch,
            "limit": limit,
            "settings": self.spec["settings"],
            "evaluation_code": self.evaluation_signature,
        }
        preserve_identity(output / "config.json", identity)
        completed = output / "completed.json"
        if completed.exists():
            receipt = read_json(completed)
            for name, digest in receipt["artifacts"].items():
                if sha256_file(output / name) != digest:
                    raise MLXUserError(f"Cached evaluation changed: {output/name}")
            return read_json(output / "metrics.json")
        started = time.perf_counter()
        try:
            timing_path = output / "timing.json"
            # Interrupted scoring can resume from verified predictions; no GPU rerun required.
            if timing_path.exists():
                timing = read_json(timing_path)
                if (
                    sha256_file(output / "predictions.json")
                    != timing["prediction_sha256"]
                ):
                    raise MLXUserError(f"Prediction cache hash mismatch: {output}")
            else:
                timing = self.evaluator.execute(
                    self.output / "models", model, dataset, output, batch, limit
                )
                timing["prediction_sha256"] = sha256_file(output / "predictions.json")
                write_json_atomic(timing_path, timing)
            scoring_start = time.perf_counter()
            scores = ScoreTransferPredictions(
                dataset,
                output / "predictions.json",
                output,
                timing["evaluated_image_ids"],
            ).execute()
            metrics = {
                **timing,
                **scores["all"],
                "method": model["method"],
                "seed": model["seed"],
                "scoring_seconds": time.perf_counter() - scoring_start,
                "evaluation_wall_seconds": time.perf_counter() - started,
                "status": "completed",
                "source_revision": self.source_revision,
            }
            write_json_atomic(output / "metrics.json", metrics)
            names = (
                "config.json",
                "predictions.json",
                "timing.json",
                "metrics.json",
                "scores.json",
                "per-image.json",
                "coco-matches.json.gz",
            )
            write_json_atomic(
                completed, {"artifacts": {n: sha256_file(output / n) for n in names}}
            )
            return metrics
        except Exception as exc:
            write_json_atomic(
                output / "failure.json",
                {
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "configuration": identity,
                    "batch_changed": False,
                },
            )
            raise MLXUserError(
                f"Zero-shot run failed without changing settings: {output}: {exc}"
            ) from exc

    def _provenance(self):
        import torch
        import libreyolo

        path = self.output / "environment.json"
        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(self.evaluator.device),
            "vram_bytes": torch.cuda.get_device_properties(
                self.evaluator.device
            ).total_memory,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
            "amp": False,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "packages": sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
            ),
        }
        if not path.exists():
            write_json_atomic(path, environment)
        sources = self.output / "source"
        sources.mkdir(exist_ok=True)
        records = []
        for name, root in (
            ("mlx", Path(__file__).resolve().parents[4]),
            ("libreyolo", Path(libreyolo.__file__).resolve().parents[1]),
        ):
            diff = subprocess.check_output(
                ["git", "-C", str(root), "diff", "HEAD", "--binary"]
            )
            untracked = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
                text=True,
            ).splitlines()
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            fingerprint = hashlib.sha256(
                diff
                + commit.encode()
                + "".join(
                    filename + sha256_file(root / filename)
                    for filename in untracked
                    if (root / filename).is_file()
                ).encode()
            ).hexdigest()
            destination = sources / name / fingerprint
            destination.mkdir(parents=True, exist_ok=True)
            archive = destination / "source.tar"
            if not archive.exists():
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        "archive",
                        "--format=tar",
                        "HEAD",
                        "-o",
                        str(archive),
                    ],
                    check=True,
                )
                (destination / "changes.patch").write_bytes(diff)
                # Include untracked implementation files absent from git archive/diff.
                import tarfile

                with tarfile.open(destination / "untracked.tar", "w") as tar:
                    for filename in untracked:
                        tar.add(root / filename, arcname=filename)
                write_json_atomic(
                    destination / "manifest.json",
                    {
                        "commit": commit,
                        "archive_sha256": sha256_file(archive),
                        "patch_sha256": sha256_file(destination / "changes.patch"),
                        "untracked_sha256": sha256_file(destination / "untracked.tar"),
                    },
                )
            records.append(
                {
                    "repository": name,
                    "revision": fingerprint,
                    "directory": str(destination.relative_to(self.output)),
                }
            )
        self.source_revision = records
        mode = Path(__file__).resolve().parent.parent
        relevant = [
            *Path(__file__).resolve().parent.glob("*.py"),
            mode / "libreyolo/zero_shot_backend.py",
            mode / "libreyolo/adapter_slice_backend.py",
            mode / "libreyolo/adapter_backend.py",
        ]
        self.evaluation_signature = {
            str(p.relative_to(mode)): sha256_file(p) for p in relevant
        }
        self.evaluation_signature["libreyolo"] = records[1]["revision"]
