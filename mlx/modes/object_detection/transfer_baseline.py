"""Read-only validation and receipts for completed taxonomy-transfer baselines."""

import json
from pathlib import Path

from mlx.core.artifacts import sha256_file
from mlx.core.exceptions import MLXUserError


class VerifyTransferBaseline:
    def __init__(self, root, configuration):
        self.root, self.configuration = Path(root), configuration

    def execute(self):
        root, config = self.root, self.configuration
        plan = json.loads((root / "plan.json").read_text())
        status = json.loads((root / "status.json").read_text())
        if status.get("status") != "completed" or status.get("completed_runs") != 30:
            raise MLXUserError("Ablation baseline must contain all 30 completed NEU-DET runs")
        for key in ("checkpoint_sha256", "dataset_selection_sha256", "seeds", "epochs",
                    "image_size", "lr", "device", "amp", "head_policy", "train_head", "target"):
            if plan.get(key) != config.get(key):
                raise MLXUserError(f"Baseline protocol mismatch: {key}")
        for name, digest in plan["dataset_files"].items():
            if config["dataset_files"].get(name) != digest:
                raise MLXUserError(f"Baseline dataset differs: {name}")
        files = {}
        for name in ("plan.json", "status.json", "environment.json", "dataset.json"):
            files[name] = sha256_file(root / name)
        for method in plan["methods"]:
            for seed in plan["seeds"]:
                directory = root / method / f"seed-{seed}"
                metrics = json.loads((directory / "metrics.json").read_text())
                expected = {"status":"completed", "physical_batch_size":8, "effective_batch_size":8,
                            "epochs":50, "head_policy":"reset-classifiers", "train_head":True,
                            "checkpoint_sha256":config["checkpoint_sha256"],
                            "dataset_selection_sha256":config["dataset_selection_sha256"]}
                if any(metrics.get(k) != v for k,v in expected.items()):
                    raise MLXUserError(f"Incompatible baseline condition: {directory}")
                checkpoint = directory / "adapter" / "checkpoint.pt"
                if not checkpoint.exists():
                    checkpoint = Path(metrics["export_checkpoint_path"])
                for path in [checkpoint, *(directory / name for name in (
                    "metrics.json", "config.json", "scores.json", "predictions.json", "inference.json"))]:
                    files[str(path.relative_to(root))] = sha256_file(path)
        return {"root":str(root.resolve()), "runs":30, "files":files,
                "policy":"read-only completed baseline; new results written only to the ablation study"}
