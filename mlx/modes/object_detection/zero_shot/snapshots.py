"""Hash-verified self-contained checkpoint bundles."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import shutil

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from .data import read_json, preserve_identity


def copy_verified(source, destination):
    source, destination = Path(source).expanduser(), Path(destination)
    digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise MLXUserError(f"Snapshot hash mismatch: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != digest:
            raise MLXUserError(f"Copy verification failed: {source}")
        temporary.replace(destination)
    return digest


class SnapshotTransferModels:
    def __init__(self, specification, output, foundation_info):
        self.spec, self.output, self.info = specification, Path(output), foundation_info

    def execute(self):
        root = self.output / "models"
        copy_verified(self.spec["foundation"], root / "foundation.pt")
        entries = [
            {
                "id": "frozen",
                "method": "frozen",
                "seed": None,
                "config": {},
                "state": "foundation.pt",
                "sha256": self.info["sha256"],
                "trainable_params": 0,
            }
        ]
        for source in self.spec["studies"]:
            for method in source["methods"]:
                for seed in self.spec["seeds"]:
                    directory = (
                        Path(source["path"]).expanduser() / method / f"seed-{seed}"
                    )
                    config = read_json(directory / "config.json")
                    metrics = read_json(directory / "metrics.json")
                    if (
                        metrics.get("status") != "completed"
                        or config["checkpoint_sha256"] != self.info["sha256"]
                    ):
                        raise MLXUserError(
                            f"Incomplete run or different foundation: {directory}"
                        )
                    if config["method"] != method or config["seed"] != seed:
                        raise MLXUserError(f"Source run identity mismatch: {directory}")
                    if (
                        method == "drax-hybrid"
                        and len(config.get("injected_modules", [])) != 26
                    ):
                        raise MLXUserError(
                            "This study requires the revised 26-convolution hybrid"
                        )
                    identifier = f"{method}/seed-{seed}"
                    dense = method in {"head-only", "full-finetune"}
                    original = (
                        Path(metrics["selected_checkpoint_path"])
                        if dense
                        else directory / "adapter/checkpoint.pt"
                    )
                    relative = (
                        f"{identifier}/selected.pt"
                        if dense
                        else f"{identifier}/adapter/checkpoint.pt"
                    )
                    digest = copy_verified(original, root / relative)
                    copy_verified(
                        directory / "config.json",
                        root / identifier / "source-config.json",
                    )
                    copy_verified(
                        directory / "metrics.json",
                        root / identifier / "source-metrics.json",
                    )
                    # Source paths are provenance only. Reconstruction uses relative bundle paths.
                    portable = {
                        k: v
                        for k, v in config.items()
                        if k
                        not in {
                            "foundation_checkpoint",
                            "selected_checkpoint_path",
                            "dataset_path",
                        }
                    }
                    entries.append(
                        {
                            "id": identifier,
                            "method": method,
                            "seed": seed,
                            "config": portable,
                            "state": relative,
                            "sha256": digest,
                            "source": str(original),
                            "trainable_params": config["trainable_params"],
                            "state_size_mib": original.stat().st_size / 2**20,
                        }
                    )
        if len({e["id"] for e in entries}) != len(entries):
            raise MLXUserError("Duplicate model instances in study")
        preserve_identity(
            root / "manifest.json",
            {
                "schema": 1,
                "foundation": "foundation.pt",
                "foundation_sha256": self.info["sha256"],
                "models": entries,
            },
        )
        return entries


def bundled_run(root, entry):
    root = Path(root)
    state = (root / entry["state"]).resolve()
    if (
        not state.is_relative_to(root.resolve())
        or sha256_file(state) != entry["sha256"]
    ):
        raise MLXUserError(f"Corrupt or unsafe model snapshot: {state}")
    config = dict(entry["config"])
    if entry["method"] in {"head-only", "full-finetune"}:
        config["selected_checkpoint_path"] = str(state)
    return SimpleNamespace(
        method=entry["method"],
        seed=entry["seed"],
        config=config,
        directory=root / entry["id"],
    )
