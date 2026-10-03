"""Experiment identity and resumable, content-verified stage publication."""
from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mlx.core.artifacts import json_safe, sha256_file, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, emit
from mlx.core.exceptions import MLXUserError


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        raise MLXUserError(f"Unable to read experiment artifact {path}: {exc}") from exc


def file_hashes(root):
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "stage.json"}


def experiment_identity(request, model_hash, datasets):
    config = asdict(request)
    for name in ("extras", "resume", "download_datasets", "output_path", "dry_run"):
        config.pop(name, None)
    for name in ("experiment_config", "embedding_source"):
        if config.get(name) is None:
            config.pop(name, None)
    packages = {}
    for name in ("numpy", "torch", "scipy", "scikit-learn", "llama-cpp-python"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Invalidate on relevant local source edits, including uncommitted changes.
    package = Path(__file__).resolve().parents[2]
    sources = {}
    for directory in (package / "core", package / "modes/text_embedding", package / "modes/autoencoder"):
        sources.update({str(p.relative_to(package)): sha256_file(p) for p in sorted(directory.rglob("*.py"))})
    return {"schema_version": 1, "config": config, "model_sha256": model_hash,
            "datasets": datasets, "packages": packages, "python": platform.python_version(), "sources": sources}


class ExperimentStages:
    def __init__(self, root, identity, *, resume=False, reporter=None):
        identity = json_safe(identity)
        self.root = Path(root)
        self.resume = resume
        self.reporter = reporter or NullWorkflowReporter()
        path = self.root / "experiment.json"
        if path.exists():
            if not resume or read_json(path) != identity:
                raise MLXUserError("Experiment output already exists or its inputs/configuration changed. Use a new --output directory.")
        elif self.root.exists() and any(self.root.iterdir()):
            raise MLXUserError("Experiment output is nonempty and has no experiment manifest.")
        else:
            write_json_atomic(path, identity)

    def run(self, relative, operation):
        target = self.root / relative
        marker = target / "stage.json"
        if target.exists():
            if not self.resume or not marker.is_file():
                raise MLXUserError(f"Unrecognized existing experiment stage: {target}")
            state = read_json(marker)
            if state.get("hashes") != file_hashes(target):
                raise MLXUserError(f"Completed stage contents changed: {target}. Use a new output directory.")
            emit(self.reporter, "info", f"Reusing {relative}", payload={"event": "retrieval_stage"})
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.incomplete")
        if temporary.exists():
            shutil.rmtree(temporary)
        emit(self.reporter, "info", f"Running {relative}", payload={"event": "retrieval_stage"})
        start = time.monotonic()
        try:
            operation(temporary)
            write_json_atomic(temporary / "stage.json", {
                "seconds": time.monotonic() - start, "hashes": file_hashes(temporary),
            })
            temporary.rename(target)
        except Exception as exc:
            write_json_atomic(temporary / "failure.json", {"stage": relative, "error": str(exc)})
            raise MLXUserError(f"Experiment stage {relative} failed: {exc}. Fix the cause and rerun with --resume.") from exc
        return target
