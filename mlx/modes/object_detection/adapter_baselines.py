"""Read-only reuse of completed, protocol-compatible adapter studies."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from mlx.core.artifacts import sha256_file
from mlx.core.exceptions import MLXUserError


PROTOCOL_FIELDS = (
    "model", "checkpoint_sha256", "dataset_selection_sha256", "image_size",
    "physical_batch_size", "effective_batch_size", "gradient_accumulation",
    "amp", "device", "optimizer", "learning_rate", "train_images", "val_images", "test_images",
)


def read_artifact(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MLXUserError(f"Cannot read adapter artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MLXUserError(f"Adapter artifact must contain a JSON object: {path}")
    return value


def baseline_root(output: Path, explicit: Path | None = None) -> Path | None:
    metadata = output / "baseline.json"
    recorded = None
    if metadata.is_file():
        value = read_artifact(metadata).get("root")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise MLXUserError(f"Baseline metadata requires an absolute root: {metadata}")
        recorded = Path(value)
    chosen = Path(explicit).expanduser().resolve() if explicit else recorded
    if recorded and chosen != recorded:
        raise MLXUserError("--baseline-study differs from the recorded baseline")
    if chosen == output.resolve():
        raise MLXUserError("Baseline study must be different from the new output directory")
    return chosen


@dataclass(frozen=True)
class BaselineRun:
    method: str
    seed: int
    directory: Path
    metrics: Mapping
    config: Mapping


class LoadAdapterBaseline:
    """Verify a complete baseline matrix against explicit experiment conditions."""

    def __init__(self, root: Path, expected: Mapping, seeds: tuple[int, ...]):
        self.root = Path(root).expanduser().resolve()
        self.expected = expected
        self.seeds = seeds

    def execute(self) -> list[BaselineRun]:
        study = read_artifact(self.root / "study.json")
        methods = study.get("available_methods") or ()
        if not methods or not self.seeds:
            raise MLXUserError("Baseline study must declare a nonempty method/seed matrix")
        runs = []
        for method in methods:
            for seed in self.seeds:
                directory = self.root / method / f"seed-{seed}"
                config = read_artifact(directory / "config.json")
                metrics = read_artifact(directory / "metrics.json")
                if metrics.get("status") != "completed":
                    raise MLXUserError(f"Baseline run is incomplete: {directory}")
                for record in (config, metrics):
                    if record.get("method") != method or record.get("seed") != seed:
                        raise MLXUserError(f"Baseline run identity mismatch: {directory}")
                    for field in PROTOCOL_FIELDS:
                        if field not in self.expected or record.get(field) != self.expected[field]:
                            raise MLXUserError(f"Baseline protocol mismatch for {field}: {directory}")
                    expected_epochs = 0 if method == "frozen" else self.expected["epochs"]
                    if record.get("epochs") != expected_epochs:
                        raise MLXUserError(f"Baseline epoch mismatch: {directory}")
                runs.append(BaselineRun(method, seed, directory, metrics, config))
        return runs


def baseline_provenance(root: Path, runs: list[BaselineRun], expected: Mapping,
                        seeds: tuple[int, ...]) -> dict:
    return {
        "root": str(root),
        "study_sha256": sha256_file(root / "study.json"),
        "environment_sha256": sha256_file(root / "environment.json"),
        "protocol": dict(expected),
        "seeds": list(seeds),
        "run_artifacts": {
            str(path.relative_to(root)): sha256_file(path)
            for run in runs
            for path in (run.directory / "metrics.json", run.directory / "config.json")
        },
    }


def recorded_baseline_runs(output: Path, explicit: Path | None = None) -> list[BaselineRun]:
    root = baseline_root(output, explicit)
    if root is None:
        return []
    metadata = read_artifact(output / "baseline.json")
    if not isinstance(metadata.get("protocol"), dict) or not isinstance(metadata.get("seeds"), list):
        raise MLXUserError("Baseline metadata requires protocol and seeds")
    loader = LoadAdapterBaseline(root, metadata["protocol"], tuple(metadata["seeds"]))
    runs = loader.execute()
    if baseline_provenance(root, runs, metadata["protocol"], tuple(metadata["seeds"])) != metadata:
        raise MLXUserError("Baseline study artifacts changed since they were recorded")
    for path in output.glob("*/seed-*/config.json"):
        config = read_artifact(path)
        for field in (*PROTOCOL_FIELDS, "epochs"):
            expected = 0 if field == "epochs" and config.get("method") == "frozen" else metadata["protocol"][field]
            if config.get(field) != expected:
                raise MLXUserError(f"New study protocol mismatch for {field}: {path}")
    return runs
