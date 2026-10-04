import json
from pathlib import Path

import pytest

from mlx.cli import build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_baselines import (
    LoadAdapterBaseline, PROTOCOL_FIELDS, recorded_baseline_runs, baseline_provenance,
)
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest
from mlx.modes.object_detection.adapter_report import GenerateAdapterReport
from mlx.modes.object_detection.adapter_slices import AdapterSliceRequest, discover_study_runs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "old"
    config = {name: 1 for name in PROTOCOL_FIELDS}
    config.update(epochs=20, model="yolox-l", device="cuda:0", amp=True)
    write(root / "study.json", {"available_methods": ["frozen", "lora"], "planned_seeds": [1, 2]})
    write(root / "environment.json", {"gpu_name": "test GPU"})
    for method in ("frozen", "lora"):
        for seed in (1, 2):
            values = {
                **config, "method": method, "seed": seed,
                "epochs": 0 if method == "frozen" else 20,
                "status": "completed", "mAP50": .4, "mAP50_95": .3,
                "precision": .5, "recall": .6, "trainable_percent": 1,
                "peak_cuda_memory_mb": 100, "training_seconds": 10,
            }
            for filename in ("config.json", "metrics.json"):
                write(root / method / f"seed-{seed}" / filename, values)
    return root, config


def test_reuses_complete_matrix_without_writing_baseline(baseline, tmp_path):
    root, config = baseline
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    loader = LoadAdapterBaseline(root, config, (1, 2))
    runs = loader.execute()
    assert len(runs) == 4
    output = tmp_path / "new"
    write(output / "baseline.json", baseline_provenance(root, runs, config, (1, 2)))
    assert len(recorded_baseline_runs(output)) == 4
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}
    config_path = root / "lora/seed-1/config.json"
    payload = json.loads(config_path.read_text())
    payload["extra"] = "changed"
    write(config_path, payload)
    with pytest.raises(MLXUserError, match="changed"):
        recorded_baseline_runs(output)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "dataset_selection_sha256", "amp", "epochs", "effective_batch_size"])
def test_incompatible_baseline_rejected(baseline, field):
    root, config = baseline
    config[field] = "wrong"
    with pytest.raises(MLXUserError, match="mismatch"):
        LoadAdapterBaseline(root, config, (1, 2)).execute()


def test_missing_seed_rejected(baseline):
    root, config = baseline
    with pytest.raises(MLXUserError, match="Cannot read adapter artifact"):
        LoadAdapterBaseline(root, config, (3,)).execute()


def test_external_discovery_and_hybrid_report(baseline, tmp_path):
    root, config = baseline
    loader = LoadAdapterBaseline(root, config, (1, 2))
    output = tmp_path / "new"
    write(output / "baseline.json", baseline_provenance(root, loader.execute(), config, (1, 2)))
    write(output / "study.json", {"available_methods": ["frozen", "drax-hybrid"], "planned_seeds": [1, 2]})
    for seed in (1, 2):
        values = json.loads((root / f"lora/seed-{seed}/metrics.json").read_text())
        values.update(method="drax-hybrid", mAP50_95=.32)
        for name in ("config.json", "metrics.json"):
            write(output / f"drax-hybrid/seed-{seed}" / name, values)
    request = AdapterSliceRequest(output, tmp_path / "foundation", tmp_path / "data", comparison_method="drax-hybrid")
    runs = discover_study_runs(request)
    assert len(runs) == 6
    assert next(r for r in runs if r.method == "frozen").directory.parent.parent == root
    result = GenerateAdapterReport(output, comparison_method="drax-hybrid").execute()
    payload = json.loads(Path(result["json"]).read_text())
    assert len(payload["runs"]) == 6
    assert payload["paired_differences"]["drax-hybrid_minus_lora"]["mean"] == pytest.approx(.02)


def test_cli_accepts_hybrid_and_baseline():
    args = build_parser().parse_args([
        "--mode", "object-detection", "--action", "adapter-experiment",
        "--adapter", "drax-hybrid", "--baseline-study", "/baseline",
        "--comparison-method", "drax-hybrid", "--device", "cuda",
    ])
    request = AdapterExperimentRequest.from_config(vars(args))
    assert request.methods == ("drax-hybrid",)
    assert request.baseline_study == Path("/baseline")
