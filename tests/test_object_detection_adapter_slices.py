from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from mlx.cli import build_parser
from mlx.core.artifacts import sha256_file
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_slices import (
    AdapterSliceRequest,
    GenerateAdapterSliceReport,
    baseline_image_quality,
    build_coco_ground_truth,
    difficulty_label,
    difficulty_tertiles,
    discover_study_runs,
    weather_category,
    weather_group,
)
from mlx.modes.object_detection.libreyolo.adapter_slice_backend import (
    LibreYOLOAdapterPredictionWriter,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _request(tmp_path: Path, **changes) -> AdapterSliceRequest:
    checkpoint = tmp_path / "foundation.pt"
    checkpoint.touch()
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    values = {
        "output": tmp_path,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "device": "cuda",
        "methods": ("frozen", "drax", "lora"),
        "seeds": (1, 2),
        "bootstrap_samples": 25,
        "analysis_seed": 42,
    }
    values.update(changes)
    return AdapterSliceRequest(**values)


@pytest.mark.parametrize(
    ("filename", "expected"),
    (
        ("dusttornado-004.jpg", "dust-tornado"),
        ("foggy-003.jpg", "fog"),
        ("haze-090.jpg", "haze"),
        ("mist-149.jpg", "mist"),
        ("rain_storm-006.jpg", "rainstorm"),
        ("sand_storm-100.jpg", "sandstorm"),
        ("sand_storm_g2_001.jpg", "sandstorm"),
        ("snow_storm-014.jpg", "snowstorm"),
    ),
)
def test_weather_category_and_groups(filename, expected):
    assert weather_category(filename) == expected
    assert weather_group(expected) in {
        "low-visibility", "precipitation", "airborne-particulate"
    }


def test_unknown_weather_is_rejected():
    with pytest.raises(MLXUserError, match="Cannot derive"):
        weather_category("sunny-001.jpg")


def test_ground_truth_conversion_preserves_original_pixel_area(tmp_path):
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(image_dir / "foggy-001.jpg")
    (label_dir / "foggy-001.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    result = build_coco_ground_truth(tmp_path, "test", ("person",))
    assert result["images"] == [
        {"id": 0, "file_name": "foggy-001.jpg", "width": 200, "height": 100}
    ]
    assert result["annotations"][0]["bbox"] == pytest.approx([90, 40, 20, 20])
    assert result["annotations"][0]["area"] == pytest.approx(400)


def test_baseline_quality_and_difficulty_are_deterministic():
    ground_truth = {
        "images": [{"id": value} for value in range(3)],
        "annotations": [
            {"image_id": value, "category_id": 0, "bbox": [0, 0, 10, 10]}
            for value in range(3)
        ],
    }
    predictions = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 5, 10], "score": 0.8},
    ]
    quality = baseline_image_quality(ground_truth, predictions)
    assert quality == pytest.approx({0: 0.9, 1: 0.4, 2: 0.0})
    lower, upper = difficulty_tertiles(quality.values())
    assert [difficulty_label(value, lower, upper) for value in quality.values()] == [
        "easy", "medium", "hard"
    ]


def test_study_discovery_rejects_incomplete_matrix(tmp_path):
    _write_json(tmp_path / "study.json", {"available_methods": ["drax"], "planned_seeds": [1, 2]})
    run = tmp_path / "drax" / "seed-1"
    _write_json(run / "metrics.json", {"status": "completed"})
    _write_json(run / "config.json", {})
    request = _request(tmp_path, methods=("drax",), seeds=(1, 2))
    with pytest.raises(MLXUserError, match="complete declared study"):
        discover_study_runs(request)


def test_slice_prediction_requires_cuda(tmp_path):
    request = _request(tmp_path, device="cpu")
    with pytest.raises(MLXUserError, match="requires --device cuda"):
        from mlx.modes.object_detection.libreyolo.adapter_slice_backend import LibreYOLOSliceBackend
        LibreYOLOSliceBackend().prepare(request, "yolox-l")


def test_cli_parses_slice_analysis_options():
    parsed = build_parser().parse_args(
        [
            "--mode", "object-detection", "--action", "adapter-slice-report",
            "--bootstrap-samples", "123", "--seed", "7",
        ]
    )
    assert parsed.bootstrap_samples == 123
    assert parsed.random_seed == 7


def test_slice_request_ignores_unrelated_global_parser_defaults(tmp_path):
    request = AdapterSliceRequest.from_config(
        {
            "output_path": str(tmp_path),
            "height": 256,
            "batch_size": 1,
            "workers": 4,
            "device": "cpu",
            "dataset_path": "./tmp/dataset",
            "action": "adapter-slice-report",
            "_explicit_options": {"output_path"},
        }
    )
    assert request.image_size == 640
    assert request.batch_size == 8
    assert request.workers == 0
    assert request.device == "cuda"


@pytest.mark.parametrize("method", ["drax-hybrid", "drax-residual-fusion"])
def test_hybrid_reconstruction_uses_recorded_convolutions(method):
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 1), torch.nn.Conv2d(8, 8, 1))
    targets = LibreYOLOAdapterPredictionWriter._adapter_targets(
        model, {"injected_modules": ["1"]}, method
    )
    assert targets == {"1": 8}


@pytest.mark.parametrize("paths", [[], ["missing"], ["0", "0"], "0", [None]])
def test_invalid_recorded_hybrid_targets_fail_clearly(paths):
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 1))
    with pytest.raises(MLXUserError, match="[Hh]ybrid"):
        LibreYOLOAdapterPredictionWriter._adapter_targets(model, {"injected_modules": paths}, "drax-hybrid")


def test_head_only_reconstruction_preserves_foundation_and_restores_head_buffers(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(2, 2)
            self.head = torch.nn.BatchNorm1d(2)

    model = TinyModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    selected = {
        name: torch.full_like(value, 3 if value.dtype.is_floating_point else 4)
        for name, value in model.state_dict().items()
    }
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    run = SimpleNamespace(
        method="head-only",
        config={"selected_checkpoint_path": str(checkpoint)},
    )
    LibreYOLOAdapterPredictionWriter._restore_dense_checkpoint(
        model, run, lambda *_args, **_kwargs: {"model": selected}
    )
    restored = model.state_dict()
    assert all(
        torch.equal(restored[name], selected[name])
        for name in restored if name.startswith("head.")
    )
    assert all(
        torch.equal(restored[name], before[name])
        for name in restored if not name.startswith("head.")
    )


@pytest.mark.parametrize("comparator", ["lora", "drax-spatial"])
def test_cached_prediction_report_end_to_end(tmp_path, comparator):
    request = _request(tmp_path, methods=("frozen", "drax", comparator))
    _write_json(
        tmp_path / "study.json",
        {"available_methods": ["frozen", "drax", comparator], "planned_seeds": [1, 2]},
    )
    images = [
        {"id": 0, "file_name": "foggy-001.jpg", "width": 100, "height": 100},
        {"id": 1, "file_name": "rain_storm-001.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "sand_storm-001.jpg", "width": 100, "height": 100},
    ]
    ground_truth = {
        "images": images,
        "annotations": [
            {
                "id": index + 1, "image_id": index, "category_id": 0,
                "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0,
            }
            for index in range(3)
        ],
        "categories": [{"id": 0, "name": "person", "supercategory": "object"}],
        "info": {},
        "licenses": [],
    }
    root = tmp_path / "sliced-analysis"
    _write_json(root / "ground_truth_test.json", ground_truth)
    _write_json(
        root / "analysis_manifest.json",
        {
            "difficulty": {"hard_max": 0.3, "medium_max": 0.7},
            "bootstrap_samples": 25,
        },
    )
    for method in ("frozen", "drax", comparator):
        for seed in (1, 2):
            run = tmp_path / method / f"seed-{seed}"
            _write_json(run / "metrics.json", {"status": "completed"})
            _write_json(run / "config.json", {})
            score = {"drax": 0.9, comparator: 0.8, "frozen": 0.7}[method]
            predictions = [
                {
                    "image_id": image["id"], "category_id": 0,
                    "bbox": [10, 10, 20, 20], "score": score,
                }
                for image in images
            ]
            path = root / "predictions" / method / f"seed-{seed}" / "test.json"
            _write_json(path, predictions)
            _write_json(
                path.with_name("test.metadata.json"),
                {"predictions_sha256": sha256_file(path)},
            )
    result = GenerateAdapterSliceReport(request).execute()
    assert result["runs"] == 6
    assert (root / "slice_results.csv").is_file()
    assert (root / "paired_differences.csv").is_file()
    assert f"drax-minus-{comparator}" in (root / "summary.md").read_text()
    rows = list(csv.DictReader((root / "slice_results.csv").open()))
    overall = [row for row in rows if row["slice_family"] == "overall"]
    assert len(overall) == 6
    assert all(float(row["mAP50_95"]) == pytest.approx(1.0) for row in overall)


def test_paired_report_rejects_missing_comparator_seeds():
    rows = [dict(method=method, seed=seed, slice_family="overall", slice_name="all")
            for method, seed in (("drax", 1), ("drax", 2), ("custom", 1))]
    with pytest.raises(MLXUserError, match="Incomplete paired seed coverage"):
        GenerateAdapterSliceReport._paired_results(rows, [], {}, {}, 10, 42)


def test_injected_cpu_backend_caches_and_reuses_predictions(tmp_path):
    from mlx.modes.object_detection.adapter_slices import CacheAdapterSlicePredictions
    request = _request(tmp_path, device="cpu", methods=("frozen",), seeds=(1,))
    dataset = request.dataset
    _write_json(dataset / "manifest.json", {"classes": ["widget"], "selection_sha256": "selection"})
    _write_json(dataset / "data.yaml", {"names": ["widget"]})
    for split in ("train", "val", "test"):
        images, labels = dataset / "images" / split, dataset / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (32, 32)).save(images / f"foggy-{index:03d}.jpg")
            (labels / f"foggy-{index:03d}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    _write_json(tmp_path / "study.json", {"dataset_selection_sha256": "selection"})
    run = tmp_path / "frozen" / "seed-1"
    _write_json(run / "metrics.json", dict(status="completed", checkpoint_sha256="foundation",
        dataset_selection_sha256="selection", mAP50=1, mAP50_95=1))
    _write_json(run / "config.json", {})
    calls = []
    class Backend:
        def prepare(self, request, model_name):
            assert request.device == "cpu"
            return dict(classes={0: "widget"}, nc=1, sha256="foundation"), "cpu"
        def create_writer(self, request, foundation, device, *, registry=None):
            def write(run, split, destination):
                calls.append(split)
                _write_json(destination, [dict(image_id=index, category_id=0, bbox=[8, 8, 16, 16], score=1) for index in range(3)])
                return dict(map_50=1, map_50_95=1)
            return write
    command = CacheAdapterSlicePredictions(request, backend=Backend())
    command.execute()
    command.execute()
    assert calls == ["val", "test"]
