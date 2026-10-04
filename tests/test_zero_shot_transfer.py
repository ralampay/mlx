import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx.core.exceptions import MLXUserError
from mlx.core.paired_superiority import AnalyzePairedSuperiority, holm_adjust
from mlx.modes.object_detection.zero_shot.data import (
    metadata,
    stratified_ids,
    preserve_identity,
)
from mlx.modes.object_detection.zero_shot.scoring import ScoreTransferPredictions
from mlx.modes.object_detection.zero_shot.snapshots import copy_verified, bundled_run
from mlx.modes.object_detection.zero_shot.gallery import (
    select_highlights,
    GenerateTransferGallery,
)


def test_paired_statistics_matches_scipy():
    from scipy.stats import ttest_rel

    a, b = [1, 2, 4, 3, 6], [0, 1, 2, 2, 2]
    result = AnalyzePairedSuperiority(a, b).execute()
    expected = ttest_rel(a, b)
    assert result["t"] == pytest.approx(expected.statistic)
    assert result["p"] == pytest.approx(expected.pvalue)
    assert result["exact_sign_flip_p"] == 0.0625
    assert result["ci95_low"] < result["mean_difference"] < result["ci95_high"]
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(MLXUserError):
        AnalyzePairedSuperiority([1], [1]).execute()


def test_metadata_and_subset():
    assert (
        metadata("fog/val/scene/scene_frame_000123_rgb_anon.png", "acdc")["sequence"]
        == "fog/scene"
    )
    assert metadata("video5_frame_202.jpg", "mrtmd")["frame"] == 202
    images = [{"id": i, "sequence": str(i % 3)} for i in range(100)]
    assert stratified_ids(images, 12) == stratified_ids(list(reversed(images)), 12)
    assert len(set(stratified_ids(images, 128))) == 100


def test_snapshots_relocation_and_corruption(tmp_path):
    source = tmp_path / "original.pt"
    source.write_bytes(b"checkpoint")
    bundle = tmp_path / "bundle"
    digest = copy_verified(source, bundle / "lora/seed-1/adapter/checkpoint.pt")
    source.unlink()
    relocated = tmp_path / "relocated"
    bundle.rename(relocated)
    entry = {
        "id": "lora/seed-1",
        "method": "lora",
        "seed": 1,
        "config": {},
        "state": "lora/seed-1/adapter/checkpoint.pt",
        "sha256": digest,
    }
    assert bundled_run(relocated, entry).directory == relocated / "lora/seed-1"
    (relocated / entry["state"]).write_bytes(b"bad")
    with pytest.raises(MLXUserError):
        bundled_run(relocated, entry)
    preserve_identity(tmp_path / "identity.json", {"batch": 2})
    with pytest.raises(MLXUserError):
        preserve_identity(tmp_path / "identity.json", {"batch": 4})
    preserve_identity(tmp_path / "integer-keys.json", {0: "person"})
    preserve_identity(tmp_path / "integer-keys.json", {0: "person"})


def test_coco_crowds_native_ids_empty_images_and_slices(tmp_path):
    gt = {
        "images": [
            {"id": 0, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 37, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 0, "name": "person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 0,
                "bbox": [0, 0, 20, 20],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 0,
                "category_id": 0,
                "bbox": [40, 40, 50, 50],
                "area": 2500,
                "iscrowd": 1,
            },
        ],
    }
    annotation = tmp_path / "gt.json"
    annotation.write_text(json.dumps(gt))
    predictions = [
        {"image_id": 0, "category_id": 0, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 0, "category_id": 0, "bbox": [45, 45, 10, 10], "score": 0.8},
        {"image_id": 37, "category_id": 0, "bbox": [0, 0, 20, 20], "score": 0.7},
    ]
    pred = tmp_path / "predictions.json"
    pred.write_text(json.dumps(predictions))
    dataset = {
        "annotations": str(annotation),
        "images": [
            {"id": 0, "group": "a", "sequence": "s"},
            {"id": 37, "group": "b", "sequence": "t"},
        ],
    }
    result = ScoreTransferPredictions(dataset, pred, tmp_path).execute()
    assert result["all"]["tp"] == 1
    assert result["all"]["fp"] == 1
    assert result["all"]["ignored"] == 1
    assert result["all"]["fn"] == 0
    assert result["all"]["precision"] == 0.5
    assert result["group:a"]["mAP50_95"] == pytest.approx(1.0)
    assert result["group:b"]["mAP50_95"] is None
    pred.write_text("[]")
    empty = ScoreTransferPredictions(dataset, pred, tmp_path).execute()
    assert empty["all"]["fn"] == 1
    assert empty["all"]["mAP50_95"] == 0


def test_balanced_highlights_are_not_only_wins():
    images = [
        {"id": i, "group": "night", "sequence": str(i), "frame": i} for i in range(12)
    ]
    differences = {
        str(i): (0.3 if i < 4 else -0.2 if i < 8 else 0.01) for i in range(12)
    }
    selected = select_highlights(images, differences)
    assert {r["category"] for r in selected} == {"win", "loss", "near-tie"}


def test_cli_and_cuda_rejection(monkeypatch):
    from mlx.cli import build_parser
    from mlx.modes.object_detection.libreyolo.zero_shot_backend import (
        LibreYOLOTransferEvaluator,
    )
    import torch

    options = build_parser().parse_args(
        [
            "--action",
            "adapter-zero-shot",
            "--study-config",
            "x.json",
            "--device",
            "cuda",
        ]
    )
    assert options.study_config == "x.json"
    with pytest.raises(MLXUserError, match="requires explicit CUDA"):
        LibreYOLOTransferEvaluator("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MLXUserError, match="CUDA was requested"):
        LibreYOLOTransferEvaluator("cuda")


def test_complete_offline_gallery(tmp_path):
    from PIL import Image
    from mlx.core.artifacts import write_json_atomic

    image = tmp_path / "source.jpg"
    Image.new("RGB", (100, 100), "white").save(image)
    gt = tmp_path / "annotations.json"
    write_json_atomic(
        gt, {"categories": [{"id": 0, "name": "person"}], "annotations": []}
    )
    models = [
        {"id": "frozen", "method": "frozen", "seed": None},
        {"id": "lora/seed-1", "method": "lora", "seed": 1},
        {"id": "drax-hybrid/seed-1", "method": "drax-hybrid", "seed": 1},
    ]
    for model in models:
        directory = tmp_path / "runs/example" / model["id"]
        write_json_atomic(directory / "predictions.json", [])
        write_json_atomic(directory / "per-image.json", {"0": {"f1": None}})
    dataset = {
        "name": "example",
        "annotations": str(gt),
        "images": [
            {
                "id": 0,
                "file_name": "source.jpg",
                "path": str(image),
                "width": 100,
                "height": 100,
                "group": "night",
                "sequence": "s",
                "frame": 0,
            }
        ],
    }
    result = GenerateTransferGallery(tmp_path).execute([dataset], models)
    assert result["images"] == 1
    assert (tmp_path / "gallery/example/0-comparison.jpg").is_file()
    assert "drax-hybrid/seed-1" in (tmp_path / "gallery/example/0.html").read_text()
    assert json.loads((tmp_path / "gallery/images.json").read_text())[0]["id"] == 0
