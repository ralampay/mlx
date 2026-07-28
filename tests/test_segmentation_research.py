from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation import evaluation as segmentation_evaluation
from mlx.modes.segmentation import train as segmentation_train
from mlx.modes.segmentation.data import resolve_segmentation_evaluation_split
from mlx.modes.segmentation.metrics import (
    aggregate_confusion_metrics,
    boundary_metrics,
    class_metrics_from_confusion,
    confusion_matrix_from_arrays,
    per_image_metrics,
    probability_metrics,
    threshold_sweep,
)
from mlx.modes.segmentation.research import (
    write_confusion_matrix_artifacts,
    write_curve_artifacts,
    write_json,
    write_metrics_csv,
    write_threshold_artifacts,
)
from mlx.modes.segmentation.utils import (
    resolve_class_names,
    resolve_train_output_paths,
)


def test_binary_confusion_and_aggregate_metrics() -> None:
    targets = np.asarray([[0, 0, 1, 1], [0, 1, 1, 0]])
    predictions = np.asarray([[0, 1, 1, 1], [0, 0, 1, 0]])

    matrix = confusion_matrix_from_arrays(targets, predictions, 2)
    rows = class_metrics_from_confusion(matrix, ["background", "pedestrian"])
    metrics = aggregate_confusion_metrics(matrix, rows)

    assert matrix.tolist() == [[3, 1], [1, 3]]
    assert metrics["pixel_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_dice"] == pytest.approx(0.75)
    assert metrics["mean_foreground_dice"] == pytest.approx(0.75)
    assert metrics["mean_iou"] == pytest.approx(0.6)
    assert metrics["cohen_kappa"] == pytest.approx(0.5)
    assert rows[1]["precision"] == pytest.approx(0.75)
    assert rows[1]["specificity"] == pytest.approx(0.75)


def test_absent_class_is_nan_and_excluded_from_macro() -> None:
    targets = np.zeros((2, 2), dtype=np.int64)
    predictions = np.zeros((2, 2), dtype=np.int64)
    matrix = confusion_matrix_from_arrays(targets, predictions, 2)
    rows = class_metrics_from_confusion(matrix, ["background", "foreground"])
    metrics = aggregate_confusion_metrics(matrix, rows)

    assert rows[0]["dice"] == pytest.approx(1.0)
    assert np.isnan(float(rows[1]["dice"]))
    assert metrics["macro_dice"] == pytest.approx(1.0)
    assert np.isnan(metrics["mean_foreground_dice"])
    assert metrics["valid_foreground_class_count"] == 0


def test_boundary_empty_mask_conventions() -> None:
    empty = np.zeros((8, 8), dtype=np.uint8)
    foreground = empty.copy()
    foreground[2:6, 2:6] = 1

    both_empty = boundary_metrics(empty, empty, tolerance=2)
    one_empty = boundary_metrics(empty, foreground, tolerance=2)
    perfect = boundary_metrics(foreground, foreground, tolerance=0)

    assert both_empty["boundary_f1"] == 1
    assert both_empty["hausdorff_95"] == 0
    assert one_empty["boundary_f1"] == 0
    assert np.isnan(one_empty["hausdorff_95"])
    assert perfect["boundary_f1"] == pytest.approx(1)
    assert perfect["hausdorff_distance"] == pytest.approx(0)


def test_probability_calibration_and_curves() -> None:
    targets = np.asarray([[[0, 1], [0, 1]]])
    probabilities = np.asarray(
        [[[[0.9, 0.1], [0.2, 0.8]], [[0.7, 0.3], [0.1, 0.9]]]],
        dtype=np.float64,
    )
    predictions = probabilities.argmax(axis=-1)
    matrix = confusion_matrix_from_arrays(targets, predictions, 2)
    rows = class_metrics_from_confusion(matrix, ["background", "foreground"])

    metrics, curves = probability_metrics(
        targets,
        probabilities,
        rows,
        calibration_bins=5,
    )

    assert metrics["negative_log_likelihood"] > 0
    assert metrics["macro_roc_auc"] == pytest.approx(1)
    assert metrics["macro_average_precision"] == pytest.approx(1)
    assert metrics["expected_calibration_error"] > 0
    assert curves["roc"]
    assert curves["precision_recall"]
    assert curves["calibration"]


def test_threshold_sweep_reports_optima() -> None:
    targets = np.asarray([[0, 0, 1, 1]])
    scores = np.asarray([[0.1, 0.4, 0.6, 0.9]])
    rows, summary = threshold_sweep(
        targets,
        scores,
        threshold_steps=5,
        configured_threshold=0.6,
    )

    assert any(row["threshold"] == pytest.approx(0.6) for row in rows)
    assert summary["best_dice_at_threshold"] == pytest.approx(1)
    assert summary["best_iou_at_threshold"] == pytest.approx(1)


def test_per_image_metrics_include_boundary_and_class_values() -> None:
    target = np.zeros((12, 12), dtype=np.int64)
    target[3:9, 3:9] = 1
    result = per_image_metrics(
        target,
        target.copy(),
        class_names=["background", "pedestrian"],
        boundary_tolerance=2,
    )

    assert result["mean_foreground_dice"] == pytest.approx(1)
    assert result["pedestrian_iou"] == pytest.approx(1)
    assert result["pedestrian_boundary_f1"] == pytest.approx(1)
    assert result["pedestrian_hausdorff_95"] == pytest.approx(0)


def test_class_names_validation_and_defaults() -> None:
    assert resolve_class_names({}, 2) == ["background", "foreground"]
    assert resolve_class_names({}, 3) == ["class_0", "class_1", "class_2"]
    assert resolve_class_names({"class_names": "background, pedestrian"}, 2) == [
        "background",
        "pedestrian",
    ]
    with pytest.raises(MLXUserError, match="Expected 2 class names"):
        resolve_class_names({"class_names": "only-one"}, 2)
    with pytest.raises(MLXUserError, match="unique"):
        resolve_class_names({"class_names": "same,same"}, 2)


def test_dual_training_output_contract(tmp_path: Path) -> None:
    directory_paths = resolve_train_output_paths(
        {"output_path": str(tmp_path / "run")},
        model_name="unet",
    )
    assert directory_paths["checkpoint_path"] == tmp_path / "run" / "unet.pth"
    assert directory_paths["dice_checkpoint_path"].name == "unet.best-dice.pth"
    assert directory_paths["last_checkpoint_path"].name == "unet.last.pth"
    assert directory_paths["training_csv_path"] == tmp_path / "run" / "training.csv"

    file_paths = resolve_train_output_paths(
        {"output_path": str(tmp_path / "model.pt")},
        model_name="unet",
    )
    assert file_paths["checkpoint_path"] == tmp_path / "model.pt"
    assert file_paths["dice_checkpoint_path"].name == "model.best-dice.pt"
    assert file_paths["last_checkpoint_path"].name == "model.last.pt"
    assert file_paths["training_csv_path"] == tmp_path / "model-research" / "training.csv"


def test_evaluation_split_resolution(tmp_path: Path) -> None:
    split = tmp_path / "test"
    (split / "images").mkdir(parents=True)
    (split / "masks").mkdir()
    assert resolve_segmentation_evaluation_split(tmp_path, split="test") == split
    assert resolve_segmentation_evaluation_split(split, split="val") == split
    with pytest.raises(MLXUserError, match="split 'val'"):
        resolve_segmentation_evaluation_split(tmp_path, split="val")


def test_research_artifacts_are_written(tmp_path: Path) -> None:
    matrix = np.asarray([[10, 2], [1, 7]])
    curves = {
        "roc": [
            {
                "class_index": 1.0,
                "false_positive_rate": 0.0,
                "true_positive_rate": 0.0,
                "threshold": float("inf"),
            },
            {
                "class_index": 1.0,
                "false_positive_rate": 1.0,
                "true_positive_rate": 1.0,
                "threshold": 0.0,
            },
        ],
        "precision_recall": [
            {"class_index": 1.0, "recall": 1.0, "precision": 0.5, "threshold": 0.0}
        ],
        "calibration": [
            {
                "bin_lower": 0.5,
                "bin_upper": 1.0,
                "count": 20.0,
                "confidence": 0.8,
                "accuracy": 0.85,
            }
        ],
    }
    threshold_rows = [
        {
            "threshold": 0.5,
            "dice": 0.8,
            "iou": 0.7,
            "precision": 0.8,
            "recall": 0.8,
            "specificity": 0.9,
        }
    ]

    write_metrics_csv(tmp_path / "metrics.csv", {"dice": 0.8, "undefined": float("nan")})
    write_json(tmp_path / "metrics.json", {"dice": 0.8, "undefined": float("nan")})
    write_confusion_matrix_artifacts(tmp_path, matrix, ["background", "foreground"])
    write_curve_artifacts(tmp_path, curves, ["background", "foreground"])
    write_threshold_artifacts(tmp_path, threshold_rows)

    expected = {
        "metrics.csv",
        "metrics.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "roc_curves.csv",
        "roc_curves.png",
        "precision_recall_curves.csv",
        "precision_recall_curves.png",
        "calibration_curve.csv",
        "calibration_curve.png",
        "threshold_metrics.csv",
        "threshold_curves.png",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "metrics.json").read_text())["undefined"] is None
    assert cv2.imread(str(tmp_path / "confusion_matrix.png")) is not None


def test_training_writes_research_outputs_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = torch.rand(4, 3, 8, 8)
    masks = torch.zeros(4, 8, 8, dtype=torch.long)
    masks[:, 2:6, 2:6] = 1
    dataset = TensorDataset(images, masks)

    def build_model(*args, **kwargs):
        return nn.Conv2d(3, 2, kernel_size=1)

    monkeypatch.setattr(
        segmentation_train,
        "load_segmentation_datasets",
        lambda *args, **kwargs: (dataset, dataset),
    )
    monkeypatch.setattr(segmentation_train, "build_segmentation_model", build_model)
    output_dir = tmp_path / "training"
    config = {
        "batch_size": 2,
        "class_names": "background,foreground",
        "colored": True,
        "dataset_path": str(tmp_path / "dataset"),
        "device": "cpu",
        "epochs": 1,
        "input_size": (8, 8),
        "lr": 1e-3,
        "model": "unet",
        "num_classes": 2,
        "output_path": str(output_dir),
        "refresh_per_second": 20,
    }

    segmentation_train.TrainSegmentationModel(config).execute()

    assert (output_dir / "unet.pth").is_file()
    assert (output_dir / "unet.best-dice.pth").is_file()
    assert (output_dir / "unet.last.pth").is_file()
    assert (output_dir / "training.csv").is_file()
    assert (output_dir / "training_curves.png").is_file()
    assert (output_dir / "training_config.json").is_file()

    resumed = {
        **config,
        "epochs": 2,
        "model_path": str(output_dir / "unet.last.pth"),
    }
    segmentation_train.TrainSegmentationModel(resumed).execute()
    rows = (output_dir / "training.csv").read_text().strip().splitlines()
    assert len(rows) == 3
    last = torch.load(output_dir / "unet.last.pth", weights_only=True)
    assert last["completed_epoch"] == 2
    assert len(last["history"]) == 2


def test_benchmark_command_writes_predictions_and_research_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "dataset" / "test"
    (split / "images").mkdir(parents=True)
    (split / "masks").mkdir()
    for index in range(2):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[2:6, 2:6] = 255
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        cv2.imwrite(str(split / "images" / f"sample_{index}.png"), image)
        cv2.imwrite(str(split / "masks" / f"sample_{index}.png"), mask)

    checkpoint_path = tmp_path / "fake.pth"
    checkpoint_path.write_bytes(b"research-checkpoint")
    model = nn.Conv2d(3, 2, kernel_size=1)
    metadata = {
        "checkpoint_path": checkpoint_path,
        "class_names": ["background", "foreground"],
        "colored": True,
        "input_size": (8, 8),
        "mask_threshold": 0.5,
        "model_name": "unet",
        "num_classes": 2,
        "palette": [[0, 0, 0], [255, 80, 80]],
    }
    monkeypatch.setattr(
        segmentation_evaluation,
        "load_checkpoint_bundle",
        lambda config: (model, metadata),
    )
    output_dir = tmp_path / "benchmark"
    result = segmentation_evaluation.BenchmarkSegmentation(
        {
            "batch_size": 1,
            "boundary_tolerance": 1,
            "calibration_bins": 5,
            "dataset_path": str(tmp_path / "dataset"),
            "device": "cpu",
            "mask_threshold": 0.5,
            "output_path": str(output_dir),
            "overlay_alpha": 0.45,
            "save_images": True,
            "split": "test",
            "threshold_steps": 5,
        }
    ).execute()

    assert result["evaluated_images"] == 2
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "class_metrics.csv").is_file()
    assert (output_dir / "image_metrics.csv").is_file()
    assert (output_dir / "run_metadata.json").is_file()
    assert len(list((output_dir / "predictions" / "masks").glob("*.png"))) == 2
    assert len(list((output_dir / "predictions" / "overlays").glob("*.png"))) == 2
    assert len(list((output_dir / "predictions" / "errors").glob("*.png"))) == 2
