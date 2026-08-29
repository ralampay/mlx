from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.data import (
    load_one_shot_datasets,
    load_standard_classification_datasets,
)
from mlx.modes.image_classification.models import (
    build_image_classification_model,
    model_family_for,
)
from mlx.modes.image_classification.models.joint_svdd import JointDeepSVDDClassifier
from mlx.modes.image_classification.ood.deep_svdd import (
    calibrate_svdd_threshold,
    collect_svdd_scores,
    compute_svdd_loss,
    initialize_svdd_center,
    validate_svdd_config,
)
from mlx.modes.image_classification.requests import ImageClassificationRequest
from mlx.modes.image_classification.utils import (
    load_training_checkpoint,
    resolve_model_name,
    resolve_train_output_paths,
    save_checkpoint,
    save_training_checkpoint,
    update_checkpoint_model,
)

TRAINING_CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
]
SVDD_TRAINING_CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "train_classification_loss",
    "train_svdd_loss",
    "val_loss",
    "val_classification_loss",
    "val_svdd_loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


class TrainImageClassificationModel:
    def __init__(
        self,
        request: ImageClassificationRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> None:
        config = self.request.to_config()
        if int(config.get("epochs", 0)) < 1:
            raise MLXUserError("--epochs must be at least 1 for image-classification training.")
        model_name = resolve_model_name(config)
        family = model_family_for(model_name)
        validate_svdd_config(config)
        if family == "one-shot":
            if config.get("ood_method", "none") != "none":
                raise MLXUserError(
                    "Deep SVDD is supported only for standard image-classification models, "
                    "not one-shot models."
                )
            _train_one_shot(model_name, config, reporter=self.reporter)
            return
        _train_standard(model_name, config, reporter=self.reporter)


class SmokeTestImageClassificationModel:
    def __init__(
        self,
        request: ImageClassificationRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> None:
        config = self.request.to_config()
        model_name = resolve_model_name(config)
        family = model_family_for(model_name)
        if family == "one-shot":
            _test_one_shot(model_name, config, reporter=self.reporter)
            return
        _test_standard(model_name, config, reporter=self.reporter)


def train_image_classification(config: dict[str, Any]) -> None:
    from mlx.modes.image_classification.presentation import RichImageClassificationReporter

    TrainImageClassificationModel(
        ImageClassificationRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
    ).execute()


def smoke_test_image_classification(config: dict[str, Any]) -> None:
    from mlx.modes.image_classification.presentation import RichImageClassificationReporter

    SmokeTestImageClassificationModel(
        ImageClassificationRequest.from_config(config),
        reporter=RichImageClassificationReporter(),
    ).execute()


def _train_one_shot(
    model_name: str,
    config: dict[str, Any],
    *,
    reporter: WorkflowReporter | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    device = config["device"]
    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 20)
    learning_rate = config.get("lr") or 1e-4
    input_size = config.get("input_size", (105, 105))
    colored = config.get("colored", True)
    use_best = bool(config.get("use_best", False))
    output_paths = resolve_train_output_paths(config, model_name=model_name)
    checkpoint_path = output_paths["checkpoint_path"]
    last_checkpoint_path = output_paths["last_checkpoint_path"]
    training_csv_path = output_paths["training_csv_path"]

    emit(reporter, "info", f"Starting one-shot training on device={device} for {epochs} epochs")

    model = build_image_classification_model(model_name, config).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch, best_val_loss, history = _prepare_training_state(
        config,
        model,
        optimizer,
        model_name=model_name,
        family="one-shot",
        checkpoint_path=checkpoint_path,
        training_csv_path=training_csv_path,
        reporter=reporter,
    )

    train_dataset, val_dataset = load_one_shot_datasets(
        dataset_path,
        input_size=input_size,
        colored=colored,
        n_pairs_per_class=config.get("num_pairs", 100),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss, val_metrics = _validate_one_shot(model, val_loader, criterion, device)
        history.append(
            _training_history_row(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                metrics=val_metrics,
            )
        )
        _write_training_csv(training_csv_path, history)

        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
        saved_checkpoint = not use_best or improved
        checkpoint_message = None
        if saved_checkpoint:
            save_checkpoint(
                checkpoint_path,
                model,
                model_name=model_name,
                family="one-shot",
                config=config,
            )
            checkpoint_message = (
                f"Saved new best model at {checkpoint_path}"
                if use_best
                else f"Saved epoch {epoch + 1} model at {checkpoint_path}"
            )

        save_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            model_name=model_name,
            family="one-shot",
            config=config,
            completed_epoch=epoch + 1,
            best_val_loss=best_val_loss,
            history=history,
        )
        _emit_epoch_result(
            reporter,
            epoch=epoch + 1,
            epochs=epochs,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            metrics=val_metrics,
            checkpoint_message=checkpoint_message,
            checkpoint_path=last_checkpoint_path,
        )

    emit(reporter, "success", "One-shot training complete!")


def _train_standard(
    model_name: str,
    config: dict[str, Any],
    *,
    reporter: WorkflowReporter | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    device = config["device"]
    dataset_path = config["dataset_path"]
    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 20)
    learning_rate = config.get("lr") or 1e-3
    input_size = config.get("input_size", (224, 224))
    colored = config.get("colored", True)
    apply_transformations = bool(config.get("apply_transformations", False))
    use_best = bool(config.get("use_best", False))
    joint_svdd = config.get("ood_method", "none") == "deep-svdd"
    svdd_weight = float(config.get("svdd_weight", 0.05))
    svdd_warmup_epochs = int(config.get("svdd_warmup_epochs", 0))
    output_paths = resolve_train_output_paths(config, model_name=model_name)
    checkpoint_path = output_paths["checkpoint_path"]
    last_checkpoint_path = output_paths["last_checkpoint_path"]
    training_csv_path = output_paths["training_csv_path"]

    emit(
        reporter,
        "info",
        f"Starting standard classification training on device={device} for {epochs} epochs",
    )
    train_dataset, val_dataset, classes = load_standard_classification_datasets(
        dataset_path,
        input_size=input_size,
        colored=colored,
        apply_transformations=apply_transformations,
    )
    model = build_image_classification_model(
        model_name,
        config,
        num_classes=len(classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    start_epoch, best_val_loss, history = _prepare_training_state(
        config,
        model,
        optimizer,
        model_name=model_name,
        family="standard",
        checkpoint_path=checkpoint_path,
        training_csv_path=training_csv_path,
        classes=classes,
        reporter=reporter,
    )
    if joint_svdd and start_epoch == 0:
        initialize_svdd_center(model, train_loader, device)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_classification_loss = 0.0
        running_svdd_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(images)
            if joint_svdd:
                classification_loss = criterion(output.logits, targets)
                svdd_loss = compute_svdd_loss(model, output.svdd_embedding)
                loss = classification_loss
                if epoch >= svdd_warmup_epochs:
                    loss = loss + svdd_weight * svdd_loss
                running_classification_loss += classification_loss.item()
                running_svdd_loss += svdd_loss.item()
            else:
                loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        if joint_svdd:
            avg_val_loss, val_metrics, val_details = _validate_joint_svdd(
                model, val_loader, criterion, device, svdd_weight
            )
            history.append(
                _svdd_training_history_row(
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    train_classification_loss=running_classification_loss / len(train_loader),
                    train_svdd_loss=running_svdd_loss / len(train_loader),
                    val_loss=avg_val_loss,
                    val_classification_loss=val_details["classification_loss"],
                    val_svdd_loss=val_details["svdd_loss"],
                    metrics=val_metrics,
                )
            )
        else:
            avg_val_loss, val_metrics = _validate_standard(model, val_loader, criterion, device)
            history.append(
                _training_history_row(
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    metrics=val_metrics,
                )
            )
        _write_training_csv(training_csv_path, history)

        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
        saved_checkpoint = not use_best or improved
        checkpoint_message = None
        if saved_checkpoint:
            save_checkpoint(
                checkpoint_path,
                model,
                model_name=model_name,
                family="standard",
                config=config,
                classes=classes,
            )
            checkpoint_message = (
                f"Saved new best model at {checkpoint_path}"
                if use_best
                else f"Saved epoch {epoch + 1} model at {checkpoint_path}"
            )

        save_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            model_name=model_name,
            family="standard",
            config=config,
            completed_epoch=epoch + 1,
            best_val_loss=best_val_loss,
            history=history,
            classes=classes,
        )
        _emit_epoch_result(
            reporter,
            epoch=epoch + 1,
            epochs=epochs,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            metrics=val_metrics,
            checkpoint_message=checkpoint_message,
            checkpoint_path=last_checkpoint_path,
        )

    if joint_svdd:
        CalibrateDeepSVDDCheckpoints(
            model=model,
            checkpoint_paths=(checkpoint_path, last_checkpoint_path),
            val_loader=val_loader,
            device=device,
            model_name=model_name,
            config=config,
            classes=classes,
            reporter=reporter,
        ).execute()
    emit(reporter, "success", "Standard classification training complete!")


def _test_one_shot(
    model_name: str,
    config: dict[str, Any],
    *,
    reporter: WorkflowReporter | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    batch = config["batch_size"]
    height, width = config["input_size"]
    device = config["device"]
    colored = config["colored"]

    emit(reporter, "info", f"Running one-shot test on device={device} | input={height}x{width} | batch={batch}")
    model = build_image_classification_model(model_name, config).to(device)

    channels = 3 if colored else 1
    x1 = torch.randn(batch, channels, height, width).to(device)
    x2 = torch.randn(batch, channels, height, width).to(device)
    output = model(x1, x2)

    _report_test_output(reporter, "One-Shot Model Output", output)


def _test_standard(
    model_name: str,
    config: dict[str, Any],
    *,
    reporter: WorkflowReporter | None = None,
) -> None:
    reporter = reporter or NullWorkflowReporter()
    batch = config["batch_size"]
    height, width = config["input_size"]
    device = config["device"]
    colored = config["colored"]
    num_classes = config.get("num_classes", 4)

    emit(
        reporter,
        "info",
        f"Running standard classification test on device={device} | input={height}x{width} | batch={batch}"
    )
    model = build_image_classification_model(
        model_name,
        config,
        num_classes=num_classes,
    ).to(device)

    channels = 3 if colored else 1
    x = torch.randn(batch, channels, height, width).to(device)
    output = model(x)
    if hasattr(output, "logits"):
        output = output.logits
    _report_test_output(reporter, "Classification Model Output", output)


def _report_test_output(
    reporter: WorkflowReporter,
    title: str,
    output: torch.Tensor,
) -> None:
    emit(
        reporter,
        "success",
        "Test completed successfully!",
        payload={
            "event": "tensor_output",
            "title": title,
            "shape": list(output.shape),
            "values": output.flatten().tolist()[:16],
        },
    )


def _validate_one_shot(model, val_loader, criterion, device: str) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss = 0.0
    preds: list[int] = []
    targets_all: list[int] = []
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            loss = criterion(output, label.unsqueeze(1))
            val_loss += loss.item()
            predictions = (output >= 0.5).float().squeeze(1)
            preds.extend(int(item) for item in predictions.cpu().tolist())
            targets_all.extend(int(item) for item in label.cpu().tolist())
    avg_loss = val_loss / len(val_loader)
    return avg_loss, _compute_classification_metrics(targets_all, preds)


def _validate_standard(model, val_loader, criterion, device: str) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss = 0.0
    preds: list[int] = []
    targets_all: list[int] = []
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()
            predictions = logits.argmax(dim=1)
            preds.extend(int(item) for item in predictions.cpu().tolist())
            targets_all.extend(int(item) for item in targets.cpu().tolist())

    avg_loss = val_loss / len(val_loader)
    return avg_loss, _compute_classification_metrics(targets_all, preds)


def _validate_joint_svdd(
    model: JointDeepSVDDClassifier,
    val_loader,
    criterion,
    device: str,
    svdd_weight: float,
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.eval()
    classification_total = 0.0
    svdd_total = 0.0
    preds: list[int] = []
    targets_all: list[int] = []
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            classification_loss = criterion(output.logits, targets)
            svdd_loss = compute_svdd_loss(model, output.svdd_embedding)
            classification_total += classification_loss.item()
            svdd_total += svdd_loss.item()
            preds.extend(int(item) for item in output.logits.argmax(dim=1).cpu().tolist())
            targets_all.extend(int(item) for item in targets.cpu().tolist())
    classification_average = classification_total / len(val_loader)
    svdd_average = svdd_total / len(val_loader)
    return (
        classification_average + svdd_weight * svdd_average,
        _compute_classification_metrics(targets_all, preds),
        {
            "classification_loss": classification_average,
            "svdd_loss": svdd_average,
        },
    )


def _compute_classification_metrics(targets: list[int], preds: list[int]) -> dict[str, float]:
    if not targets:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    labels = sorted(set(targets) | set(preds))
    total = len(targets)
    accuracy = sum(int(target == pred) for target, pred in zip(targets, preds, strict=True)) / total

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for label in labels:
        tp = sum(1 for target, pred in zip(targets, preds, strict=True) if target == label and pred == label)
        fp = sum(1 for target, pred in zip(targets, preds, strict=True) if target != label and pred == label)
        fn = sum(1 for target, pred in zip(targets, preds, strict=True) if target == label and pred != label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    num_labels = len(labels)
    return {
        "accuracy": accuracy,
        "precision": macro_precision / num_labels,
        "recall": macro_recall / num_labels,
        "f1": macro_f1 / num_labels,
    }


def _prepare_training_state(
    config: dict[str, Any],
    model,
    optimizer,
    *,
    model_name: str,
    family: str,
    checkpoint_path: Path,
    training_csv_path: Path,
    classes: list[str] | None = None,
    reporter: WorkflowReporter | None = None,
) -> tuple[int, float, list[dict[str, float | int]]]:
    reporter = reporter or NullWorkflowReporter()
    resume_path = config.get("model_path")
    if not resume_path:
        _write_training_csv(training_csv_path, [])
        return 0, float("inf"), []

    training_state = load_training_checkpoint(
        resume_path,
        model,
        optimizer,
        model_name=model_name,
        family=family,
        config=config,
        classes=classes,
    )
    completed_epoch = training_state["completed_epoch"]
    target_epochs = int(config.get("epochs", 0))
    if completed_epoch > target_epochs:
        raise MLXUserError(
            f"Resume checkpoint has completed epoch {completed_epoch}, which exceeds "
            f"the requested --epochs target of {target_epochs}."
        )
    history = training_state["history"]
    _write_training_csv(training_csv_path, history)
    if not checkpoint_path.exists():
        save_checkpoint(
            checkpoint_path,
            model,
            model_name=model_name,
            family=family,
            config=config,
            classes=classes,
        )
    emit(
        reporter,
        "info",
        f"Resuming {model_name} from completed epoch {completed_epoch}: {resume_path}"
    )
    return completed_epoch, training_state["best_val_loss"], history


def _training_history_row(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, float],
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def _svdd_training_history_row(
    *,
    epoch: int,
    train_loss: float,
    train_classification_loss: float,
    train_svdd_loss: float,
    val_loss: float,
    val_classification_loss: float,
    val_svdd_loss: float,
    metrics: dict[str, float],
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_classification_loss": train_classification_loss,
        "train_svdd_loss": train_svdd_loss,
        "val_loss": val_loss,
        "val_classification_loss": val_classification_loss,
        "val_svdd_loss": val_svdd_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def _write_training_csv(
    csv_path: Path,
    history: list[dict[str, float | int]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        columns = (
            SVDD_TRAINING_CSV_COLUMNS
            if history and "train_svdd_loss" in history[0]
            else TRAINING_CSV_COLUMNS
        )
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    **{
                        column: f"{float(row[column]):.6f}"
                        for column in columns[1:]
                    },
                }
            )


class CalibrateDeepSVDDCheckpoints:
    def __init__(
        self,
        *,
        model: JointDeepSVDDClassifier,
        checkpoint_paths: tuple[Path, ...],
        val_loader,
        device: str,
        model_name: str,
        config: dict[str, Any],
        classes: list[str],
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_paths = checkpoint_paths
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.config = config
        self.classes = classes
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> None:
        calibrated = []
        for checkpoint_path in self.checkpoint_paths:
            checkpoint = self._load_model(checkpoint_path)
            scores = collect_svdd_scores(self.model, self.val_loader, self.device)
            threshold = calibrate_svdd_threshold(
                self.model, scores, float(self.config.get("svdd_quantile", 0.95))
            )
            update_checkpoint_model(
                checkpoint_path,
                checkpoint,
                self.model,
                model_name=self.model_name,
                family="standard",
                config=self.config,
                classes=self.classes,
            )
            calibrated.append(f"{checkpoint_path.name}={threshold:.6f}")
        emit(
            self.reporter,
            "success",
            "Calibrated Deep SVDD rejection thresholds from validation images: "
            + ", ".join(calibrated)
        )

    def _load_model(self, checkpoint_path: Path) -> dict[str, Any]:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            return checkpoint
        except (OSError, KeyError, RuntimeError, ValueError) as exc:
            raise MLXUserError(
                f"Could not reload checkpoint '{checkpoint_path}' for Deep SVDD "
                f"threshold calibration: {exc}"
            ) from exc


def _emit_epoch_result(
    reporter: WorkflowReporter,
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, float],
    checkpoint_message: str | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    values = [
        ("loss", train_loss),
        ("val_loss", val_loss),
        ("accuracy", metrics["accuracy"]),
        ("precision", metrics["precision"]),
        ("recall", metrics["recall"]),
        ("f1", metrics["f1"]),
    ]
    formatted_values = "  ".join(f"{label}: {value:.6f}" for label, value in values)
    emit(
        reporter,
        "progress",
        f"Epoch {epoch}/{epochs}  {formatted_values}",
        current=epoch,
        total=epochs,
        payload={
            "event": "training_epoch",
            "metrics": {"train_loss": train_loss, "val_loss": val_loss, **metrics},
            "checkpoint_message": checkpoint_message,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        },
    )
    if checkpoint_message:
        emit(reporter, "success", checkpoint_message)
