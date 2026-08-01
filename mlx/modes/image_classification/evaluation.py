from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_success, print_warning
from mlx.modes.image_classification.data import (
    load_image_tensor,
    load_standard_classification_directory,
    resolve_evaluation_dir,
)
from mlx.modes.image_classification.utils import load_checkpoint_bundle
from mlx.modes.image_classification.models.joint_svdd import JointDeepSVDDClassifier


@dataclass(frozen=True)
class OneShotPairRecord:
    image_one: Path
    label_one: str
    image_two: Path
    label_two: str
    target: int


@dataclass(frozen=True)
class OneShotPredictionRecord:
    image_one: Path
    label_one: str
    image_two: Path
    label_two: str
    target: int
    predicted: int
    same_probability: float


def benchmark_image_classification(config: dict[str, Any]) -> dict[str, float]:
    model, metadata = load_checkpoint_bundle(config)
    family = metadata["family"]
    device = config.get("device", "cpu")
    model = model.to(device)
    model.eval()

    if family == "one-shot":
        return _benchmark_one_shot(model, metadata, config, device)
    return _benchmark_standard(model, metadata, config, device)


def _benchmark_one_shot(model, metadata: dict[str, Any], config: dict[str, Any], device: str) -> dict[str, float]:
    test_path = resolve_evaluation_dir(config["dataset_path"])
    pairs_per_class = config.get("num_pairs", 100)
    pairs = _build_one_shot_benchmark_pairs(
        test_path,
        pairs_per_class=pairs_per_class,
        random_seed=config.get("random_seed"),
    )
    console.print(
        f"[green]Loaded {len(pairs)} deterministic one-shot benchmark pairs "
        f"from {test_path}[/green]"
    )

    prediction_records: list[OneShotPredictionRecord] = []
    batch_size = max(1, int(config.get("batch_size", 1)))
    batch_count = max(1, (len(pairs) + batch_size - 1) // batch_size)
    with torch.no_grad():
        for pair_batch in tqdm(
            _batched(pairs, batch_size),
            desc="Evaluating pairs",
            total=batch_count,
        ):
            images_one = torch.stack(
                [
                    load_image_tensor(
                        pair.image_one,
                        input_size=metadata["input_size"],
                        colored=metadata["colored"],
                    )
                    for pair in pair_batch
                ]
            ).to(device)
            images_two = torch.stack(
                [
                    load_image_tensor(
                        pair.image_two,
                        input_size=metadata["input_size"],
                        colored=metadata["colored"],
                    )
                    for pair in pair_batch
                ]
            ).to(device)
            outputs = model(images_one, images_two).detach().reshape(-1).cpu().tolist()
            for pair, same_probability in zip(pair_batch, outputs):
                prediction_records.append(
                    OneShotPredictionRecord(
                        image_one=pair.image_one,
                        label_one=pair.label_one,
                        image_two=pair.image_two,
                        label_two=pair.label_two,
                        target=pair.target,
                        predicted=1 if same_probability >= 0.5 else 0,
                        same_probability=float(same_probability),
                    )
                )

    targets = [record.target for record in prediction_records]
    preds = [record.predicted for record in prediction_records]
    probs = np.asarray([record.same_probability for record in prediction_records], dtype=np.float64)
    output_dir = _resolve_benchmark_output_dir(config)
    threshold_rows, threshold_metrics = _compute_one_shot_threshold_metrics(targets, probs)
    n_way_summary = _benchmark_one_shot_n_way(
        model,
        metadata,
        test_path,
        config,
        device,
        output_dir=output_dir,
    )
    threshold_metrics.update(n_way_summary)

    results = _render_metrics(
        targets,
        preds,
        output_dir=output_dir,
        probabilities=probs,
        class_names=["different", "same"],
        extra_metrics=threshold_metrics,
        title="One-Shot Pair Verification Results",
    )
    if output_dir is not None:
        _write_one_shot_research_artifacts(
            output_dir,
            prediction_records=prediction_records,
            threshold_rows=threshold_rows,
            targets=targets,
            probabilities=probs,
        )
    return results


def _benchmark_standard(model, metadata: dict[str, Any], config: dict[str, Any], device: str) -> dict[str, float]:
    if not metadata["classes"]:
        raise MLXUserError("The checkpoint does not contain class labels for standard evaluation.")
    eval_dir = resolve_evaluation_dir(config["dataset_path"])
    dataset = load_standard_classification_directory(
        eval_dir,
        label_names=metadata["classes"],
        input_size=metadata["input_size"],
        colored=metadata["colored"],
    )
    console.print(f"[green]Loaded {len(dataset)} labelled images from {eval_dir}[/green]")

    loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=2)
    preds = []
    probabilities = []
    targets = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating images"):
            images = images.to(device)
            output = model(images)
            logits = output.logits if isinstance(model, JointDeepSVDDClassifier) else output
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            batch_preds = logits.argmax(dim=1).cpu().tolist()
            probabilities.extend(batch_probs)
            preds.extend(batch_preds)
            targets.extend(labels.tolist())

    return _render_metrics(
        targets,
        preds,
        output_dir=_resolve_benchmark_output_dir(config),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        class_names=metadata["classes"],
    )


def _build_one_shot_benchmark_pairs(
    dataset_path: Path,
    *,
    pairs_per_class: int,
    random_seed: int | None,
) -> list[OneShotPairRecord]:
    if pairs_per_class <= 0:
        raise MLXUserError("--num-pairs must be greater than zero for one-shot benchmarking.")

    label_to_images = _index_one_shot_images(dataset_path)
    if len(label_to_images) < 2:
        raise MLXUserError(
            "One-shot benchmarking requires at least two label directories with at least two images each."
        )

    rng = random.Random(random_seed)
    labels = sorted(label_to_images)
    positive_count = (pairs_per_class + 1) // 2
    negative_count = pairs_per_class // 2
    pairs: list[OneShotPairRecord] = []

    for label in labels:
        images = label_to_images[label]
        for _ in range(positive_count):
            image_one, image_two = rng.sample(images, 2)
            pairs.append(
                OneShotPairRecord(
                    image_one=image_one,
                    label_one=label,
                    image_two=image_two,
                    label_two=label,
                    target=1,
                )
            )

        negative_labels = [candidate for candidate in labels if candidate != label]
        for _ in range(negative_count):
            negative_label = rng.choice(negative_labels)
            pairs.append(
                OneShotPairRecord(
                    image_one=rng.choice(images),
                    label_one=label,
                    image_two=rng.choice(label_to_images[negative_label]),
                    label_two=negative_label,
                    target=0,
                )
            )

    rng.shuffle(pairs)
    return pairs


def _index_one_shot_images(dataset_path: Path) -> dict[str, list[Path]]:
    label_to_images: dict[str, list[Path]] = {}
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    for label_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        if len(images) >= 2:
            label_to_images[label_dir.name] = images
        else:
            print_warning(
                f"Skipping label '{label_dir.name}' for one-shot benchmark because it has fewer than two images."
            )
    return label_to_images


def _batched(items: list[OneShotPairRecord], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _benchmark_one_shot_n_way(
    model,
    metadata: dict[str, Any],
    dataset_path: Path,
    config: dict[str, Any],
    device: str,
    *,
    output_dir: Path | None,
) -> dict[str, float]:
    label_to_images = _index_one_shot_images(dataset_path)
    labels = sorted(label_to_images)
    if len(labels) < 2:
        return {}

    label_to_index = {label: index for index, label in enumerate(labels)}
    samples = [
        (image_path, label)
        for label in labels
        for image_path in label_to_images[label]
    ]
    targets: list[int] = []
    preds: list[int] = []
    probabilities: list[list[float]] = []
    prediction_rows: list[dict[str, Any]] = []
    batch_size = max(1, int(config.get("batch_size", 1)))

    with torch.no_grad():
        for query_path, query_label in tqdm(samples, desc="Evaluating N-way queries"):
            reference_records = [
                (reference_path, reference_label)
                for reference_label in labels
                for reference_path in label_to_images[reference_label]
                if reference_path != query_path
            ]
            label_scores = {label: 0.0 for label in labels}
            best_label = None
            best_path = None
            best_score = -1.0
            query_tensor = load_image_tensor(
                query_path,
                input_size=metadata["input_size"],
                colored=metadata["colored"],
            )

            for reference_batch in _batched_reference_records(reference_records, batch_size):
                query_batch = torch.stack([query_tensor for _ in reference_batch]).to(device)
                reference_batch_tensor = torch.stack(
                    [
                        load_image_tensor(
                            reference_path,
                            input_size=metadata["input_size"],
                            colored=metadata["colored"],
                        )
                        for reference_path, _ in reference_batch
                    ]
                ).to(device)
                scores = model(query_batch, reference_batch_tensor).detach().reshape(-1).cpu().tolist()
                for (reference_path, reference_label), score in zip(reference_batch, scores):
                    score = float(score)
                    if score > label_scores[reference_label]:
                        label_scores[reference_label] = score
                    if score > best_score:
                        best_score = score
                        best_label = reference_label
                        best_path = reference_path

            class_scores = np.asarray([label_scores[label] for label in labels], dtype=np.float64)
            class_probabilities = _softmax(class_scores)
            target_index = label_to_index[query_label]
            predicted_index = label_to_index[best_label] if best_label is not None else int(class_probabilities.argmax())
            targets.append(target_index)
            preds.append(predicted_index)
            probabilities.append(class_probabilities.tolist())
            prediction_rows.append(
                {
                    "query_image": str(query_path),
                    "query_label": query_label,
                    "predicted_label": labels[predicted_index],
                    "best_reference_image": str(best_path) if best_path is not None else "",
                    "best_same_probability": best_score,
                    "correct": int(target_index == predicted_index),
                }
            )

    n_way_output_dir = output_dir / "n_way_classification" if output_dir is not None else None
    if n_way_output_dir is not None:
        n_way_output_dir.mkdir(parents=True, exist_ok=True)
    n_way_results = _render_metrics(
        targets,
        preds,
        output_dir=n_way_output_dir,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        class_names=labels,
        title="One-Shot N-Way Classification Results",
    )
    if n_way_output_dir is not None:
        _write_n_way_predictions(n_way_output_dir / "predictions.csv", prediction_rows)

    return {
        "n_way_accuracy": n_way_results["accuracy"],
        "n_way_precision": n_way_results["precision"],
        "n_way_recall": n_way_results["recall"],
        "n_way_f1": n_way_results["f1"],
    }


def _batched_reference_records(items: list[tuple[Path, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    total = exponentials.sum()
    if total == 0:
        return np.full_like(values, 1.0 / len(values), dtype=np.float64)
    return exponentials / total


def _resolve_benchmark_output_dir(config: dict[str, Any]) -> Path | None:
    output_path = config.get("output_path")
    if not output_path:
        return None
    output_dir = Path(output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _render_metrics(
    targets: list[int],
    preds: list[int],
    *,
    output_dir: Path | None = None,
    probabilities: np.ndarray | None = None,
    class_names: list[str] | None = None,
    extra_metrics: dict[str, float] | None = None,
    title: str = "Benchmark Results",
) -> dict[str, float]:
    results = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
        "f1": f1_score(targets, preds, average="macro", zero_division=0),
    }
    results["avg_precision"] = results["precision"]
    results["avg_recall"] = results["recall"]
    results.update(_compute_classwise_metrics(targets, preds, class_names=class_names))
    roc_results = _compute_roc_metrics(targets, probabilities, class_names=class_names)
    results.update(roc_results)
    if extra_metrics:
        results.update(extra_metrics)

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Score", justify="right")
    table.add_row("Accuracy", f"{results['accuracy']:.4f}")
    table.add_row("Ave Precision", f"{results['avg_precision']:.4f}")
    table.add_row("Ave Recall", f"{results['avg_recall']:.4f}")
    table.add_row("F1-score", f"{results['f1']:.4f}")
    if "roc_auc_macro_ovr" in results:
        table.add_row("ROC AUC (macro)", f"{results['roc_auc_macro_ovr']:.4f}")
    if "roc_auc_weighted_ovr" in results:
        table.add_row("ROC AUC (weighted)", f"{results['roc_auc_weighted_ovr']:.4f}")
    if "average_precision" in results:
        table.add_row("Average Precision", f"{results['average_precision']:.4f}")
    if "equal_error_rate" in results:
        table.add_row("Equal Error Rate", f"{results['equal_error_rate']:.4f}")
    if "best_f1_threshold" in results:
        table.add_row("Best F1 Threshold", f"{results['best_f1_threshold']:.4f}")
    console.print(table)
    _render_classwise_metrics_table(results, class_names=class_names)
    if output_dir is not None:
        _write_benchmark_artifacts(
            output_dir,
            results=results,
            targets=targets,
            preds=preds,
            probabilities=probabilities,
            class_names=class_names,
        )
    return results


def _compute_classwise_metrics(
    targets: list[int],
    preds: list[int],
    *,
    class_names: list[str] | None,
) -> dict[str, float]:
    if len(targets) == 0:
        return {}

    labels = list(range(len(class_names))) if class_names else sorted(set(targets) | set(preds))
    matrix = confusion_matrix(targets, preds, labels=labels)
    total = matrix.sum()
    resolved_names = class_names or [str(label) for label in labels]

    results: dict[str, float] = {}
    for class_index, class_name in enumerate(resolved_names):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        tn = float(total - tp - fp - fn)

        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        slug = _metric_slug(class_name)
        results[f"sensitivity_{slug}"] = sensitivity
        results[f"specificity_{slug}"] = specificity

    return results


def _compute_roc_metrics(
    targets: list[int],
    probabilities: np.ndarray | None,
    *,
    class_names: list[str] | None,
) -> dict[str, float]:
    if probabilities is None or len(targets) == 0:
        return {}

    target_array = np.asarray(targets, dtype=np.int64)
    unique_targets = np.unique(target_array)
    if unique_targets.size < 2:
        print_warning("ROC/AUC skipped because the evaluation data contains fewer than two classes.")
        return {}

    try:
        if probabilities.ndim == 1 or (probabilities.ndim == 2 and probabilities.shape[1] == 1):
            positive_scores = probabilities.reshape(-1)
            score = roc_auc_score(target_array, positive_scores)
            results = {"roc_auc_macro_ovr": score}
            if class_names and len(class_names) >= 2:
                negative_scores = 1.0 - positive_scores
                results[f"auc_{_metric_slug(class_names[0])}"] = roc_auc_score(
                    (target_array == 0).astype(int),
                    negative_scores,
                )
                results[f"auc_{_metric_slug(class_names[1])}"] = roc_auc_score(
                    (target_array == 1).astype(int),
                    positive_scores,
                )
            else:
                positive_label = class_names[0] if class_names else "positive"
                results[f"auc_{_metric_slug(positive_label)}"] = score
            return results

        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            positive_scores = probabilities[:, 1]
            score = roc_auc_score(target_array, positive_scores)
            results = {
                "roc_auc_macro_ovr": score,
                "roc_auc_weighted_ovr": score,
            }
            resolved_names = class_names or ["negative", "positive"]
            for class_index in range(2):
                class_label = resolved_names[class_index] if class_index < len(resolved_names) else f"class_{class_index}"
                results[f"auc_{_metric_slug(class_label)}"] = roc_auc_score(
                    (target_array == class_index).astype(int),
                    probabilities[:, class_index],
                )
            return results

        class_count = probabilities.shape[1]
        classes = np.arange(class_count)
        binarized_targets = label_binarize(target_array, classes=classes)
        results = {
            "roc_auc_macro_ovr": roc_auc_score(
                binarized_targets,
                probabilities,
                multi_class="ovr",
                average="macro",
            ),
            "roc_auc_weighted_ovr": roc_auc_score(
                binarized_targets,
                probabilities,
                multi_class="ovr",
                average="weighted",
            ),
        }
        resolved_names = class_names or [f"class_{index}" for index in range(class_count)]
        for class_index in range(class_count):
            if binarized_targets[:, class_index].max() == 0:
                continue
            class_label = (
                resolved_names[class_index] if class_index < len(resolved_names) else f"class_{class_index}"
            )
            results[f"auc_{_metric_slug(class_label)}"] = roc_auc_score(
                binarized_targets[:, class_index],
                probabilities[:, class_index],
            )
        return results
    except ValueError as exc:
        class_summary = f" for classes {class_names}" if class_names else ""
        print_warning(f"ROC/AUC skipped{class_summary}: {exc}")
        return {}


def _metric_slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _render_classwise_metrics_table(results: dict[str, float], *, class_names: list[str] | None) -> None:
    if not class_names:
        return

    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for class_name in class_names:
        slug = _metric_slug(class_name)
        auc_value = results.get(f"auc_{slug}")
        sensitivity = results.get(f"sensitivity_{slug}")
        specificity = results.get(f"specificity_{slug}")
        if auc_value is None and sensitivity is None and specificity is None:
            continue
        rows.append((class_name, auc_value, sensitivity, specificity))

    if not rows:
        return

    table = Table(title="Per-Class Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Class", style="dim")
    table.add_column("AUC", justify="right")
    table.add_column("Sensitivity", justify="right")
    table.add_column("Specificity", justify="right")
    for class_name, auc_value, sensitivity, specificity in rows:
        table.add_row(
            class_name,
            f"{auc_value:.4f}" if auc_value is not None else "-",
            f"{sensitivity:.4f}" if sensitivity is not None else "-",
            f"{specificity:.4f}" if specificity is not None else "-",
        )
    console.print(table)


def _write_benchmark_artifacts(
    output_dir: Path,
    *,
    results: dict[str, float],
    targets: list[int],
    preds: list[int],
    probabilities: np.ndarray | None,
    class_names: list[str] | None,
) -> None:
    _write_metrics_csv(output_dir / "metrics.csv", results)
    _write_confusion_matrix_artifacts(
        output_dir,
        targets=targets,
        preds=preds,
        class_names=class_names,
    )
    _write_roc_curve_artifact(
        output_dir / "roc_curve.png",
        targets=targets,
        probabilities=probabilities,
        class_names=class_names,
    )
    print_success(f"Benchmark artifacts written to {output_dir}")


def _write_metrics_csv(csv_path: Path, results: dict[str, float]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for metric_name, metric_value in sorted(results.items()):
            writer.writerow([metric_name, f"{metric_value:.6f}"])


def _write_confusion_matrix_artifacts(
    output_dir: Path,
    *,
    targets: list[int],
    preds: list[int],
    class_names: list[str] | None,
) -> None:
    labels = list(range(len(class_names))) if class_names else sorted(set(targets) | set(preds))
    matrix = confusion_matrix(targets, preds, labels=labels)
    resolved_names = class_names or [str(label) for label in labels]

    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual/predicted", *resolved_names])
        for class_name, row in zip(resolved_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    figure_width = max(6, min(16, len(resolved_names) * 1.2))
    figure_height = max(5, min(14, len(resolved_names) * 1.0))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(resolved_names)),
        yticks=np.arange(len(resolved_names)),
        xticklabels=resolved_names,
        yticklabels=resolved_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_roc_curve_artifact(
    image_path: Path,
    *,
    targets: list[int],
    probabilities: np.ndarray | None,
    class_names: list[str] | None,
) -> None:
    if probabilities is None or len(targets) == 0:
        return

    target_array = np.asarray(targets, dtype=np.int64)
    unique_targets = np.unique(target_array)
    if unique_targets.size < 2:
        return

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        if probabilities.ndim == 1 or (probabilities.ndim == 2 and probabilities.shape[1] == 1):
            positive_scores = probabilities.reshape(-1)
            fpr, tpr, _ = roc_curve(target_array, positive_scores)
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}", linewidth=2)
        elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
            positive_scores = probabilities[:, 1]
            fpr, tpr, _ = roc_curve(target_array, positive_scores)
            positive_label = class_names[1] if class_names and len(class_names) > 1 else "positive"
            ax.plot(fpr, tpr, label=f"{positive_label} (AUC={auc(fpr, tpr):.4f})", linewidth=2)
        else:
            class_count = probabilities.shape[1]
            classes = np.arange(class_count)
            binarized_targets = label_binarize(target_array, classes=classes)
            for class_index in range(class_count):
                if binarized_targets[:, class_index].max() == 0:
                    continue
                fpr, tpr, _ = roc_curve(binarized_targets[:, class_index], probabilities[:, class_index])
                label = class_names[class_index] if class_names and class_index < len(class_names) else f"class_{class_index}"
                ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.4f})", linewidth=2)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set(
            title="ROC Curve",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.05),
        )
        ax.legend(loc="lower right", fontsize="small")
        fig.tight_layout()
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ValueError as exc:
        print_warning(f"ROC curve export skipped: {exc}")


def _compute_one_shot_threshold_metrics(
    targets: list[int],
    probabilities: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if len(targets) == 0:
        return [], {}

    target_array = np.asarray(targets, dtype=np.int64)
    thresholds = sorted(
        {0.0, 0.5, 1.0, *probabilities.tolist()},
        reverse=True,
    )
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(np.int64)
        tp = float(((target_array == 1) & (predicted == 1)).sum())
        fp = float(((target_array == 0) & (predicted == 1)).sum())
        tn = float(((target_array == 0) & (predicted == 0)).sum())
        fn = float(((target_array == 1) & (predicted == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
            }
        )

    metrics: dict[str, float] = {}
    unique_targets = np.unique(target_array)
    if unique_targets.size >= 2:
        metrics["average_precision"] = float(average_precision_score(target_array, probabilities))

        fpr, tpr, roc_thresholds = roc_curve(target_array, probabilities)
        fnr = 1.0 - tpr
        finite_indices = [index for index, value in enumerate(roc_thresholds) if np.isfinite(value)]
        if finite_indices:
            eer_index = min(finite_indices, key=lambda index: abs(fpr[index] - fnr[index]))
            metrics["equal_error_rate"] = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
            metrics["equal_error_rate_threshold"] = float(roc_thresholds[eer_index])

    if rows:
        best_f1_row = max(rows, key=lambda row: (row["f1"], row["accuracy"], row["threshold"]))
        best_youden_row = max(
            rows,
            key=lambda row: (row["recall"] + row["specificity"] - 1.0, row["accuracy"], row["threshold"]),
        )
        metrics.update(
            {
                "best_f1": best_f1_row["f1"],
                "best_f1_threshold": best_f1_row["threshold"],
                "best_f1_accuracy": best_f1_row["accuracy"],
                "best_youden_j": best_youden_row["recall"] + best_youden_row["specificity"] - 1.0,
                "best_youden_threshold": best_youden_row["threshold"],
                "best_youden_accuracy": best_youden_row["accuracy"],
            }
        )

    return rows, metrics


def _write_one_shot_research_artifacts(
    output_dir: Path,
    *,
    prediction_records: list[OneShotPredictionRecord],
    threshold_rows: list[dict[str, float]],
    targets: list[int],
    probabilities: np.ndarray,
) -> None:
    _write_one_shot_pair_predictions(output_dir / "pair_predictions.csv", prediction_records)
    _write_one_shot_threshold_metrics(output_dir / "threshold_metrics.csv", threshold_rows)
    _write_precision_recall_curve_artifact(
        output_dir / "precision_recall_curve.png",
        targets=targets,
        probabilities=probabilities,
    )
    _write_score_distribution_artifact(
        output_dir / "score_distribution.png",
        targets=targets,
        probabilities=probabilities,
    )
    print_success(f"One-shot research artifacts written to {output_dir}")


def _write_one_shot_pair_predictions(
    csv_path: Path,
    prediction_records: list[OneShotPredictionRecord],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "pair_id",
                "image_1",
                "label_1",
                "image_2",
                "label_2",
                "target",
                "predicted",
                "same_probability",
            ]
        )
        for index, record in enumerate(prediction_records, start=1):
            writer.writerow(
                [
                    index,
                    str(record.image_one),
                    record.label_one,
                    str(record.image_two),
                    record.label_two,
                    record.target,
                    record.predicted,
                    f"{record.same_probability:.8f}",
                ]
            )


def _write_one_shot_threshold_metrics(
    csv_path: Path,
    threshold_rows: list[dict[str, float]],
) -> None:
    fieldnames = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in threshold_rows:
            writer.writerow(
                {
                    key: f"{row[key]:.8f}" if key in row else ""
                    for key in fieldnames
                }
            )


def _write_n_way_predictions(csv_path: Path, prediction_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query_image",
        "query_label",
        "predicted_label",
        "best_reference_image",
        "best_same_probability",
        "correct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow(
                {
                    **row,
                    "best_same_probability": f"{row['best_same_probability']:.8f}",
                }
            )


def _write_precision_recall_curve_artifact(
    image_path: Path,
    *,
    targets: list[int],
    probabilities: np.ndarray,
) -> None:
    if len(targets) == 0 or np.unique(np.asarray(targets, dtype=np.int64)).size < 2:
        return

    precision, recall, _ = precision_recall_curve(targets, probabilities)
    average_precision = average_precision_score(targets, probabilities)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"AP={average_precision:.4f}", linewidth=2)
    ax.set(
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.05),
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_score_distribution_artifact(
    image_path: Path,
    *,
    targets: list[int],
    probabilities: np.ndarray,
) -> None:
    if len(targets) == 0:
        return

    target_array = np.asarray(targets, dtype=np.int64)
    same_scores = probabilities[target_array == 1]
    different_scores = probabilities[target_array == 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0.0, 1.0, 41)
    if different_scores.size:
        ax.hist(different_scores, bins=bins, alpha=0.65, label="different", density=True)
    if same_scores.size:
        ax.hist(same_scores, bins=bins, alpha=0.65, label="same", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set(
        title="One-Shot Similarity Score Distribution",
        xlabel="Predicted same-class probability",
        ylabel="Density",
        xlim=(0.0, 1.0),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
