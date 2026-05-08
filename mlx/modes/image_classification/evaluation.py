from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console, print_info, print_success, print_warning
from mlx.modes.image_classification.data import (
    OneShotPairDataset,
    load_standard_classification_directory,
    resolve_evaluation_dir,
)
from mlx.modes.image_classification.utils import load_checkpoint_bundle


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
    num_pairs = config.get("num_pairs", 2000)
    dataset = OneShotPairDataset(
        test_path,
        input_size=metadata["input_size"],
        colored=metadata["colored"],
        n_pairs_per_class=num_pairs,
    )
    console.print(f"[green]Loaded {len(dataset)} one-shot pairs from {test_path}[/green]")

    pairs_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds = []
    probs = []
    targets = []
    with torch.no_grad():
        for img1, img2, target in tqdm(pairs_loader, desc="Evaluating pairs"):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            out = model(img1, img2)
            prob = out.item()
            probs.append(prob)
            preds.append(1 if prob > 0.5 else 0)
            targets.append(target.item())

    return _render_metrics(
        targets,
        preds,
        output_dir=_resolve_benchmark_output_dir(config),
        probabilities=np.asarray(probs, dtype=np.float64),
        class_names=["different", "same"],
    )


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
            logits = model(images)
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

    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
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
