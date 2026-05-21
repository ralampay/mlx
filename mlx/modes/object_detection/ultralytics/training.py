from __future__ import annotations

import csv
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any, Optional, Union

from rich.panel import Panel
from rich.table import Table

from mlx.core.ui import console, print_info, print_success, print_warning
from mlx.modes.object_detection.ultralytics.utils import (
    initialize_model,
    resolve_dataset_source,
    resolve_imgsz,
    resolve_model_paths,
)


def train_object_detection(config: dict[str, Any]):
    resolved_cfg, resolved_weights = resolve_model_paths(
        config,
        require_yaml=True,
        require_weights=False,
    )
    resolved_dataset = resolve_dataset_source(config)
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 16)
    device = config.get("device", "cpu")
    use_best = bool(config.get("use_best", True))
    requested_imgsz = resolve_imgsz(config)
    imgsz = requested_imgsz
    if isinstance(requested_imgsz, tuple):
        imgsz = max(requested_imgsz)
        print_warning(
            "Ultralytics training currently uses square image sizes. "
            f"Requested imgsz={requested_imgsz} will fall back to imgsz={imgsz}."
        )
    project_dir = resolved_dataset.project_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.get("run_name", "mlx-ultralytics")
    lr0 = config.get("lr0")
    loss_clip = config.get("loss_clip")
    auto_resume_checkpoint, auto_warm_start_weights = _detect_existing_training_artifacts(
        project_dir=project_dir,
        run_name=run_name,
        explicit_weights=resolved_weights,
    )
    effective_weights = resolved_weights or auto_warm_start_weights

    console.print(Panel.fit("Ultralytics Object Detection - Training", border_style="cyan"))
    console.print(_training_summary_table(
        resolved_cfg=resolved_cfg,
        resolved_weights=effective_weights,
        resume_checkpoint=auto_resume_checkpoint,
        dataset_source=resolved_dataset.source,
        dataset_root=resolved_dataset.root_dir,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        imgsz=imgsz,
        project_dir=project_dir,
        run_name=run_name,
        use_best=use_best,
        config=config,
    ))

    if auto_resume_checkpoint is not None:
        print_info(
            "Continuing training from existing output directory "
            f"using checkpoint: {auto_resume_checkpoint}"
        )
    elif auto_warm_start_weights is not None:
        print_info(f"Warm-starting from checkpoint found in output directory: {auto_warm_start_weights}")

    print_info("Loading Ultralytics model...")
    model = initialize_model(resolved_cfg, effective_weights, prefer_cfg=True)
    overrides = getattr(model, "overrides", {})
    overrides["pretrained"] = bool(config.get("pretrained", False))
    overrides["model"] = str(resolved_cfg) if resolved_cfg else overrides.get("model")
    overrides.pop("weights", None)
    overrides["optimizer"] = config.get("optimizer", overrides.get("optimizer", "auto"))
    overrides["nbs"] = int(config.get("nbs", overrides.get("nbs", 64)))
    overrides["warmup_epochs"] = float(
        config.get("warmup_epochs", overrides.get("warmup_epochs", 3.0))
    )
    overrides["amp"] = bool(config.get("amp", overrides.get("amp", True)))
    model.overrides = overrides
    model.ckpt_path = str(effective_weights) if effective_weights else None

    train_kwargs = {
        "batch": batch_size,
        "data": resolved_dataset.data,
        "device": device,
        "epochs": epochs,
        "exist_ok": True,
        "imgsz": imgsz,
        "name": run_name,
        "plots": bool(config.get("plots", True)),
        "pretrained": overrides["pretrained"],
        "project": str(project_dir),
    }
    if auto_resume_checkpoint is not None:
        train_kwargs["resume"] = str(auto_resume_checkpoint)
    if lr0 is not None:
        train_kwargs["lr0"] = float(lr0)
    if loss_clip is not None:
        train_kwargs["loss_clip"] = float(loss_clip)
    if config.get("random_seed") is not None:
        train_kwargs["seed"] = int(config["random_seed"])

    print_info("Starting training loop...")
    results = model.train(**train_kwargs)
    _print_training_metrics(results)
    _export_training_graphs(results, project_dir=project_dir, run_name=run_name)
    selected_checkpoint = _select_training_checkpoint(
        results,
        project_dir=project_dir,
        run_name=run_name,
        use_best=use_best,
    )
    if selected_checkpoint is not None:
        config["model_path"] = str(selected_checkpoint)
        model.ckpt_path = str(selected_checkpoint)
        try:
            setattr(results, "model_path", selected_checkpoint)
            setattr(results, "checkpoint_path", selected_checkpoint)
        except Exception:
            pass
        checkpoint_label = "best" if use_best else "last"
        print_success(f"Selected {checkpoint_label} checkpoint for downstream use: {selected_checkpoint}")
    print_success("Training complete!")
    return results


def _detect_existing_training_artifacts(
    *,
    project_dir: Path,
    run_name: Optional[str],
    explicit_weights: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    if explicit_weights is not None or not project_dir.exists():
        return None, None

    run_dir = project_dir / run_name if run_name else None
    resume_checkpoint = _find_existing_checkpoint(
        project_dir=project_dir,
        run_dir=run_dir,
        file_name="last.pt",
    )
    if resume_checkpoint is not None:
        return resume_checkpoint, None

    warm_start_weights = _find_existing_checkpoint(
        project_dir=project_dir,
        run_dir=run_dir,
        file_name="best.pt",
    )
    if warm_start_weights is not None:
        return None, warm_start_weights

    warm_start_weights = _find_latest_checkpoint(project_dir, pattern="*.pt")
    if warm_start_weights is not None:
        return None, warm_start_weights

    return None, None


def _find_existing_checkpoint(
    *,
    project_dir: Path,
    run_dir: Optional[Path],
    file_name: str,
) -> Optional[Path]:
    preferred_candidates = []
    if run_dir is not None:
        preferred_candidates.extend((run_dir / "weights" / file_name, run_dir / file_name))

    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate.resolve()

    return _find_latest_checkpoint(project_dir, pattern=file_name)


def _find_latest_checkpoint(project_dir: Path, *, pattern: str) -> Optional[Path]:
    matches = [path for path in project_dir.rglob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path))).resolve()


def _training_summary_table(
    *,
    resolved_cfg,
    resolved_weights,
    resume_checkpoint: Optional[Path],
    dataset_source: str,
    dataset_root: Optional[Path],
    epochs: int,
    batch_size: int,
    device: str,
    imgsz: Union[int, tuple[int, int]],
    project_dir: Path,
    run_name: str,
    use_best: bool,
    config: dict[str, Any],
) -> Table:
    summary = Table(title="Training Configuration", show_lines=True)
    summary.add_column("Key", justify="right", style="cyan", no_wrap=True)
    summary.add_column("Value", style="magenta")
    summary.add_row(
        "Training Mode",
        "continue existing run" if resume_checkpoint else "new run",
    )
    summary.add_row("Init Weights", str(resolved_weights) if resolved_weights else "random init")
    summary.add_row("Resume From", str(resume_checkpoint) if resume_checkpoint else "disabled")
    summary.add_row("Model YAML", str(resolved_cfg) if resolved_cfg else "not set")
    summary.add_row("Dataset", dataset_source)
    summary.add_row("Dataset Root", str(dataset_root) if dataset_root else "managed by dataset YAML")
    summary.add_row("Epochs", str(epochs))
    summary.add_row("Batch Size", str(batch_size))
    summary.add_row("Device", str(device))
    summary.add_row("Image Size", str(imgsz))
    summary.add_row("Project", str(project_dir))
    summary.add_row("Run Name", run_name)
    summary.add_row("Use Best Checkpoint", str(use_best))
    summary.add_row("Pretrained", str(bool(config.get("pretrained", False))))
    summary.add_row("lr0", str(config.get("lr0")) if config.get("lr0") is not None else "default")
    summary.add_row("Optimizer", config.get("optimizer", "auto"))
    summary.add_row("nbs", str(config.get("nbs", 64)))
    summary.add_row("Warmup Epochs", str(config.get("warmup_epochs", 3.0)))
    summary.add_row("AMP", str(bool(config.get("amp", True))))
    summary.add_row(
        "Loss Clip",
        str(config.get("loss_clip")) if config.get("loss_clip") is not None else "disabled",
    )
    return summary


def _select_training_checkpoint(
    results: Any,
    *,
    project_dir: Path,
    run_name: str,
    use_best: bool,
) -> Optional[Path]:
    output_dir = _resolve_training_output_dir(results, project_dir=project_dir, run_name=run_name)
    preferred_name = "best.pt" if use_best else "last.pt"
    fallback_name = "last.pt" if use_best else "best.pt"

    preferred = _find_existing_checkpoint(
        project_dir=output_dir,
        run_dir=output_dir,
        file_name=preferred_name,
    )
    if preferred is not None:
        return preferred

    fallback = _find_existing_checkpoint(
        project_dir=output_dir,
        run_dir=output_dir,
        file_name=fallback_name,
    )
    if fallback is not None:
        print_warning(
            f"Preferred checkpoint {preferred_name} was not found; using {fallback_name} instead."
        )
        return fallback

    latest = _find_latest_checkpoint(output_dir, pattern="*.pt")
    if latest is not None:
        print_warning(f"Preferred checkpoint {preferred_name} was not found; using newest checkpoint.")
        return latest

    print_warning("Training finished, but no .pt checkpoint was found in the run directory.")
    return None


def _print_training_metrics(results: Any) -> None:
    metrics = _collect_training_metrics(results)
    if not metrics:
        print_warning("Training finished, but no validation metrics were exposed by Ultralytics.")
        return

    table = Table(title="Final Validation Metrics", show_lines=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="magenta")

    prioritized_metrics = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/F1(B)", "F1"),
        ("metrics/mAP50(B)", "mAP@0.50"),
        ("metrics/mAP50-95(B)", "mAP@0.50:0.95"),
        ("fitness", "Fitness"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Class Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Class Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
    ]

    rendered_keys: set[str] = set()
    for key, label in prioritized_metrics:
        if key not in metrics:
            continue
        table.add_row(label, _format_metric_value(metrics[key]))
        rendered_keys.add(key)

    auc_keys = _find_metric_keys(metrics, ("auc", "roc"))
    for key in auc_keys:
        if key in rendered_keys:
            continue
        table.add_row(_humanize_metric_label(key), _format_metric_value(metrics[key]))
        rendered_keys.add(key)

    remaining_keys = sorted(
        key for key in metrics if key not in rendered_keys and _is_scalar_metric(metrics[key])
    )
    for key in remaining_keys:
        table.add_row(_humanize_metric_label(key), _format_metric_value(metrics[key]))

    console.print(table)
    if not auc_keys:
        print_info(
            "ROC/AUC is not typically reported for object detection training. "
            "Ultralytics detection validation is primarily driven by IoU-based precision, recall, and AP."
        )


def _export_training_graphs(results: Any, *, project_dir: Path, run_name: str) -> None:
    output_dir = _resolve_training_output_dir(results, project_dir=project_dir, run_name=run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotted_files: list[Path] = []
    plotted_files.extend(_write_results_csv_graphs(output_dir))

    per_class_outputs = _write_per_class_map_artifacts(output_dir, results)
    plotted_files.extend(per_class_outputs)

    if plotted_files:
        print_info(
            "Saved training graphs to "
            f"{output_dir}: {', '.join(path.name for path in plotted_files)}"
        )
    else:
        print_warning(
            "Training completed, but MLX could not generate additional graphs from the run artifacts."
        )


def _collect_training_metrics(results: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    sources: list[dict[str, Any]] = []

    if isinstance(results, dict):
        sources.append(results)

    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        sources.append(results_dict)
    elif callable(results_dict):
        resolved = results_dict()
        if isinstance(resolved, dict):
            sources.append(resolved)

    box_metrics = getattr(results, "box", None)
    mean_results = getattr(box_metrics, "mean_results", None)
    if callable(mean_results):
        resolved = mean_results()
        if isinstance(resolved, Iterable):
            values = list(resolved)
            aliases = [
                ("metrics/precision(B)", 0),
                ("metrics/recall(B)", 1),
                ("metrics/mAP50(B)", 2),
                ("metrics/mAP50-95(B)", 3),
            ]
            for key, index in aliases:
                if index < len(values):
                    metrics[key] = values[index]

    for source in sources:
        for key, value in source.items():
            if _is_scalar_metric(value):
                metrics[str(key)] = float(value)

    speed = getattr(results, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            if _is_scalar_metric(value):
                metrics[f"speed/{key}"] = float(value)

    maps = getattr(results, "maps", None)
    if isinstance(maps, Iterable) and not isinstance(maps, (str, bytes, dict)):
        map_values = [float(value) for value in maps if _is_scalar_metric(value)]
        if map_values:
            metrics["metrics/per_class_mAP50-95_mean"] = sum(map_values) / len(map_values)

    per_class_map50, per_class_map50_95, _ = _collect_per_class_map_metrics(results)
    if per_class_map50:
        metrics["metrics/per_class_mAP50_mean"] = sum(per_class_map50) / len(per_class_map50)
    if per_class_map50_95:
        metrics["metrics/per_class_mAP50-95_mean"] = sum(per_class_map50_95) / len(per_class_map50_95)

    return metrics


def _resolve_training_output_dir(results: Any, *, project_dir: Path, run_name: str) -> Path:
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        return Path(save_dir).expanduser().resolve()
    return (project_dir / run_name).resolve()


def _write_results_csv_graphs(output_dir: Path) -> list[Path]:
    results_csv = output_dir / "results.csv"
    if not results_csv.exists():
        print_warning(f"Graph export skipped because {results_csv.name} was not found in {output_dir}.")
        return []

    plt = _load_pyplot()
    if plt is None:
        return []

    rows = _read_results_csv(results_csv)
    if not rows:
        print_warning(f"Graph export skipped because {results_csv.name} did not contain any rows.")
        return []

    x_values = _epoch_axis(rows)
    written: list[Path] = []

    chart_specs = [
        (
            "loss_curves.png",
            "Loss Curves",
            "Loss",
            [
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "train/obj_loss",
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "val/obj_loss",
            ],
        ),
        (
            "detection_metrics.png",
            "Detection Metrics",
            "Score",
            [
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/F1(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
        ),
        (
            "learning_rate.png",
            "Learning Rate",
            "LR",
            _columns_with_prefix(rows, "lr/"),
        ),
        (
            "speed_metrics.png",
            "Speed Metrics",
            "Milliseconds",
            _columns_with_prefix(rows, "speed/"),
        ),
    ]

    for file_name, title, y_label, columns in chart_specs:
        output_path = output_dir / file_name
        if _plot_training_series(
            plt,
            output_path=output_path,
            rows=rows,
            x_values=x_values,
            title=title,
            y_label=y_label,
            columns=columns,
        ):
            written.append(output_path)

    return written


def _write_per_class_map_artifacts(output_dir: Path, results: Any) -> list[Path]:
    map50_values, map50_95_values, labels = _collect_per_class_map_metrics(results)
    if not map50_values and not map50_95_values:
        return []

    written: list[Path] = []

    combined_csv_path = output_dir / "per_class_map.csv"
    with combined_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "map50", "map50_95"])
        for index, label in enumerate(labels):
            writer.writerow([
                label,
                _format_optional_csv_metric(_value_at(map50_values, index)),
                _format_optional_csv_metric(_value_at(map50_95_values, index)),
            ])
    written.append(combined_csv_path)

    if map50_values:
        map50_csv_path = output_dir / "per_class_map50.csv"
        _write_per_class_metric_csv(
            map50_csv_path,
            labels=labels,
            column_name="map50",
            values=map50_values,
        )
        written.append(map50_csv_path)

    if map50_95_values:
        map50_95_csv_path = output_dir / "per_class_map50_95.csv"
        _write_per_class_metric_csv(
            map50_95_csv_path,
            labels=labels,
            column_name="map50_95",
            values=map50_95_values,
        )
        written.append(map50_95_csv_path)

    plt = _load_pyplot()
    if plt is None:
        return written

    if map50_values:
        output_path = output_dir / "per_class_map50.png"
        if _plot_per_class_metric(
            plt,
            output_path=output_path,
            labels=labels,
            values=map50_values,
            title="Per-Class mAP@0.50",
            y_label="mAP@0.50",
            color="#2ca02c",
        ):
            written.append(output_path)

    if map50_95_values:
        output_path = output_dir / "per_class_map50_95.png"
        if _plot_per_class_metric(
            plt,
            output_path=output_path,
            labels=labels,
            values=map50_95_values,
            title="Per-Class mAP@0.50:0.95",
            y_label="mAP@0.50:0.95",
            color="#1f77b4",
        ):
            written.append(output_path)

    return written


def _collect_per_class_map_metrics(results: Any) -> tuple[list[float], list[float], list[str]]:
    box_metrics = getattr(results, "box", None)
    map50_values = _metric_sequence(getattr(box_metrics, "ap50", None))
    map50_95_values = _metric_sequence(getattr(box_metrics, "ap", None))

    all_ap = _metric_matrix(getattr(box_metrics, "all_ap", None))
    if all_ap:
        if not map50_values:
            map50_values = [
                row[0]
                for row in all_ap
                if row and _is_scalar_metric(row[0])
            ]
        if not map50_95_values:
            map50_95_values = [
                sum(row) / len(row)
                for row in all_ap
                if row and all(_is_scalar_metric(value) for value in row)
            ]

    if not map50_95_values:
        map50_95_values = _metric_sequence(getattr(results, "maps", None))

    value_count = max(len(map50_values), len(map50_95_values))
    class_indices = _metric_sequence(getattr(box_metrics, "ap_class_index", None))
    if len(class_indices) != value_count:
        class_indices = [float(index) for index in range(value_count)]

    labels = _per_class_labels(results, [int(index) for index in class_indices], value_count)
    return map50_values, map50_95_values, labels


def _write_per_class_metric_csv(
    output_path: Path,
    *,
    labels: list[str],
    column_name: str,
    values: list[float],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", column_name])
        for label, value in zip(labels, values):
            writer.writerow([label, f"{value:.6f}"])


def _plot_per_class_metric(
    plt,
    *,
    output_path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    y_label: str,
    color: str,
) -> bool:
    if not values:
        return False
    chart_labels = labels[: len(values)]
    figure_width = max(8, min(18, len(chart_labels) * 0.75))
    fig, ax = plt.subplots(figsize=(figure_width, 6))
    ax.bar(chart_labels, values, color=color)
    ax.set(title=title, xlabel="Class", ylabel=y_label, ylim=(0.0, 1.0))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _per_class_labels(results: Any, class_indices: list[int], value_count: int) -> list[str]:
    names = getattr(results, "names", None) or {}
    labels = []
    for position in range(value_count):
        class_index = class_indices[position] if position < len(class_indices) else position
        if isinstance(names, dict):
            labels.append(str(names.get(class_index, class_index)))
        elif isinstance(names, list) and class_index < len(names):
            labels.append(str(names[class_index]))
        else:
            labels.append(str(class_index))
    return labels


def _metric_sequence(value: Any) -> list[float]:
    if value is None or isinstance(value, (str, bytes, dict)):
        return []
    if _is_scalar_metric(value):
        return [float(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Iterable):
        return []
    return [float(item) for item in value if _is_scalar_metric(item)]


def _metric_matrix(value: Any) -> list[list[float]]:
    if value is None or isinstance(value, (str, bytes, dict)):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Iterable):
        return []

    rows: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes, dict)):
            row_values = [float(item) for item in row if _is_scalar_metric(item)]
            if row_values:
                rows.append(row_values)
    return rows


def _value_at(values: list[float], index: int) -> Optional[float]:
    if index >= len(values):
        return None
    return values[index]


def _format_optional_csv_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _read_results_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            normalized: dict[str, float] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                cleaned_key = key.strip()
                if value is None:
                    continue
                cleaned_value = value.strip()
                if not cleaned_value:
                    continue
                try:
                    normalized[cleaned_key] = float(cleaned_value)
                except ValueError:
                    continue
            if normalized:
                rows.append(normalized)
    return rows


def _epoch_axis(rows: list[dict[str, float]]) -> list[float]:
    if rows and "epoch" in rows[0]:
        return [row.get("epoch", float(index + 1)) for index, row in enumerate(rows)]
    return [float(index + 1) for index in range(len(rows))]


def _columns_with_prefix(rows: list[dict[str, float]], prefix: str) -> list[str]:
    columns: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith(prefix):
                columns.add(key)
    return sorted(columns)


def _plot_training_series(
    plt,
    *,
    output_path: Path,
    rows: list[dict[str, float]],
    x_values: list[float],
    title: str,
    y_label: str,
    columns: list[str],
) -> bool:
    available_columns = [column for column in columns if any(column in row for row in rows)]
    if not available_columns:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for column in available_columns:
        series = []
        x_series = []
        for x_value, row in zip(x_values, rows):
            if column not in row:
                continue
            x_series.append(x_value)
            series.append(row[column])
        if not series:
            continue
        ax.plot(x_series, series, label=_humanize_metric_label(column), linewidth=2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set(title=title, xlabel="Epoch", ylabel=y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print_warning(
            "matplotlib is not installed, so MLX could not generate training graphs in the run directory."
        )
        return None
    return plt


def _find_metric_keys(metrics: dict[str, float], patterns: tuple[str, ...]) -> list[str]:
    pattern_set = tuple(pattern.lower() for pattern in patterns)
    return sorted(
        key for key in metrics if any(pattern in key.lower() for pattern in pattern_set)
    )


def _humanize_metric_label(key: str) -> str:
    label = key.replace("(B)", "").replace("_", " ").replace("/", " / ")
    replacements = {
        "metrics / ": "",
        "val / ": "Val ",
        "train / ": "Train ",
        "speed / ": "Speed ",
        "map50-95": "mAP50-95",
        "map50": "mAP50",
        "auc": "AUC",
        "roc": "ROC",
        "dfl": "DFL",
    }
    lowered = label.lower()
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    words = []
    for token in lowered.split():
        if token.startswith("mAP") or token in {"AUC", "ROC", "DFL"}:
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _format_metric_value(value: float) -> str:
    return f"{value:.4f}"


def _is_scalar_metric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
