from __future__ import annotations

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
