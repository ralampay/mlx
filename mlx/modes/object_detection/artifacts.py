from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlx.core.exceptions import MLXUserError


def find_latest_checkpoint(project_dir: Path, *, pattern: str) -> Optional[Path]:
    matches = [path for path in project_dir.rglob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path))).resolve()


def find_existing_checkpoint(
    *,
    project_dir: Path,
    run_dir: Optional[Path],
    file_name: str,
) -> Optional[Path]:
    if run_dir is not None:
        for candidate in (run_dir / "weights" / file_name, run_dir / file_name):
            if candidate.exists():
                return candidate.resolve()
    return find_latest_checkpoint(project_dir, pattern=file_name)


def detect_existing_training_artifacts(
    *,
    project_dir: Path,
    run_name: Optional[str],
    explicit_weights: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    if explicit_weights is not None or not project_dir.exists():
        return None, None

    run_dir = project_dir / run_name if run_name else None
    resume_checkpoint = find_existing_checkpoint(
        project_dir=project_dir,
        run_dir=run_dir,
        file_name="last.pt",
    )
    if resume_checkpoint is not None:
        return resume_checkpoint, None

    warm_start_weights = find_existing_checkpoint(
        project_dir=project_dir,
        run_dir=run_dir,
        file_name="best.pt",
    )
    if warm_start_weights is not None:
        return None, warm_start_weights

    return None, find_latest_checkpoint(project_dir, pattern="*.pt")


def resolve_onnx_output_target(config: dict, resolved_weights: Path) -> Path:
    output_path = config.get("output_path")
    default_target = resolved_weights.with_suffix(".onnx")
    if not output_path:
        return default_target.resolve()

    candidate = Path(output_path).expanduser()
    if candidate.exists() and candidate.is_dir():
        return (candidate / default_target.name).resolve()
    if candidate.suffix.lower() == ".onnx":
        return candidate.resolve()
    if candidate.exists():
        return (candidate / default_target.name).resolve()
    if candidate.suffix:
        raise MLXUserError(
            "For ONNX export, --output must be a directory or a path ending in .onnx."
        )
    return (candidate / default_target.name).resolve()
