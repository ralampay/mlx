"""Dataset staging shared by SageMaker training and evaluation containers."""

from pathlib import Path
import shutil

from mlx.core.datasets import extract_zip_safely
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.data import object_detection_dataset_root


def extract_sagemaker_dataset(input_dir: Path, dataset_dir: Path, *, max_uncompressed_bytes: int | None) -> Path:
    archives = sorted(path for path in input_dir.rglob("*.zip") if path.is_file())
    if len(archives) != 1:
        raise MLXUserError(
            f"Expected exactly one dataset ZIP in {input_dir}; found {len(archives)}."
        )
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)
    extract_zip_safely(archives[0], dataset_dir, max_uncompressed_bytes=max_uncompressed_bytes)
    return object_detection_dataset_root(dataset_dir)
