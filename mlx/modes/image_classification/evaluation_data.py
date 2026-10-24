"""Image-directory and pair construction boundary for classification benchmarks."""
from __future__ import annotations

import random
from pathlib import Path

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.data import (
    load_image_tensor, load_standard_classification_directory, resolve_evaluation_dir,
)
from mlx.modes.image_classification.evaluation_records import OneShotPairRecord

def build_one_shot_benchmark_pairs(
    dataset_path: Path,
    *,
    pairs_per_class: int,
    random_seed: int | None,
    reporter: WorkflowReporter | None = None,
) -> list[OneShotPairRecord]:
    if pairs_per_class <= 0:
        raise MLXUserError("--num-pairs must be greater than zero for one-shot benchmarking.")

    label_to_images = _index_one_shot_images(dataset_path, reporter=reporter)
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


def _index_one_shot_images(
    dataset_path: Path,
    *,
    reporter: WorkflowReporter | None = None,
) -> dict[str, list[Path]]:
    reporter = reporter or NullWorkflowReporter()
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
            emit(
                reporter,
                "warning",
                f"Skipping label '{label_dir.name}' for one-shot benchmark because it has fewer than two images."
            )
    return label_to_images


class ClassificationEvaluationData:
    """Filesystem-backed image loading for standard and pairwise evaluation."""

    resolve_root = staticmethod(resolve_evaluation_dir)
    load_standard = staticmethod(load_standard_classification_directory)
    load_image = staticmethod(load_image_tensor)

    def build_pairs(self, *args, **kwargs):
        return build_one_shot_benchmark_pairs(*args, **kwargs)

    def index_images(self, *args, **kwargs):
        return _index_one_shot_images(*args, **kwargs)
