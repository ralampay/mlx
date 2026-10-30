"""Results shared by segmentation training and experiment orchestration."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentationTrainingResult:
    checkpoint_path: Path
    best_validation_loss: float
    output_dir: Path
    last_checkpoint_path: Path | None = None
    best_dice_checkpoint_path: Path | None = None
