from __future__ import annotations

import pytest

from mlx.core.commands import WorkflowEvent
from mlx.core.presentation import (
    RichDatasetDownloadProgress,
    RichInfrastructureEventRenderer,
    RichTrainingMetricsRenderer,
    TrainingMetricSpec,
)
from mlx.modes.image_classification.presentation import RichImageClassificationReporter
from mlx.modes.object_detection.presentation import RichWorkflowReporter
from mlx.modes.segmentation.presentation import RichSegmentationReporter
from mlx.modes.video_anomaly_detection.presentation import (
    RichVideoAnomalyReporter,
    RichVideoAnomalyTrainingProgress,
)


class FakeProgress:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.tasks: list[tuple[str, dict[str, object]]] = []
        self.updates: list[tuple[int, dict[str, object]]] = []

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def add_task(self, description: str, **fields: object) -> int:
        self.tasks.append((description, fields))
        return 7

    def update(self, task_id: int, **fields: object) -> None:
        self.updates.append((task_id, fields))


def _download_event(current: int, total: int, status: str) -> WorkflowEvent:
    return WorkflowEvent(
        level="progress",
        message="[cyan]Downloading dataset.zip[/cyan]",
        current=current,
        total=total,
        payload={"event": "dataset_download", "status": status},
    )


def test_dataset_download_progress_reuses_one_rich_task_until_completion() -> None:
    progress = FakeProgress()
    renderer = RichDatasetDownloadProgress(progress_factory=lambda: progress)

    assert renderer.handle(_download_event(0, 100, "start")) is True
    assert renderer.handle(_download_event(40, 100, "update")) is True
    assert renderer.handle(_download_event(100, 100, "update")) is True
    assert renderer.handle(_download_event(100, 100, "complete")) is True

    assert progress.started == 1
    assert progress.stopped == 1
    assert len(progress.tasks) == 1
    assert progress.tasks[0][1] == {"total": 100, "completed": 0}
    assert [fields["completed"] for _, fields in progress.updates] == [0, 40, 100, 100]
    assert progress.updates[-1][1]["description"] == "[green]S3 dataset downloaded[/green]"


def test_dataset_download_progress_ignores_unrelated_events() -> None:
    progress = FakeProgress()
    renderer = RichDatasetDownloadProgress(progress_factory=lambda: progress)

    assert renderer.handle(WorkflowEvent(level="info", message="Training")) is False
    assert progress.tasks == []


def test_dataset_download_progress_closes_the_live_line_on_failure() -> None:
    progress = FakeProgress()
    renderer = RichDatasetDownloadProgress(progress_factory=lambda: progress)

    renderer.handle(_download_event(0, 100, "start"))
    renderer.handle(_download_event(20, 100, "failed"))

    assert progress.started == 1
    assert progress.stopped == 1
    assert progress.updates[-1][1]["description"] == (
        "[red]S3 dataset download interrupted[/red]"
    )


def test_video_anomaly_training_progress_updates_one_task_per_phase() -> None:
    progress = FakeProgress()
    renderer = RichVideoAnomalyTrainingProgress(progress_factory=lambda: progress)

    def event(current: int, status: str) -> WorkflowEvent:
        return WorkflowEvent(
            level="progress",
            message="Epoch 1/2: training normal clips",
            current=current,
            total=3,
            payload={
                "event": "video_anomaly_training_progress",
                "status": status,
                "phase": "train",
                "epoch": 1,
                "epochs": 2,
            },
        )

    assert renderer.handle(event(0, "start")) is True
    assert renderer.handle(event(1, "update")) is True
    assert renderer.handle(event(3, "complete")) is True

    assert len(progress.tasks) == 1
    assert progress.started == 1
    assert progress.stopped == 1
    assert [fields["completed"] for _, fields in progress.updates] == [0, 1, 3]


def test_training_metrics_render_persistent_direction_aware_rows() -> None:
    rendered = []
    renderer = RichTrainingMetricsRenderer(
        (
            TrainingMetricSpec("loss", "loss", higher_is_better=False),
            TrainingMetricSpec("score", "score", higher_is_better=True),
            TrainingMetricSpec("lr", "lr", higher_is_better=None, precision=3),
        ),
        highlights=(("is_best_val_loss", "best loss"),),
        render=rendered.append,
    )

    renderer.handle(
        WorkflowEvent(
            level="progress",
            message="epoch",
            current=1,
            total=3,
            payload={
                "event": "training_epoch",
                "metrics": {"loss": 1.0, "score": 0.5, "lr": 0.01},
            },
        )
    )
    renderer.handle(
        WorkflowEvent(
            level="progress",
            message="epoch",
            current=2,
            total=3,
            payload={
                "event": "training_epoch",
                "metrics": {"loss": 0.8, "score": 0.4, "lr": 0.005},
                "is_best_val_loss": True,
            },
        )
    )

    assert rendered[0].plain == (
        "Epoch 1/3  loss 1.0000 —  score 0.5000 —  lr 0.010 —"
    )
    assert rendered[1].plain == (
        "Epoch 2/3  loss 0.8000 ↓0.2000  score 0.4000 ↓0.1000  "
        "lr 0.005 ↓0.005  best loss"
    )
    loss_delta = rendered[1].plain.index("↓0.2000")
    score_delta = rendered[1].plain.index("↓0.1000")
    lr_delta = rendered[1].plain.index("↓0.005")
    assert any(
        span.start <= loss_delta < span.end and span.style == "green"
        for span in rendered[1].spans
    )
    assert any(
        span.start <= score_delta < span.end and span.style == "red"
        for span in rendered[1].spans
    )
    assert any(
        span.start <= lr_delta < span.end and span.style == "dim"
        for span in rendered[1].spans
    )


def test_training_metrics_use_resume_history_and_reset_for_a_new_run() -> None:
    rendered = []
    renderer = RichTrainingMetricsRenderer(
        (TrainingMetricSpec("loss", "loss", higher_is_better=False),),
        render=rendered.append,
    )

    renderer.handle(
        WorkflowEvent(
            level="progress",
            message="resumed epoch",
            current=4,
            total=5,
            payload={
                "event": "training_epoch",
                "metrics": {"loss": 0.7, "ignored": float("nan")},
                "previous_metrics": {"loss": 0.9},
            },
        )
    )
    renderer.handle(
        WorkflowEvent(
            level="progress",
            message="new run",
            current=1,
            total=2,
            payload={"event": "training_epoch", "metrics": {"loss": 1.2}},
        )
    )

    assert rendered[0].plain.endswith("loss 0.7000 ↓0.2000")
    assert rendered[1].plain.endswith("loss 1.2000 —")


def test_segmentation_reporter_renders_one_epoch_row_without_checkpoint_lines(
    monkeypatch,
) -> None:
    rendered = []
    monkeypatch.setattr("mlx.core.presentation.console.print", rendered.append)
    reporter = RichSegmentationReporter()

    reporter.emit(
        WorkflowEvent(
            level="progress",
            message="completed epoch",
            current=1,
            total=2,
            payload={
                "event": "segmentation_epoch",
                "metrics": {
                    "train_loss": 1.0,
                    "val_loss": 0.9,
                    "mean_foreground_dice": 0.5,
                    "mean_iou": 0.4,
                    "pixel_accuracy": 0.7,
                    "learning_rate": 0.001,
                    "epoch_seconds": 2.0,
                },
                "checkpoints": ["last → /tmp/model.last.pth"],
                "is_best_val_loss": True,
            },
        )
    )

    assert len(rendered) == 1
    assert rendered[0].plain.startswith("Epoch 1/2  train 1.0000 —")
    assert "model.last.pth" not in rendered[0].plain


@pytest.mark.parametrize(
    "reporter_type",
    (
        RichImageClassificationReporter,
        RichWorkflowReporter,
        RichSegmentationReporter,
        RichVideoAnomalyReporter,
    ),
)
def test_train_capable_rich_reporters_consume_shared_download_progress(reporter_type) -> None:
    progress = FakeProgress()
    reporter = reporter_type(
        infrastructure_events=RichInfrastructureEventRenderer(
            dataset_download_progress=RichDatasetDownloadProgress(
                progress_factory=lambda: progress
            )
        )
    )

    reporter.emit(_download_event(0, 10, "start"))
    reporter.emit(_download_event(10, 10, "complete"))

    assert len(progress.tasks) == 1
    assert progress.stopped == 1
