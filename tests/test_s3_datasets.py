from __future__ import annotations

import io
import json
import stat
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mlx.core.commands import CallbackWorkflowReporter
from mlx.core.datasets import (
    StageS3Dataset,
    TrainWithDatasetSource,
    extract_zip_safely,
    parse_s3_zip_uri,
    validate_dataset_source_options,
)
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.data import classification_dataset_root
from mlx.modes.image_classification.requests import ImageClassificationRequest
from mlx.modes.object_detection.data import object_detection_dataset_root
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest
from mlx.modes.segmentation.data import segmentation_dataset_root
from mlx.modes.segmentation.requests import SegmentationRequest
from mlx.modes.video_anomaly_detection.data import video_anomaly_dataset_root
from mlx.modes.video_anomaly_detection.requests import TrainVideoAnomalyRequest


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return output.getvalue()


class FakeS3:
    def __init__(self, content: bytes, *, etag: str = "etag-1") -> None:
        self.content = content
        self.etag = etag
        self.get_calls = 0

    def head_object(self, **kwargs):
        return {
            "ContentLength": len(self.content),
            "ETag": f'"{self.etag}"',
            "LastModified": datetime(2025, 1, 1, tzinfo=timezone.utc),
        }

    def get_object(self, **kwargs):
        self.get_calls += 1
        return {"Body": io.BytesIO(self.content)}


def test_parse_s3_zip_uri_normalizes_and_validates():
    location = parse_s3_zip_uri("s3://datasets/path/data%20set.zip")
    assert location.bucket == "datasets"
    assert location.key == "path/data%20set.zip"
    assert location.uri == "s3://datasets/path/data%20set.zip"
    with pytest.raises(MLXUserError, match=".zip"):
        parse_s3_zip_uri("s3://datasets/path/data.tar")
    with pytest.raises(MLXUserError, match="S3 object URI"):
        parse_s3_zip_uri("https://example.com/data.zip")


def test_stage_s3_dataset_caches_by_remote_identity(tmp_path):
    client = FakeS3(
        _zip_bytes(
            {
                "wrapper/train/cat/a.jpg": b"train",
                "wrapper/val/cat/b.jpg": b"val",
            }
        )
    )
    command = StageS3Dataset(
        "s3://datasets/classification.zip",
        root_resolver=classification_dataset_root,
        cache_dir=tmp_path / "cache",
        s3_client=client,
    )
    first = command.execute()
    second = command.execute()
    assert first.dataset_path == second.dataset_path
    assert first.dataset_path.name == "wrapper"
    assert client.get_calls == 1
    assert first.source["etag"] == "etag-1"
    assert len(first.source["sha256"]) == 64
    assert not (first.cache_path / "dataset.zip").exists()

    client.etag = "etag-2"
    third = command.execute()
    assert third.dataset_path != first.dataset_path
    assert client.get_calls == 2


def test_stage_s3_dataset_emits_one_download_progress_lifecycle(tmp_path):
    client = FakeS3(_zip_bytes({"train/a/x": b"x", "val/a/y": b"y"}))
    events = []
    StageS3Dataset(
        "s3://datasets/data.zip",
        root_resolver=classification_dataset_root,
        cache_dir=tmp_path / "cache",
        s3_client=client,
        reporter=CallbackWorkflowReporter(events.append),
    ).execute()

    download_events = [
        event
        for event in events
        if isinstance(event.payload, dict)
        and event.payload.get("event") == "dataset_download"
    ]
    statuses = [event.payload["status"] for event in download_events]
    assert statuses[0] == "start"
    assert statuses[-1] == "complete"
    assert statuses.count("start") == 1
    assert statuses.count("complete") == 1
    assert all(event.level == "progress" for event in download_events)
    assert download_events[-1].current == download_events[-1].total


def test_stage_s3_dataset_rebuilds_incomplete_cache(tmp_path):
    client = FakeS3(_zip_bytes({"train/a/x": b"x", "val/a/y": b"y"}))
    command = StageS3Dataset(
        "s3://datasets/data.zip",
        root_resolver=classification_dataset_root,
        cache_dir=tmp_path / "cache",
        s3_client=client,
    )
    result = command.execute()
    (result.cache_path / "mlx-dataset-cache.json").unlink()
    rebuilt = command.execute()
    assert rebuilt.dataset_path.is_dir()
    assert client.get_calls == 2


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "/absolute.txt", "C:/drive.txt", "safe\\..\\escape.txt"],
)
def test_safe_extractor_rejects_unsafe_paths(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    archive.write_bytes(_zip_bytes({name: b"bad"}))
    with pytest.raises(MLXUserError, match="unsafe path"):
        extract_zip_safely(archive, tmp_path / "out")


def test_safe_extractor_rejects_links_special_files_and_duplicates(tmp_path):
    symlink_archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_archive, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target")
    with pytest.raises(MLXUserError, match="symbolic link"):
        extract_zip_safely(symlink_archive, tmp_path / "symlink-out")

    special_archive = tmp_path / "special.zip"
    with zipfile.ZipFile(special_archive, "w") as archive:
        info = zipfile.ZipInfo("pipe")
        info.create_system = 3
        info.external_attr = (stat.S_IFIFO | 0o644) << 16
        archive.writestr(info, "")
    with pytest.raises(MLXUserError, match="special file"):
        extract_zip_safely(special_archive, tmp_path / "special-out")

    duplicate_archive = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate_archive, "w") as archive:
            archive.writestr("file.txt", "one")
            archive.writestr("file.txt", "two")
    with pytest.raises(MLXUserError, match="duplicate path"):
        extract_zip_safely(duplicate_archive, tmp_path / "duplicate-out")


def test_dataset_root_resolvers_support_wrapped_archives(tmp_path):
    detection = tmp_path / "detection" / "wrapper"
    detection.mkdir(parents=True)
    (detection / "data.yaml").write_text("names: [object]", encoding="utf-8")
    assert object_detection_dataset_root(tmp_path / "detection") == detection

    classification = tmp_path / "classification" / "wrapper"
    for child in ("train", "val"):
        (classification / child).mkdir(parents=True)
    assert classification_dataset_root(tmp_path / "classification") == classification

    segmentation = tmp_path / "segmentation" / "wrapper"
    for child in ("train/images", "train/masks", "val/images", "val/masks"):
        (segmentation / child).mkdir(parents=True)
    assert segmentation_dataset_root(tmp_path / "segmentation") == segmentation

    anomaly = tmp_path / "anomaly" / "wrapper"
    for child in ("train/normal", "val/normal"):
        (anomaly / child).mkdir(parents=True)
    assert video_anomaly_dataset_root(tmp_path / "anomaly") == anomaly


@dataclass(frozen=True)
class FakeRequest:
    dataset_path: str = "./tmp/dataset"
    dataset_s3_uri: str | None = None
    dataset_cache_dir: str = "~/.cache/mlx/datasets"
    output_path: str | None = None


class FakeTrainer:
    def __init__(self, request: FakeRequest, captured: list[FakeRequest]) -> None:
        self.request = request
        self.captured = captured

    def execute(self):
        self.captured.append(self.request)
        return "trained"


def test_train_with_dataset_source_injects_local_path_and_writes_provenance(tmp_path):
    client = FakeS3(_zip_bytes({"train/cat/a": b"a", "val/cat/b": b"b"}))
    captured: list[FakeRequest] = []
    output = tmp_path / "artifacts"
    request = FakeRequest(
        dataset_s3_uri="s3://datasets/data.zip",
        dataset_cache_dir=str(tmp_path / "cache"),
        output_path=str(output),
    )
    result = TrainWithDatasetSource(
        request,
        trainer_factory=lambda value: FakeTrainer(value, captured),
        root_resolver=classification_dataset_root,
        artifact_dir_resolver=lambda value: Path(str(value.output_path)),
        s3_client=client,
    ).execute()
    assert result == "trained"
    assert Path(captured[0].dataset_path).is_dir()
    source = json.loads((output / "dataset_source.json").read_text(encoding="utf-8"))
    assert source["uri"] == "s3://datasets/data.zip"
    assert "profile" not in source


def test_train_with_dataset_source_preserves_local_training_and_requires_s3_output(tmp_path):
    captured: list[FakeRequest] = []
    local = FakeRequest(dataset_path=str(tmp_path / "local"))
    assert TrainWithDatasetSource(
        local,
        trainer_factory=lambda value: FakeTrainer(value, captured),
        root_resolver=classification_dataset_root,
        artifact_dir_resolver=lambda value: tmp_path,
    ).execute() == "trained"
    assert captured == [local]

    remote = FakeRequest(dataset_s3_uri="s3://datasets/data.zip")
    with pytest.raises(MLXUserError, match="requires --output"):
        TrainWithDatasetSource(
            remote,
            trainer_factory=lambda value: FakeTrainer(value, captured),
            root_resolver=classification_dataset_root,
            artifact_dir_resolver=lambda value: tmp_path,
        ).execute()

    ambiguous = FakeRequest(
        dataset_path="./explicit-local",
        dataset_s3_uri="s3://datasets/data.zip",
        output_path=str(tmp_path / "output"),
    )
    with pytest.raises(MLXUserError, match="either a local dataset"):
        TrainWithDatasetSource(
            ambiguous,
            trainer_factory=lambda value: FakeTrainer(value, captured),
            root_resolver=classification_dataset_root,
            artifact_dir_resolver=lambda value: tmp_path,
        ).execute()


@pytest.mark.parametrize(
    ("request_type", "entries", "resolver"),
    [
        (
            TrainObjectDetectionRequest,
            {"wrapper/data.yaml": b"names: [object]"},
            object_detection_dataset_root,
        ),
        (
            ImageClassificationRequest,
            {"wrapper/train/cat/a.jpg": b"a", "wrapper/val/cat/b.jpg": b"b"},
            classification_dataset_root,
        ),
        (
            SegmentationRequest,
            {
                "wrapper/train/images/a.png": b"a",
                "wrapper/train/masks/a.png": b"a",
                "wrapper/val/images/b.png": b"b",
                "wrapper/val/masks/b.png": b"b",
            },
            segmentation_dataset_root,
        ),
        (
            TrainVideoAnomalyRequest,
            {
                "wrapper/train/normal/clip/001.jpg": b"a",
                "wrapper/val/normal/clip/001.jpg": b"b",
            },
            video_anomaly_dataset_root,
        ),
    ],
)
def test_all_training_request_types_accept_staged_dataset(
    tmp_path, request_type, entries, resolver
):
    captured = []
    output = tmp_path / request_type.__name__
    request = request_type(
        dataset_s3_uri="s3://datasets/data.zip",
        dataset_cache_dir=str(tmp_path / f"cache-{request_type.__name__}"),
        output_path=str(output),
    )
    result = TrainWithDatasetSource(
        request,
        trainer_factory=lambda value: FakeTrainer(value, captured),
        root_resolver=resolver,
        artifact_dir_resolver=lambda value: output,
        s3_client=FakeS3(_zip_bytes(entries)),
    ).execute()
    assert result == "trained"
    assert Path(captured[0].dataset_path).name == "wrapper"
    assert (output / "dataset_source.json").is_file()


def test_dataset_source_cli_validation_rejects_ambiguity_and_non_training_actions():
    with pytest.raises(MLXUserError, match="either --dataset"):
        validate_dataset_source_options(
            {
                "dataset_s3_uri": "s3://bucket/data.zip",
                "dataset_path": "./data",
                "_explicit_options": {"dataset_s3_uri", "dataset_path"},
            },
            action="train",
        )
    with pytest.raises(MLXUserError, match="only for training"):
        validate_dataset_source_options(
            {"dataset_s3_uri": "s3://bucket/data.zip"},
            action="benchmark",
        )
    validate_dataset_source_options(
        {
            "dataset_s3_uri": "s3://bucket/data.zip",
            "dataset_path": "./tmp/dataset",
            "_explicit_options": {"dataset_s3_uri"},
        },
        action="train",
    )
