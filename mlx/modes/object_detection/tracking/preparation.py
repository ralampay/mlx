"""Copy locally available tracking datasets into portable benchmark sequences."""
from __future__ import annotations

import configparser
import json
import shutil
import tempfile
from pathlib import Path

from mlx.core.artifacts import write_json_atomic
from mlx.core.commands import NullWorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.core.streaming import OpenCVFrameSource
from mlx.modes.object_detection.tracking.data import load_tracking_sequence, numeric_images
from mlx.modes.object_detection.tracking.mot import MOTRecord


class MOTSequenceAdapter:
    """Normalize image-based DanceTrack and PETS sources without modifying them."""

    def describe(self, source: Path, dataset: str, split: str):
        import cv2
        images = numeric_images(source / "img1")
        first = cv2.imread(str(images[0]))
        if first is None:
            raise MLXUserError(f"Unreadable first image: {images[0]}")
        height, width = first.shape[:2]
        fps, provenance = 30.0, "fallback: 30 FPS playback"
        info_path = source / "seqinfo.ini"
        if info_path.exists():
            info = configparser.ConfigParser()
            info.read(info_path)
            section = info["Sequence"]
            fps, provenance = section.getfloat("frameRate"), "seqinfo.ini"
            if section.getint("seqLength") != len(images) or section.getint("imWidth") != width or section.getint("imHeight") != height:
                raise MLXUserError(f"Frame count or dimensions disagree with seqinfo.ini: {source}")
        raw_rows = [line.split(",") for line in (source / "gt/gt.txt").read_text().splitlines() if line.strip()]
        if any(len(row) not in (9, 10) for row in raw_rows):
            raise MLXUserError(f"Expected 9- or 10-column MOT annotations: {source}")
        raw_rows = [r for r in raw_rows if float(r[6]) > 0 and (len(r) != 9 or int(r[7]) == 1)]
        identities = {identity: index + 1 for index, identity in enumerate(sorted({int(r[1]) for r in raw_rows}))}
        first_number = int(images[0].stem)
        records = [MOTRecord(int(r[0]) - first_number + 1, identities[int(r[1])],
                             *map(float, r[2:6]), 1.0) for r in raw_rows]
        manifest = {
            "version": 1, "dataset": dataset, "split": split, "name": source.name,
            "kind": "images", "media": "img1", "ground_truth": "gt/gt.txt",
            "fps": fps, "width": width, "height": height,
            "frame_indices": list(range(len(images))),
            "provenance": {"source": str(source), "fps": provenance,
                           "original_first_frame": first_number, "track_id_map": identities,
                           "policy": "Positive-confidence person rows; MLX MOT metrics."},
        }
        copies = [(p, Path("img1") / p.name) for p in images]
        copies.append((source / "gt/gt.txt", Path("original/gt.txt")))
        if info_path.exists():
            copies.append((info_path, Path("original/seqinfo.ini")))
        return manifest, records, copies


class PersonPathSequenceAdapter:
    """Visible person boxes evaluated only at explicitly annotated source frames."""

    def describe(self, video: Path, annotation: Path, split: str):
        data = json.loads(annotation.read_text())
        entities = data["entities"]
        indices = sorted({int(e["blob"]["frame_idx"]) for e in entities})
        selected = [e for e in entities if e.get("labels", {}).get("person") and not e.get("labels", {}).get("crowd")]
        identities = {identity: i + 1 for i, identity in enumerate(sorted({int(e["id"]) for e in selected}))}
        frame_map = {index: i + 1 for i, index in enumerate(indices)}
        records = [MOTRecord(frame_map[int(e["blob"]["frame_idx"])], identities[int(e["id"])],
                             *map(float, e["bb"]), 1.0) for e in selected]
        source = OpenCVFrameSource(source="video", file_path=str(video))
        try:
            info = source.metadata()
            ok, frame = source.read()
            if not ok or not indices or info.frame_count is None or indices[-1] >= info.frame_count:
                raise MLXUserError(f"Video is incomplete or does not cover annotation frames: {video}")
            height, width = frame.shape[:2]
        finally:
            source.release()
        resolution = data["metadata"]["resolution"]
        if width != int(resolution["width"]) or height != int(resolution["height"]):
            raise MLXUserError(f"Video dimensions disagree with annotations: {video}")
        manifest = {
            "version": 1, "dataset": "PersonPath22", "split": split, "name": video.stem,
            "kind": "video", "media": "source.mp4", "ground_truth": "gt/gt.txt",
            "fps": 5.0, "width": width, "height": height, "frame_indices": indices,
            "provenance": {"source": str(video), "annotation": str(annotation),
                           "source_fps": info.fps, "track_id_map": identities,
                           "policy": "Visible individually identified people, including occluded/background people; crowd-only regions excluded. Only explicitly annotated source frames evaluated; no official ignore-region suppression. 5 FPS presentation."},
        }
        return manifest, records, [(video, Path("source.mp4")), (annotation, Path("original/annotations.json"))]


class PrepareTrackingBenchmarks:
    """Stage independent copies and publish only validated sequence directories."""

    def __init__(self, source_root: Path, output_root: Path, *, reporter=None,
                 mot_adapter=None, personpath_adapter=None):
        self.source_root = Path(source_root).expanduser()
        self.output_root = Path(output_root).expanduser()
        self.reporter = reporter or NullWorkflowReporter()
        self.mot_adapter = mot_adapter or MOTSequenceAdapter()
        self.personpath_adapter = personpath_adapter or PersonPathSequenceAdapter()

    def execute(self):
        if not self.source_root.is_dir():
            raise MLXUserError(f"Tracking dataset source directory not found: {self.source_root}")
        if self.output_root.exists():
            raise MLXUserError(f"Preparation output already exists: {self.output_root}. Choose a new output directory.")
        self.output_root.mkdir(parents=True)
        report = {"version": 1, "ready": [], "unavailable": [], "evaluation": "MLX MOT metrics; not official dataset protocols"}
        try:
            for label, describe in self._candidates(report):
                emit(self.reporter, "info", f"Preparing {label}.")
                try:
                    manifest, records, copies = describe()
                    relative = self._publish(manifest, records, copies)
                    report["ready"].append(str(relative))
                except (MLXUserError, ValueError, KeyError, OSError, configparser.Error) as exc:
                    report["unavailable"].append({"sequence": label, "reason": str(exc)})
                    emit(self.reporter, "warning", f"Skipped {label}: {exc}")
                write_json_atomic(self.output_root / "preparation.json", report)
        finally:
            write_json_atomic(self.output_root / "preparation.json", report)
        (self.output_root / "README.md").write_text(
            "# Prepared tracking benchmarks\n\n"
            "Independent copies of complete locally available labeled sequences. See preparation.json for omissions.\n\n"
            "Run: `python -m mlx --mode track --action benchmark --dataset PATH_TO_THIS_DIRECTORY "
            "--model-path MODEL --track-class-id 0 --output RESULTS`\n\n"
            "Use `--real-time-results False` for direct evaluation, or `--no-display` to save videos without a window. "
            "Use --track-class-id appropriate to your detector (0 is person for COCO models). "
            "Results use MLX MOT metrics, not official dataset protocols. "
            "PersonPath22 uses visible person boxes on explicitly annotated frames only; crowd-only regions "
            "are omitted without ignore-region suppression. Manifests preserve conversion and frame mappings.\n"
        )
        return report

    def _candidates(self, report):
        root = self.source_root
        dance = root / "DanceTrack/dataset"
        found = False
        for directory in sorted(dance.glob("*/*")):
            if not directory.is_dir():
                continue
            split = directory.parent.name
            if not (directory / "gt/gt.txt").exists():
                report["unavailable"].append({"sequence": f"DanceTrack/{split}/{directory.name}", "reason": "No local ground truth."})
                continue
            found = True
            canonical_split = "train" if split.startswith("train") else split
            yield f"DanceTrack/{split}/{directory.name}", lambda p=directory, s=canonical_split: self.mot_adapter.describe(p, "DanceTrack", s)
        if not found:
            report["unavailable"].append({"dataset": "DanceTrack", "reason": "No labeled sequences found."})
        pets = root / "PETS2009/motchallenge/2DMOT2015/train/PETS09-S2L1"
        if pets.is_dir():
            yield "PETS2009/train/PETS09-S2L1", lambda: self.mot_adapter.describe(pets, "PETS2009", "train")
        else:
            report["unavailable"].append({"dataset": "PETS2009", "reason": "PETS09-S2L1 frames unavailable."})
        report["unavailable"].append({"dataset": "PETS2009", "reason": "Other XML annotation sequences have no matching local image sequences."})
        person = root / "PersonPath22/dataset/personpath22"
        splits = person / "annotation/splits.json"
        if splits.is_file():
            for split, names in sorted(json.loads(splits.read_text()).items()):
                for name in sorted(names):
                    video = person / "raw_data" / name
                    annotation = person / "annotation/anno_visible_2022" / f"{name}.json"
                    if not video.is_file() or not annotation.is_file():
                        report["unavailable"].append({"sequence": f"PersonPath22/{split}/{name}", "reason": "Video or visible annotations missing."})
                        continue
                    yield f"PersonPath22/{split}/{name}", lambda v=video, a=annotation, s=split: self.personpath_adapter.describe(v, a, s)
        else:
            report["unavailable"].append({"dataset": "PersonPath22", "reason": "Split annotations unavailable."})
        report["unavailable"].append({"dataset": "HiEve", "reason": "No authorized local media/annotations; see original HiEve/ACCESS_REQUIRED.md."})

    def _publish(self, manifest, records, copies):
        relative = Path(manifest["dataset"]) / manifest["split"] / manifest["name"]
        target = self.output_root / relative
        if target.exists():
            raise MLXUserError(f"Duplicate preparation target: {target}")
        if not records:
            raise MLXUserError(f"No evaluable person annotations in {relative}.")
        observed = set()
        for record in records:
            key = (record.frame_id, record.track_id)
            if key in observed:
                raise MLXUserError(
                    f"Conflicting annotations in {relative}: track {record.track_id} "
                    f"appears more than once in evaluation frame {record.frame_id}. "
                    "Repair duplicate source identities before preparing this sequence."
                )
            observed.add(key)
            if record.frame_id > len(manifest["frame_indices"]):
                raise MLXUserError(f"Annotations extend beyond available frames in {relative}.")
        required = sum(source.stat().st_size for source, _ in copies)
        if shutil.disk_usage(self.output_root).free < required + 1024**3:
            raise MLXUserError(f"Insufficient space to copy {relative}: requires {required} bytes plus working space.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as temporary:
            stage = Path(temporary) / "sequence"
            stage.mkdir()
            for source, destination in copies:
                destination = stage / destination
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if destination.stat().st_size != source.stat().st_size:
                    raise MLXUserError(f"Incomplete dataset copy: {source}")
            (stage / "gt").mkdir(exist_ok=True)
            (stage / "gt/gt.txt").write_text("".join(r.to_line() + "\n" for r in sorted(records, key=lambda r: (r.frame_id, r.track_id))))
            write_json_atomic(stage / "sequence.json", manifest)
            load_tracking_sequence(stage / "sequence.json")
            stage.rename(target)
        return relative
