# Tracking by Detection

MLX exposes provider-neutral multi-object tracking through `--mode track`. A video
frame is detected by the selected object-detection provider, normalized to MLX
`TrackingDetection` values, and then passed to one stateful tracking class.

```text
video → Ultralytics or LibreYOLO adapter → normalized detections
      → selected tracker → tracks.jsonl + tracks.txt → optional MOT benchmark
```

The tracker never imports or invokes YOLO. This makes the same tracker compatible
with every detector that implements MLX's `DetectionAdapter` contract.

## Installation

The repository `requirements.txt` includes tracking dependencies. Package consumers
can combine the tracking extra with either detection provider:

```bash
python -m pip install ".[tracking,object-detection-ultralytics]"
python -m pip install ".[tracking,object-detection-libreyolo]"
```

## CLI Usage

List the built-in tracker aliases:

```bash
python -m mlx --mode track --action ls-trackers
```

Run ByteTrack with an Ultralytics checkpoint:

```bash
python -m mlx --mode track --action run \
    --provider ultralytics \
    --tracker bytetrack \
    --model yolo26 \
    --model-path ./runs/yolo/weights/best.pt \
    --file-path ~/Desktop/video.mp4 \
    --confidence 0.1 \
    --track-class-id 0 \
    --output ./tracking/bytetrack
```

Run SORT with a LibreYOLO checkpoint and benchmark it:

```bash
python -m mlx --mode track --action run \
    --provider libreyolo \
    --tracker sort \
    --model-path ./runs/libreyolo/weights/best.onnx \
    --file-path ~/Desktop/video.mp4 \
    --ground-truth ~/Desktop/gt.txt \
    --track-class-id 0 \
    --output ./tracking/sort
```

`--track-class-id` is repeatable and defaults to all detector classes. MOT ground
truth does not contain a usable detector class, so select the matching detector
class when benchmarking class-specific sequences. COCO person is class `0`.
For ByteTrack, set the detector `--confidence` floor at or below the tracker's
`low_threshold`; otherwise the provider discards low-confidence boxes before the
second association pass can use them.

The output directory contains:

```text
tracking/sort/
├── tracks.jsonl  # lossless class-aware MLX tracking records
├── tracks.txt    # strict 10-column MOTChallenge predictions
├── metrics.json  # only when --ground-truth is supplied
├── replay.json   # portable projected boxes, classes, metadata, GT, and metrics
└── replay.html   # self-contained interactive 2D replay
```

Pass `--overwrite` to replace existing artifacts.

## Live Tracking Visualization

The tracking CLI opens an OpenCV playback window by default. Every currently
observed track is drawn with a stable per-ID color and an overlay containing:

- the bounding box;
- the track ID;
- detector class label, or class ID when no label is available;
- confidence score;
- lifecycle status (`tentative` or `confirmed`).

The top-left summary shows the 1-based frame number and visible-track count.
Temporarily lost tracks remain available to the algorithm for recovery but are not
drawn because they were not observed in the displayed frame.

Press `q` or `Esc` to stop playback. MLX finalizes `tracks.jsonl` and `tracks.txt`
for the frames that were processed. If ground truth was supplied, benchmarking is
skipped after an early stop because the partial prediction file does not cover the
complete video.

Use `--no-display` on a headless machine or when processing without an interactive
window:

```bash
python -m mlx --mode track --action run \
    --tracker bytetrack \
    --model-path ./detector.onnx \
    --file-path ./video.mp4 \
    --output ./tracking/bytetrack \
    --no-display
```

Visualization does not change the MOT output format or tracker contract. Custom
trackers automatically receive the same display behavior when they return valid
`TrackingFrameResult` values.

## Portable 2D Replay Without the Video

Every completed tracking run exports `replay.json` and `replay.html`. Open
`replay.html` directly in a browser to inspect the run after the model and source
video have been removed. The HTML embeds its data and assets, so it requires no web
server, CDN, Python environment, or network connection.

The player provides:

- play/pause, frame scrubbing, keyboard stepping, and playback speed;
- stable per-ID prediction colors, class labels, confidence, and two-second
  center-point motion trails;
- ground-truth overlays when `--ground-truth` was used;
- independent toggles for predictions, ground truth, and trails;
- benchmark metric cards and per-frame prediction/GT counts.

`replay.json` contains the same information separately for notebooks, dashboards,
projection tools, or another renderer. It records the pixel coordinate convention,
canvas dimensions, FPS, frame count, run settings, predictions, optional ground
truth, and optional metrics. It deliberately stores no provider-specific result
objects and no absolute source-video path.

Use [notebooks/tracking_replay.ipynb](./notebooks/tracking_replay.ipynb) for a
Matplotlib trajectory projection and notebook-native 2D animation. Change only its
`RESULT_DIR`; neither visualization needs the source video.

## Class-Aware Output and MOT Extraction

Each tracking run writes the same confirmed, currently observed tracks to two
separate artifacts. `tracks.jsonl` is the lossless MLX interchange format and keeps
`class_id`, `label`, `xyxy` coordinates, confidence, frame ID, and track ID.
`tracks.txt` is the class-agnostic, exactly 10-column MOTChallenge projection used
by the benchmark command and standard MOT tools. MLX does not put an unofficial
eleventh class column in the MOT file.

In short, class information is stored in `tracks.jsonl` and copied into prediction
rows in `replay.json`; it is intentionally absent from `tracks.txt` because the
standard MOT prediction format has no class column. Ground-truth replay rows also
have `class_id: null` and `label: null` when the input is standard 10-column MOT.

One `tracks.jsonl` line looks like this:

```json
{"schema_version":"mlx.tracking.record/v1","frame_id":1,"track_id":1,"class_id":0,"label":"person","bounding_box":{"x1":88.0,"y1":99.0,"x2":149.08,"y2":317.56},"confidence":0.92}
```

The class fields come from the `TrackResult` returned by the selected tracking
class. Built-in trackers preserve the normalized detector class and label. Custom
trackers must do the same if those values should appear in logs and replay output.

You can recreate a standard MOT file from any MLX class-aware result without the
video or detector. Omit `--track-class-id` to export all classes, or repeat it to
select classes before class-agnostic MOT evaluation:

```bash
python -m mlx --mode track --action export-mot \
    --tracking-jsonl ./tracking/run/tracks.jsonl \
    --track-class-id 0 \
    --output ./tracking/person-mot
```

The command validates the JSONL schema and rejects duplicate frame/track pairs. It
writes `./tracking/person-mot/tracks.txt`; pass `--overwrite` if that output exists.
With no class filter, the extracted file is the same MOT projection produced during
the original run.

`--track-class-id` has two related but distinct meanings:

- with `--action run`, it filters detections before tracking and can therefore
  affect association and assigned track IDs;
- with `--action export-mot`, it filters completed `tracks.jsonl` records and never
  reruns the detector or tracker.

## MOT Input and Output

For the complete in-memory `TrackingDetection`/`TrackingFrameResult` contracts,
class-aware JSONL, MOT, and replay JSON schemas, coordinate conversion rules,
column definitions, and export eligibility, see
[Expected Tracking Formats](./CUSTOM_TRACKING.md#expected-tracking-formats).

Ground truth and predictions are headerless, comma-separated MOTChallenge rows:

```text
frame,id,left,top,width,height,confidence,world_x,world_y,world_z
```

Frame and track IDs are 1-based. MLX writes `-1,-1,-1` for unavailable world
coordinates. Only confirmed tracks observed in the current frame are written;
tentative and temporarily lost internal tracks are not exported.

Benchmarking uses class-agnostic framewise matching at `--benchmark-iou 0.5` by
default and writes MOTA, mean matched IoU (MOTP), IDF1, precision, recall, matches,
false positives, misses, and ID switches. Ground-truth rows with a non-positive
confidence/mark field are ignored.

The normal run benchmarks its generated `tracks.txt` automatically when
`--ground-truth` is supplied:

```bash
python -m mlx --mode track --action run \
    --provider libreyolo \
    --tracker bytetrack \
    --model-path ./best.pt \
    --file-path ./video.mp4 \
    --ground-truth ./gt.txt \
    --track-class-id 0 \
    --output ./tracking/run \
    --no-display
```

To evaluate a class-filtered MOT file extracted after a run, use the reusable
benchmark command from Python or pass the resulting `tracks.txt` to another standard
MOT evaluator:

```python
from pathlib import Path

from mlx.modes.object_detection.tracking.evaluation import BenchmarkMOTTracking

result = BenchmarkMOTTracking(
    ground_truth_path=Path("./gt.txt"),
    predictions_path=Path("./tracking/person-mot/tracks.txt"),
    output_path=Path("./tracking/person-mot/metrics.json"),
    iou_threshold=0.5,
    overwrite=True,
).execute()
print(result.mota, result.idf1)
```

## Built-in Examples

`sort` is a compact class-aware SORT implementation using constant-velocity Kalman
motion, IoU/Hungarian association, tentative confirmation, and bounded expiry.

`bytetrack` adds ByteTrack's two-stage association: high-confidence detections are
matched first, then unmatched tracks may be recovered by low-confidence detections.
Low-confidence detections do not create new tracks.

Both trackers accept constructor options from a JSON object:

```json
{
  "iou_threshold": 0.3,
  "max_age": 30,
  "min_hits": 1
}
```

```bash
python -m mlx --mode track --tracker sort \
    --tracker-config ./sort.json \
    --model-path ./detector.onnx \
    --file-path ./video.mp4 \
    --output ./tracking/sort
```

ByteTrack additionally accepts `high_threshold`, `low_threshold`, and
`new_track_threshold`.

## Writing Your Own Tracker

For a source-level walkthrough covering new modules, built-in registration,
constructor configuration, tests, and modifications to SORT, ByteTrack, or shared
motion code, see [CUSTOM_TRACKING.md](./CUSTOM_TRACKING.md).

A tracker is any class with `update(...)` and `reset()` methods. It does not need to
inherit from an MLX base class. Constructor keyword arguments come directly from
`--tracker-config`.

```python
# my_trackers/simple.py
from mlx.modes.object_detection.tracking import (
    TrackResult,
    TrackStatus,
    TrackingFrameResult,
)


class MyTracker:
    def __init__(self, *, start_id=1):
        self.start_id = start_id
        self.next_id = start_id

    def update(self, *, frame_index, detections, frame=None):
        tracks = []
        for detection in detections:
            tracks.append(
                TrackResult(
                    track_id=self.next_id,
                    bounding_box=detection.bounding_box,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    label=detection.label,
                    status=TrackStatus.CONFIRMED,
                    hits=1,
                    missing_frames=0,
                    last_seen_frame=frame_index,
                )
            )
            self.next_id += 1
        return TrackingFrameResult(frame_index=frame_index, tracks=tuple(tracks))

    def reset(self):
        self.next_id = self.start_id
```

Use it without modifying MLX:

```bash
python -m mlx --mode track \
    --tracker my_trackers.simple:MyTracker \
    --tracker-config ./my-tracker.json \
    --model-path ./detector.onnx \
    --file-path ./video.mp4 \
    --output ./tracking/custom
```

Applications that need short aliases can create an immutable custom registry and
inject it into `CreateTrackingAlgorithm`:

```python
from mlx.modes.object_detection.tracking import (
    CreateTrackingAlgorithm,
    register_tracker,
)

registry = register_tracker(
    "my-tracker",
    "my_trackers.simple:MyTracker",
)
tracker = CreateTrackingAlgorithm(
    tracker="my-tracker",
    registry=registry,
).execute()
```

`register_tracker` returns a new registry; it does not mutate process-wide state.
CLI-only users should prefer the direct `package.module:ClassName` selector, which
requires no registration or MLX source changes.

The tracker may retain active and temporarily lost track state. It must not retain
frames, raw provider results, every previous `TrackingFrameResult`, or unbounded
trajectory history. `reset()` must discard all state belonging to the previous
video.

## Python API

`RunTrackingVideo.execute()` is the complete video command. For applications that
already own a detector or frame loop, `RunObjectDetectionTrackingCommand` performs
detection plus one tracking update, while `RunTrackByDetectionCommand` accepts
already normalized `TrackingDetection` values.
