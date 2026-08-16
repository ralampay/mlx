# Creating and Modifying Tracking Algorithms

This guide explains how to add a tracking algorithm directly to the MLX source
tree or modify the built-in SORT and ByteTrack implementations. For CLI usage,
MOT output, and benchmarking, see [TRACKING.md](./TRACKING.md).

## How Tracking Fits into MLX

Tracking algorithms do not load models or call Ultralytics or LibreYOLO. MLX owns
that part of the workflow and supplies provider-neutral detections:

```text
video frame
    → selected object-detection provider
    → TrackingDetection values
    → your TrackingAlgorithm.update(...)
    → TrackingFrameResult
    → tracks.jsonl + tracks.txt and optional MOT metrics
```

This boundary makes one tracking class usable with both detection providers.

The relevant source layout is:

```text
mlx/modes/object_detection/tracking/
├── protocols.py              TrackingAlgorithm contract
├── models.py                 input, state, and result types
├── registry.py               built-in names used by --tracker
├── detection.py              detection-to-tracking composition
├── class_aware.py            class-aware writer and MOT extraction command
├── mot.py                    strict MOT serialization and parsing
├── replay.py                 portable replay projection command
├── session.py                complete video workflow
└── algorithms/
    ├── motion.py             shared Kalman, IoU, and association code
    ├── sort.py               built-in SORT example
    └── bytetrack.py          built-in ByteTrack example
```

## Expected Tracking Formats

There are five related formats in the workflow:

1. `TrackingDetection` values supplied to the algorithm;
2. `TrackingFrameResult` values returned by the algorithm;
3. class-aware `tracks.jsonl` records written to disk;
4. 10-column MOT rows written to disk and accepted as ground truth;
5. the provider-neutral `replay.json` projection exported from those artifacts.

Do not pass provider-specific Ultralytics or LibreYOLO result objects into a tracker.
Do not return MOT `left, top, width, height` rows directly from `update()`. MLX owns
the conversion between its in-memory types and the file format.

The class fields have one clear source of truth: the `class_id` and `label` on each
returned `TrackResult`. MLX copies those values into `tracks.jsonl`, `replay.json`,
the live overlay, and the offline HTML replay. They are not stored in `tracks.txt`,
because strict MOTChallenge prediction rows do not define a class column.

| Consumer/artifact | Stores class ID and label? | Notes |
| --- | --- | --- |
| Live OpenCV overlay | Yes | Uses the current `TrackingFrameResult` directly |
| `tracks.jsonl` | Yes | Versioned MLX class-aware interchange format |
| `tracks.txt` | No | Strict, class-agnostic 10-column MOT format |
| Prediction rows in `replay.json` | Yes | Joined from `tracks.jsonl` and `tracks.txt` |
| Ground-truth rows in `replay.json` | No | Class fields are `null` for standard MOT GT |

The default CLI visualization consumes the same `TrackingFrameResult`; a custom
tracker does not implement any OpenCV drawing. For every track whose
`last_seen_frame` equals the current `frame_index`, MLX draws the `xyxy` bounding
box and shows the ID, label or class ID, confidence, and status. This makes the
visualization another consumer of the documented output contract rather than a
tracker-specific integration.

### Algorithm input: `TrackingDetection`

Each detection represents one object candidate in the current frame:

```python
TrackingDetection(
    bounding_box=BoundingBox(
        x1=88.0,
        y1=99.0,
        x2=149.08,
        y2=317.56,
    ),
    confidence=0.92,
    class_id=0,
    label="person",
)
```

Bounding boxes use absolute image pixels in `xyxy` form:

```text
(x1, y1) ──────────────┐
   │                   │
   │                   │
   └────────────── (x2, y2)
```

- `x1, y1` are the top-left coordinates.
- `x2, y2` are the bottom-right coordinates.
- Width is `x2 - x1`; height is `y2 - y1`.
- Coordinates are floating-point values and must be finite.
- `x2` must be greater than or equal to `x1`; the same applies to `y2` and `y1`.
- `confidence` is the detector score and must be finite.
- `class_id` and `label` have already been normalized by the selected provider.

For a matched track, preserve the detection's class values in the emitted
`TrackResult`. If an algorithm permits a track's class to change, the value emitted
for that frame is what MLX records; most trackers should instead restrict
association to detections with the same class.

The sequence contains only the current frame's detections. `--track-class-id`
filtering happens before the sequence reaches the algorithm.

### Algorithm output: `TrackingFrameResult`

Return one frame result whose `frame_index` matches the value passed to `update()`:

```python
TrackingFrameResult(
    frame_index=1,
    tracks=(
        TrackResult(
            track_id=1,
            bounding_box=BoundingBox(88.0, 99.0, 149.08, 317.56),
            confidence=0.92,
            class_id=0,
            label="person",
            status=TrackStatus.CONFIRMED,
            hits=3,
            missing_frames=0,
            last_seen_frame=1,
        ),
    ),
)
```

`TrackResult.bounding_box` remains in `xyxy` form. Do not convert it to width and
height. MLX performs that conversion when writing MOT output.

The video command starts at frame `1` and increments once per decoded frame. A frame
may validly return an empty `tracks` tuple. Track IDs must be unique within the
result, and IDs intended for MOT export must be positive integers.

### File output and ground truth: MOT rows

`tracks.txt` and the supplied ground-truth file are headerless CSV files with exactly
10 fields per non-empty row:

```text
frame,id,left,top,width,height,confidence,world_x,world_y,world_z
```

Example prediction corresponding to the in-memory result above:

```text
1,1,88,99,61.08,218.56,0.92,-1,-1,-1
```

| Column | Type | Meaning |
| --- | --- | --- |
| `frame` | positive integer | 1-based video frame number |
| `id` | positive integer | identity that remains stable for the same object |
| `left` | finite float | bounding-box top-left x coordinate in pixels |
| `top` | finite float | bounding-box top-left y coordinate in pixels |
| `width` | non-negative float | `x2 - x1` in pixels |
| `height` | non-negative float | `y2 - y1` in pixels |
| `confidence` | finite float | prediction confidence; for GT this is the validity/mark field |
| `world_x` | finite float | optional world coordinate; predictions use `-1` |
| `world_y` | finite float | optional world coordinate; predictions use `-1` |
| `world_z` | finite float | optional world coordinate; predictions use `-1` |

MLX preserves fractional coordinates and writes up to six decimal places without
unnecessary trailing zeros. It does not add one pixel when converting `xyxy` to
width and height.

The MOT file has no detector class column. If the ground truth covers one class,
use one or more `--track-class-id` flags to filter detector output before tracking.
Benchmark matching is class-agnostic after that filtering.

Ground-truth rows whose seventh field is less than or equal to zero are ignored.
The final three ground-truth fields may contain world coordinates, but 2D evaluation
does not use them. Prediction rows always contain `-1,-1,-1` in those fields.

Within a frame, the same track ID may appear only once. Empty frames are represented
by the absence of rows for that frame; do not write a placeholder row.

### Which returned tracks are written

Returning a track from `update()` does not automatically make it a MOT prediction.
The writer exports it only when both conditions are true:

```python
track.status is TrackStatus.CONFIRMED
track.last_seen_frame == tracking_frame_result.frame_index
```

Therefore:

- a newly created `TENTATIVE` track is retained in memory but not written;
- a matched `CONFIRMED` track is written;
- a retained `LOST` track is not written;
- a predicted but unmatched box is not written, even if the algorithm keeps it for
  future recovery.

This policy ensures `tracks.txt` contains observations supported by a detection in
that frame while the tracking class remains free to keep bounded lost-track state.

The same eligibility rule applies to `tracks.jsonl`, so both output files describe
the same `(frame_id, track_id)` rows.

### Class-aware file output: `tracks.jsonl`

MOTChallenge prediction rows have no class field. MLX therefore writes a separate,
versioned JSON Lines file that preserves each track's detector class without
changing the MOT standard. Every non-empty line is one independent JSON object:

```json
{"schema_version":"mlx.tracking.record/v1","frame_id":1,"track_id":1,"class_id":0,"label":"person","bounding_box":{"x1":88.0,"y1":99.0,"x2":149.08,"y2":317.56},"confidence":0.92}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Must be `mlx.tracking.record/v1` |
| `frame_id` | positive integer | 1-based video frame number |
| `track_id` | positive integer | Stable identity within the run |
| `class_id` | non-negative integer | Provider-normalized detector class |
| `label` | string or null | Optional human-readable detector label |
| `bounding_box` | object | Finite `x1`, `y1`, `x2`, `y2` pixel coordinates |
| `confidence` | finite number | Most recent matched detector confidence |

The file is safe to stream line by line and does not contain provider-specific
objects. Within one frame, a track ID can occur only once. Use the public parser
when consuming it from Python:

```python
from pathlib import Path

from mlx.modes.object_detection.tracking import read_class_aware_tracking_file

records = read_class_aware_tracking_file(Path("./tracking/tracks.jsonl"))
for record in records:
    print(record.frame_id, record.track_id, record.class_id, record.label)
```

To extract a strict MOTChallenge prediction file, including an optional class
selection, use the command API:

```python
from pathlib import Path

from mlx.modes.object_detection.tracking import ExportMOTFromClassAwareTracking

result = ExportMOTFromClassAwareTracking(
    source_path=Path("./tracking/tracks.jsonl"),
    output_dir=Path("./tracking/person-mot"),
    class_ids=(0,),  # Empty tuple exports all classes.
    overwrite=True,
).execute()
print(result.output_path)
```

The equivalent CLI is:

```bash
python -m mlx --mode track --action export-mot \
    --tracking-jsonl ./tracking/tracks.jsonl \
    --track-class-id 0 \
    --output ./tracking/person-mot
```

This separation is intentional: `tracks.jsonl` is lossless and class-aware, while
the derived `tracks.txt` always remains interoperable with MOT tooling.

During `track --action run`, repeatable `--track-class-id` values filter detections
before they enter the tracking algorithm. During `track --action export-mot`, the
same flag filters already-written JSONL records. Post-run extraction does not alter
track IDs or rerun association.

### Benchmark matching

When `--ground-truth` is supplied, MLX groups prediction and GT rows by frame and
performs class-agnostic IoU matching. The default minimum match overlap is `0.5` and
can be changed with `--benchmark-iou`.

The benchmark writes `metrics.json` containing:

- MOTA;
- MOTP, reported as mean matched IoU;
- IDF1;
- precision and recall;
- match, false-positive, miss, and ID-switch counts;
- processed frame, prediction, and ground-truth-object counts.

The video command invokes this benchmark automatically against its strict
`tracks.txt` when `--ground-truth` is supplied. An extracted MOT projection can be
benchmarked independently:

```python
from pathlib import Path

from mlx.modes.object_detection.tracking.evaluation import BenchmarkMOTTracking

benchmark = BenchmarkMOTTracking(
    ground_truth_path=Path("./gt.txt"),
    predictions_path=Path("./tracking/person-mot/tracks.txt"),
    output_path=Path("./tracking/person-mot/metrics.json"),
    iou_threshold=0.5,
    overwrite=True,
).execute()
```

Because standard MOT ground truth does not identify detector classes, make sure the
extracted prediction classes correspond to the population represented by the GT.

### Portable replay JSON

`RunTrackingVideo` also exports `replay.json` using schema version
`mlx.tracking.replay/v1`. It combines `tracks.txt`, `tracks.jsonl`, and optional
ground truth; trackers do not return or write this schema themselves. A shortened
example is:

```json
{
  "schema_version": "mlx.tracking.replay/v1",
  "coordinate_system": {
    "origin": "top-left",
    "units": "pixels",
    "bounding_box": "left,top,width,height",
    "y_axis": "down"
  },
  "canvas": {"width": 640, "height": 480},
  "frame_count": 179,
  "fps": 25.0,
  "run": {
    "provider": "libreyolo",
    "tracker": "bytetrack",
    "detector_confidence": 0.03,
    "tracked_class_ids": [0],
    "benchmark_iou": 0.5
  },
  "predictions": {
    "record_count": 1,
    "track_count": 1,
    "records": [
      {
        "frame_id": 1,
        "track_id": 1,
        "left": 88.0,
        "top": 99.0,
        "width": 61.08,
        "height": 218.56,
        "confidence": 0.92,
        "class_id": 0,
        "label": "person"
      }
    ]
  },
  "ground_truth": null,
  "metrics": null
}
```

Prediction rows receive `class_id` and `label` from `tracks.jsonl`. Ground-truth
rows use the same collection shape but set both fields to `null`, because standard
10-column MOT ground truth has no class column. When metrics are available,
`metrics` contains the values from `metrics.json`.
`source.name` may preserve only the input filename; no absolute video path or video
pixels are required. `replay.html` embeds this payload for offline playback.

Applications can export an existing MOT result without loading a detector:

```python
from pathlib import Path

from mlx.modes.object_detection.tracking import ExportTrackingReplay

result = ExportTrackingReplay(
    predictions_path=Path("./tracking/tracks.txt"),
    class_aware_path=Path("./tracking/tracks.jsonl"),
    ground_truth_path=Path("./gt.txt"),
    output_dir=Path("./tracking"),
    frame_width=640,
    frame_height=480,
    frame_count=179,
    fps=25,
    overwrite=True,
).execute()
```

If dimensions are omitted, the exporter expands the canvas to the maximum box
extent. Supplying the decoded frame dimensions is preferred because it preserves
empty margins and makes projections comparable across runs.

When `class_aware_path` is omitted, replay export remains compatible with an
existing MOT-only file, but prediction `class_id` and `label` fields are `null` and
the player cannot display a class label. When it is supplied, the exporter validates
that both files contain the same frame/track pairs and matching geometry before
combining them.

## Required Class Contract

MLX uses structural typing. Your class does not need to inherit from a base class,
but it must implement these two methods with the same keyword-oriented interface:

```python
from collections.abc import Sequence

import numpy as np

from mlx.modes.object_detection.tracking import (
    TrackingDetection,
    TrackingFrameResult,
)


class MyTrackingAlgorithm:
    def update(
        self,
        *,
        frame_index: int,
        detections: Sequence[TrackingDetection],
        frame: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        ...

    def reset(self) -> None:
        ...
```

`update()` is called exactly once for every decoded video frame. The normal video
command supplies 1-based, monotonically increasing frame indexes.

Each `TrackingDetection` contains:

- `bounding_box`: immutable floating-point `x1, y1, x2, y2` coordinates;
- `confidence`: detector confidence;
- `class_id`: provider-normalized integer class ID;
- `label`: optional human-readable class label.

The optional `frame` is the current BGR NumPy image. Geometry-only algorithms may
ignore it. Appearance or camera-motion algorithms may inspect it, but must not keep
the image after `update()` returns.

## Result Rules

Return one `TrackingFrameResult` containing immutable `TrackResult` snapshots:

```python
from mlx.modes.object_detection.tracking import (
    TrackResult,
    TrackStatus,
    TrackingFrameResult,
)

result = TrackingFrameResult(
    frame_index=frame_index,
    tracks=(
        TrackResult(
            track_id=1,
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
            status=TrackStatus.CONFIRMED,
            hits=3,
            missing_frames=0,
            last_seen_frame=frame_index,
        ),
    ),
)
```

Follow these invariants:

- Track IDs must be positive and unique within a frame result.
- Reuse an ID only for the same tracked object during one session.
- Use `TENTATIVE` before a track satisfies your confirmation policy.
- Use `CONFIRMED` for established tracks.
- Use `LOST` for retained state that was not observed in the current frame.
- Set `last_seen_frame` to the frame of the track's latest matched detection.
- Set `missing_frames` to the number of frames since the latest match.
- `reset()` must remove every track and restart ID allocation for a new session.

The MOT writer exports only tracks whose status is `CONFIRMED` and whose
`last_seen_frame` equals the current result's `frame_index`. Tentative and lost
tracks may be returned for API consumers, but they are not written to `tracks.txt`.
The live display is intentionally broader: it shows both tentative and confirmed
tracks observed in the current frame, allowing practitioners to inspect lifecycle
transitions while tuning an algorithm. Lost tracks are not drawn.

## Worked Improvement: Add a Confidence Gate to SORT

This example improves the existing `SortTrackingAlgorithm` by preventing weak
detections from creating or updating tracks. It demonstrates why MLX is a utility:
you change one algorithm class, while MLX continues to provide the detector,
provider normalization, video loop, visualization, MOT writer, benchmark, and
offline replay.

The public contract does not change:

| Boundary | Before and after the change |
| --- | --- |
| Input | `Sequence[TrackingDetection]` for one frame, plus its `frame_index` and optional BGR frame |
| Internal change | Ignore detections whose `confidence` is below `min_detection_confidence` |
| Output | One `TrackingFrameResult` containing `TrackResult` snapshots |
| Downstream behavior | Confirmed current tracks are written to MOT; current tracks are visualized; MOT rows feed metrics and replay |

### 1. Modify only `SortTrackingAlgorithm`

Edit `mlx/modes/object_detection/tracking/algorithms/sort.py`. Add a keyword-only
constructor option, validate it, and associate against a filtered tuple:

```diff
 class SortTrackingAlgorithm:
     def __init__(
         self,
         *,
         iou_threshold: float = 0.3,
         max_age: int = 30,
         min_hits: int = 3,
+        min_detection_confidence: float = 0.0,
     ) -> None:
         validate_common_settings(
             iou_threshold=iou_threshold,
             max_age=max_age,
             min_hits=min_hits,
         )
+        if not 0.0 <= min_detection_confidence <= 1.0:
+            raise ValueError(
+                "min_detection_confidence must be between 0 and 1."
+            )
         self.iou_threshold = float(iou_threshold)
         self.max_age = int(max_age)
         self.min_hits = int(min_hits)
+        self.min_detection_confidence = float(min_detection_confidence)

     def update(self, *, frame_index, detections, frame=None):
         del frame
         if frame_index < 0:
             raise ValueError("Tracking frame index must be zero or greater.")
+        eligible_detections = tuple(
+            detection
+            for detection in detections
+            if detection.confidence >= self.min_detection_confidence
+        )
         self._predict_tracks()
         matches, _, unmatched_detection_indices = associate_detections(
             self._tracks,
-            detections,
+            eligible_detections,
             iou_threshold=self.iou_threshold,
         )
         for track_index, detection_index in matches:
             self._tracks[track_index].update(
-                detections[detection_index],
+                eligible_detections[detection_index],
                 frame_index=frame_index,
                 min_hits=self.min_hits,
             )
         for detection_index in unmatched_detection_indices:
             self._create_track(
-                detections[detection_index], frame_index=frame_index
+                eligible_detections[detection_index], frame_index=frame_index
             )
```

Keep every remaining line of `update()` unchanged. The filtered detections still use
the same provider-neutral type, so the change works with both Ultralytics and
LibreYOLO. A detector cannot restore boxes it has already discarded, so set CLI
`--confidence` at or below `min_detection_confidence`.

When a weak detection would otherwise update an existing track, that track instead
advances through its normal unmatched lifecycle. It may be returned as `LOST`, but
MLX will neither draw it as a current observation nor write it as a MOT prediction.
This is an intentional output consequence to verify during benchmarking.

### 2. Configure the new option without changing the CLI

Create `sort-confidence.json`:

```json
{
  "iou_threshold": 0.3,
  "max_age": 30,
  "min_hits": 2,
  "min_detection_confidence": 0.35
}
```

`CreateTrackingAlgorithm` already passes JSON keys to the tracker constructor. The
existing `sort` registry entry and CLI flag therefore need no change:

```bash
python -m mlx --mode track --action run \
    --tracker sort \
    --tracker-config ./sort-confidence.json \
    --confidence 0.1 \
    --model-path ./detector.onnx \
    --file-path ./video.mp4 \
    --ground-truth ./gt.txt \
    --output ./tracking/sort-confidence
```

The run automatically produces live boxes, `tracks.jsonl`, `tracks.txt`,
`metrics.json`, `replay.json`, and `replay.html`. Compare the benchmark with the
unmodified default before deciding whether `0.35` improves your sequence.

### 3. Add one focused regression test

```python
def test_sort_confidence_gate_ignores_weak_detections():
    tracker = SortTrackingAlgorithm(
        min_hits=1,
        min_detection_confidence=0.5,
    )

    result = tracker.update(
        frame_index=1,
        detections=(make_detection(confidence=0.49),),
    )

    assert result == TrackingFrameResult(frame_index=1, tracks=())
```

Also test that `0.50` is accepted, a confidence above the gate creates a track, and
values outside `[0, 1]` raise `ValueError`. This modification is deliberately local:
no detector adapter, runner, writer, evaluator, or presenter needs to know the new
option exists.

## New Tracker Exercise: Center-Distance Association

The exercise is to add a real stateful class named
`CenterDistanceTrackingAlgorithm`. It reuses MLX's bounded `MotionTrack` lifecycle
and Kalman prediction, but replaces IoU/Hungarian association with a small,
class-aware greedy center-distance method. This leaves one clear algorithmic method,
`_associate()`, for a practitioner to experiment with.

### Exercise input and output

For every decoded frame MLX calls:

```python
result = tracker.update(
    frame_index=17,
    detections=(detection_a, detection_b),
    frame=current_bgr_frame,
)
```

The class consumes:

- `frame_index`: the current 1-based frame number in normal video execution;
- `detections`: only the current frame's provider-neutral detections, already
  filtered by any `--track-class-id` flags;
- `frame`: the optional current BGR NumPy array, unused by this geometry-only
  exercise and never retained.

It returns:

```python
TrackingFrameResult(
    frame_index=17,
    tracks=(
        TrackResult(
            track_id=3,
            bounding_box=BoundingBox(...),
            confidence=0.91,
            class_id=0,
            label="person",
            status=TrackStatus.CONFIRMED,
            hits=8,
            missing_frames=0,
            last_seen_frame=17,
        ),
    ),
)
```

The returned snapshots may include tentative, confirmed, and temporarily lost
tracks. MLX, rather than this class, decides which snapshots are written or drawn:

```text
your TrackingFrameResult
    ├── current tentative/confirmed → live bounding-box visualization
    ├── current confirmed           → tracks.jsonl + tracks.txt
    ├── tracks.txt + gt.txt         → metrics.json
    └── both tracking projections   → replay.json + replay.html
```

### 1. Add the algorithm module

Create:

```text
mlx/modes/object_detection/tracking/algorithms/center_distance.py
```

Copy this complete baseline implementation:

```python
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mlx.modes.object_detection.tracking.algorithms.motion import MotionTrack
from mlx.modes.object_detection.tracking.models import (
    BoundingBox,
    TrackingDetection,
    TrackingFrameResult,
)


class CenterDistanceTrackingAlgorithm:
    """Associate same-class tracks by normalized bounding-box center distance."""

    def __init__(
        self,
        *,
        max_center_distance_ratio: float = 1.5,
        max_age: int = 30,
        min_hits: int = 2,
    ) -> None:
        if max_age < 0:
            raise ValueError("max_age must be zero or greater.")
        if min_hits < 1:
            raise ValueError("min_hits must be one or greater.")
        if (
            not math.isfinite(max_center_distance_ratio)
            or max_center_distance_ratio <= 0
        ):
            raise ValueError(
                "max_center_distance_ratio must be a finite value above zero."
            )
        self.max_center_distance_ratio = float(max_center_distance_ratio)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._tracks: list[MotionTrack] = []
        self._next_track_id = 1

    def update(
        self,
        *,
        frame_index: int,
        detections: Sequence[TrackingDetection],
        frame: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        del frame
        if frame_index < 0:
            raise ValueError("Tracking frame index must be zero or greater.")

        for track in self._tracks:
            track.predict()
        matches, unmatched_detection_indices = self._associate(detections)

        for track_index, detection_index in matches:
            self._tracks[track_index].update(
                detections[detection_index],
                frame_index=frame_index,
                min_hits=self.min_hits,
            )
        for detection_index in unmatched_detection_indices:
            self._create_track(
                detections[detection_index],
                frame_index=frame_index,
            )

        self._tracks = [
            track for track in self._tracks if track.missing_frames <= self.max_age
        ]
        return TrackingFrameResult(
            frame_index=frame_index,
            tracks=tuple(track.snapshot() for track in self._tracks),
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def _associate(
        self,
        detections: Sequence[TrackingDetection],
    ) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
        candidates = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                distance = _normalized_center_distance(
                    track.bounding_box,
                    detection.bounding_box,
                )
                if distance <= self.max_center_distance_ratio:
                    candidates.append((distance, track_index, detection_index))

        matches = []
        matched_tracks = set()
        matched_detections = set()
        for _, track_index, detection_index in sorted(candidates):
            if track_index in matched_tracks:
                continue
            if detection_index in matched_detections:
                continue
            matches.append((track_index, detection_index))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        unmatched_detections = tuple(
            index
            for index in range(len(detections))
            if index not in matched_detections
        )
        return tuple(matches), unmatched_detections

    def _create_track(
        self,
        detection: TrackingDetection,
        *,
        frame_index: int,
    ) -> None:
        self._tracks.append(
            MotionTrack.create(
                track_id=self._next_track_id,
                detection=detection,
                frame_index=frame_index,
                min_hits=self.min_hits,
            )
        )
        self._next_track_id += 1


def _normalized_center_distance(first: BoundingBox, second: BoundingBox) -> float:
    first_center = ((first.x1 + first.x2) / 2, (first.y1 + first.y2) / 2)
    second_center = ((second.x1 + second.x2) / 2, (second.y1 + second.y2) / 2)
    center_distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    scale = max(
        math.hypot(first.width, first.height),
        math.hypot(second.width, second.height),
        1.0,
    )
    return center_distance / scale
```

The distance is divided by the larger box diagonal, so the setting is less tied to
one video resolution than a raw pixel threshold. The greedy association is the
exercise's deliberate simplification. Try replacing only `_associate()` with a
Hungarian assignment, motion-direction penalty, or appearance-distance term while
leaving the MLX contract untouched.

This is an in-repository exercise, so it deliberately reuses the mode-owned
`MotionTrack` building block. A separately distributed tracker package should own
its internal state implementation and depend only on the public types and
`TrackingAlgorithm` contract shown earlier in this guide.

### 2. Add a lazy built-in alias

Edit `BUILTIN_TRACKERS` in
`mlx/modes/object_detection/tracking/registry.py`:

```python
BUILTIN_TRACKERS = MappingProxyType(
    {
        "bytetrack": "...",
        "sort": "...",
        "center-distance": (
            "mlx.modes.object_detection.tracking.algorithms.center_distance:"
            "CenterDistanceTrackingAlgorithm"
        ),
    }
)
```

The registry stores an import string rather than importing the class eagerly. This
keeps optional tracker dependencies isolated and allows `ls-trackers` to stay cheap.

Optionally export the class from `algorithms/__init__.py` for Python callers:

```python
from mlx.modes.object_detection.tracking.algorithms.center_distance import (
    CenterDistanceTrackingAlgorithm,
)

__all__ = [
    "ByteTrackAlgorithm",
    "CenterDistanceTrackingAlgorithm",
    "DetectionAsTrackAlgorithm",
    "SortTrackingAlgorithm",
]
```

### 3. Configure and run it through all MLX facilities

Create `center-distance.json`:

```json
{
  "max_center_distance_ratio": 1.25,
  "max_age": 20,
  "min_hits": 2
}
```

Constructor keys are injected by `CreateTrackingAlgorithm`; the class does not
parse JSON or CLI arguments:

```bash
python -m mlx --mode track --action run \
    --provider libreyolo \
    --tracker center-distance \
    --tracker-config ./center-distance.json \
    --model-path ./best.pt \
    --file-path ./video.mp4 \
    --ground-truth ./gt.txt \
    --track-class-id 0 \
    --output ./tracking/center-distance
```

Switching `--provider` to `ultralytics` requires no tracker change because both
providers supply identical `TrackingDetection` inputs.

### 4. Prove the input/output contract with tests

Add these tests to `tests/test_tracking_mode.py`. They use normalized detections and
never load a detector or video:

```python
from mlx.modes.object_detection.tracking import TrackingAlgorithm
from mlx.modes.object_detection.tracking.algorithms.center_distance import (
    CenterDistanceTrackingAlgorithm,
)


def test_center_distance_preserves_nearby_identity():
    tracker = CenterDistanceTrackingAlgorithm(min_hits=1)

    first = tracker.update(
        frame_index=1,
        detections=(make_detection(x=10),),
    )
    second = tracker.update(
        frame_index=2,
        detections=(make_detection(x=12),),
    )

    assert isinstance(tracker, TrackingAlgorithm)
    assert first.tracks[0].track_id == second.tracks[0].track_id == 1
    assert second.tracks[0].last_seen_frame == 2
    assert second.tracks[0].missing_frames == 0


def test_center_distance_reset_starts_a_clean_session():
    tracker = CenterDistanceTrackingAlgorithm(min_hits=1)
    tracker.update(frame_index=1, detections=(make_detection(),))

    tracker.reset()
    restarted = tracker.update(frame_index=1, detections=(make_detection(),))

    assert restarted.tracks[0].track_id == 1
```

Then add cases for a distant detection receiving a new ID, cross-class detections
not matching, tentative confirmation, lost-track expiry, empty detections, and an
invalid `max_center_distance_ratio`. The generic session tests already cover
downstream MOT, metrics, visualization, and replay behavior for any conforming
tracker.

## Use a Source Tracker Without Adding an Alias

During development, the CLI can import the class directly. This lets you test the
module before changing the built-in registry:

```bash
python -m mlx --mode track --action run \
    --provider libreyolo \
    --tracker \
      mlx.modes.object_detection.tracking.algorithms.center_distance:CenterDistanceTrackingAlgorithm \
    --tracker-config ./center-distance.json \
    --model-path ./best.pt \
    --file-path ./video.mp4 \
    --output ./tracking/center-distance
```

Once registered, verify the short alias:

```bash
python -m mlx --mode track --action ls-trackers
```

## Modify SORT or ByteTrack

The built-in implementations are intentionally separated into algorithm-specific
control flow and shared motion/association primitives.

### SORT

Modify `algorithms/sort.py` when changing:

- the single-stage association sequence;
- track creation or expiry policy;
- which detections are eligible for matching;
- the public constructor settings for SORT only.

`SortTrackingAlgorithm.update()` currently predicts all retained tracks, performs
one class-aware IoU/Hungarian assignment, updates matches, creates tracks from
unmatched detections, expires old tracks, and returns snapshots.

### ByteTrack

Modify `algorithms/bytetrack.py` when changing:

- high/low detection partitioning;
- first- or second-stage association;
- new-track confidence policy;
- ByteTrack-specific constructor settings.

`ByteTrackAlgorithm.update()` first associates high-confidence detections, then
offers low-confidence detections to unmatched tracks. Only unmatched high-confidence
detections that meet `new_track_threshold` can create tracks.

For ByteTrack, the detector-level CLI `--confidence` must be less than or equal to
the algorithm's `low_threshold`. A higher detector threshold discards boxes before
the tracker can use its second association stage.

### Shared motion and association

Modify `algorithms/motion.py` only when the change should affect both built-ins. It
owns:

- the constant-velocity bounding-box Kalman filter;
- mutable `MotionTrack` lifecycle state;
- class-aware IoU/Hungarian association;
- box IoU calculations;
- common threshold validation.

A shared change must be tested against both SORT and ByteTrack. If behavior is
specific to only one algorithm, keep it in that algorithm's module instead.

## State and Memory Requirements

Tracker state should remain proportional to active and temporarily lost objects:

```text
O(active tracks + retained lost tracks + current detections)
```

Do not retain:

- full video frames;
- raw Ultralytics or LibreYOLO results;
- every historical `TrackingFrameResult`;
- expired tracks indefinitely;
- unbounded appearance embeddings or trajectories.

Bound every cache, history, gallery, or lost-track pool with a documented setting.
Use `reset()` to clear all session-specific state.

## Validation Workflow

Run focused tests while developing:

```bash
python -m pytest -q tests/test_tracking_mode.py \
    tests/test_object_detection_tracking.py
```

Then run the full suite and basic CLI checks:

```bash
python -m pytest -q
python -m mlx --mode track --action ls-trackers
python -m compileall -q mlx
```

Finally, run the tracker on a representative video and inspect its artifacts:

```bash
python -m mlx --mode track --action run \
    --tracker center-distance \
    --tracker-config ./center-distance.json \
    --model-path ./detector.onnx \
    --file-path ./video.mp4 \
    --ground-truth ./gt.txt \
    --track-class-id 0 \
    --output ./tracking/center-distance
```

Confirm that `tracks.txt` has exactly 10 comma-separated fields per row and inspect
`tracks.jsonl` to confirm that class IDs and labels are retained. Inspect
`metrics.json` for MOTA, MOTP, IDF1, precision, recall, false positives, misses, and
ID switches. Open `replay.html` to confirm that IDs, classes, and boxes behave as
expected over time, and inspect `replay.json` when integrating another projection
tool.

## Contribution Checklist

- The class implements `update()` and `reset()` exactly.
- Inputs and outputs use provider-neutral MLX tracking types.
- The algorithm does not load detectors or perform CLI/presentation work.
- Constructor options are keyword-only, validated, and documented.
- Track IDs and lifecycle fields follow the result rules above.
- State is bounded and `reset()` clears the session.
- Optional dependencies fail with an actionable message.
- The lazy registry path is correct and `ls-trackers` lists the alias.
- Focused and full tests pass.
- `TRACKING.md` and `ARCHITECTURE.md` are updated if public behavior, ownership,
  data flow, or extension contracts change.
