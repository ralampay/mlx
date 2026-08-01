# Tracking by Detection

MLX provides a detector-independent online tracking layer for processing one video
frame at a time. Detection and tracking remain separate so tracking algorithms do
not depend on Ultralytics, a particular model format, or a particular video source.

## Processing Flow

```text
camera, video, or another frame source
    ↓
object-detection adapter
    ↓
normalized DetectionResult
    ↓
to_tracking_detections(...)
    ↓
TrackingDetection values
    ↓
RunTrackByDetectionCommand
    ↓
TrackingAlgorithm
    ↓
immutable TrackingFrameResult
```

The detector processes the image and produces normalized detections. The tracker
receives those detections and may optionally inspect the current frame. A tracking
algorithm never invokes YOLO or another detector directly.

## CLI Workflow with a Trained Detection Model

The object-detection CLI is used to train, convert, and verify the detector that
will supply detections to the tracking API.

### 1. Train or select a checkpoint

Train a model when a checkpoint is not already available:

```bash
python -m mlx --mode object_detection --action train \
    --dataset coco8 \
    --model draxnet-yolo26 \
    --output ./runs/draxnet \
    --epochs 10 \
    --device cpu
```

The selected checkpoint is normally under the run's `weights/` directory. Training
prefers `best.pt` by default when Ultralytics produces it; pass `--no-use-best` to
prefer `last.pt`.

The remaining examples use this checkpoint:

```text
./runs/draxnet/exp/weights/best.pt
```

Adjust the run-directory component when Ultralytics created a different run name.

### 2. Verify the `.pt` detector through the CLI

A PyTorch checkpoint requires both its architecture alias or YAML and its trained
weights:

```bash
python -m mlx --mode object_detection --action infer-video \
    --model draxnet-yolo26 \
    --model-path ./runs/draxnet/exp/weights/best.pt \
    --file-path ./videos/input.mp4 \
    --confidence 0.25 \
    --device cpu
```

For a camera source, use the same trained model with `infer-camera`:

```bash
python -m mlx --mode object_detection --action infer-camera \
    --model draxnet-yolo26 \
    --model-path ./runs/draxnet/exp/weights/best.pt \
    --camera-index 0 \
    --confidence 0.25 \
    --device cpu
```

These commands verify that the model loads and produces normalized detections. They
display detections only; they do not assign persistent track IDs.

### 3. Optionally convert and verify an ONNX detector

Convert the trained checkpoint:

```bash
python -m mlx --mode object_detection --action convert \
    --model-path ./runs/draxnet/exp/weights/best.pt \
    --output ./exports \
    --device cpu
```

Then run inference with the exported model:

```bash
python -m mlx --mode object_detection --action infer-video \
    --model-path ./exports/best.onnx \
    --file-path ./videos/input.mp4 \
    --confidence 0.25 \
    --device cpu
```

An `.onnx` model does not require `--model`; MLX selects the ONNX adapter from the
file extension.

### 4. Connect the verified model to tracking

Tracking is currently a Python API rather than a CLI action. After verifying the
trained detector with `infer-video` or `infer-camera`, use the same model path to
build the detector adapter described in the next section. This keeps CLI model
loading and tracking model loading on the same project adapter boundary.

## Using the Current Object-Detection Models

The recommended integration accepts the same detector returned by the existing
`build_detection_adapter(...)` factory. This works with the current
`UltralyticsDetectionAdapter` for `.pt` models and
`OnnxRuntimeDetectionAdapter` for `.onnx` models.

```python
from mlx.modes.object_detection.tracking.algorithms import DetectionAsTrackAlgorithm
from mlx.modes.object_detection.ultralytics import (
    RunObjectDetectionTrackingCommand,
)
from mlx.modes.object_detection.ultralytics.adapters import build_detection_adapter

detector = build_detection_adapter(
    resolved_cfg=resolved_cfg,
    resolved_weights=resolved_weights,
    device="cpu",
    imgsz=640,
    confidence=0.25,
)

tracking = RunObjectDetectionTrackingCommand(
    detection_model=detector,
    algorithm=DetectionAsTrackAlgorithm(),
)

while True:
    ok, frame = capture.read()
    if not ok:
        break

    result = tracking.execute(frame=frame)
    write_or_render(result)

tracking.reset()
```

`RunObjectDetectionTrackingCommand.execute()` performs these operations for each
frame:

1. Validates that the frame is a non-empty two- or three-dimensional NumPy array.
2. Calls `detection_model.predict(frame)`.
3. Converts the normalized detector output into immutable tracking detections.
4. Increments the tracking frame index.
5. Passes the detections and current frame to the selected tracking algorithm.
6. Returns an immutable per-frame result.

The injected detection model is the project's normalized detection adapter rather
than a raw Ultralytics `YOLO` object. This preserves one consistent boundary for
`.pt`, `.onnx`, and future detector implementations.

## Using Detection and Tracking Separately

Callers that already perform detection can use the lower-level command directly:

```python
from mlx.modes.object_detection.tracking import RunTrackByDetectionCommand
from mlx.modes.object_detection.tracking.algorithms import DetectionAsTrackAlgorithm
from mlx.modes.object_detection.ultralytics import to_tracking_detections

tracking = RunTrackByDetectionCommand(
    algorithm=DetectionAsTrackAlgorithm(),
)

detection_result = detector.predict(frame)
result = tracking.execute(
    detections=to_tracking_detections(detection_result.detections),
    frame=frame,
)
```

This form is useful when detections must also be rendered, filtered, recorded, or
sent to another consumer before tracking.

## Public Tracking Types

- `BoundingBox`: immutable float-based `x1, y1, x2, y2` coordinates.
- `TrackingDetection`: detector-independent input containing a box, confidence,
  class ID, and optional label.
- `TrackStatus`: `tentative`, `confirmed`, or `lost` lifecycle state.
- `TrackResult`: immutable public snapshot of one track.
- `TrackingFrameResult`: frame index and an immutable tuple of track snapshots.
- `TrackingAlgorithm`: runtime-checkable protocol for tracking implementations.
- `RunTrackByDetectionCommand`: detector-neutral per-frame tracking command.

Internal mutable track state is not returned to callers.

## Placeholder Algorithm

`DetectionAsTrackAlgorithm` validates the architecture but is not a temporal
tracker. It creates one confirmed track for every detection, allocates monotonically
increasing IDs, and does not associate detections across frames. Calling `reset()`
restarts its ID allocation at `1`.

## Implementing a Tracking Algorithm

Future algorithms use structural typing and do not need to inherit from a project
base class:

```python
from collections.abc import Sequence

import numpy as np

from mlx.modes.object_detection.tracking import (
    TrackingDetection,
    TrackingFrameResult,
)


class IoUTrackingAlgorithm:
    def update(
        self,
        *,
        frame_index: int,
        detections: Sequence[TrackingDetection],
        frame: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        # Associate current detections with bounded active track state.
        ...

    def reset(self) -> None:
        # Discard state belonging to the previous video session.
        ...
```

Geometry-only trackers can ignore `frame`. Appearance-based or camera-motion-aware
trackers may inspect it but must not retain it.

## Session and Memory Rules

Reuse one command and algorithm instance across every frame in a video session.
Call `reset()` before starting a new video or independent camera session.

The tracking layer does not retain:

- image frames;
- raw detector result objects;
- every prior `TrackingFrameResult`;
- full trajectories or every historical bounding box;
- expired tracks or unbounded appearance history.

Memory must remain proportional to active or temporarily lost tracks plus the
current frame's detections:

```text
O(active tracks + current-frame detections)
```

Applications that require session history are responsible for streaming each frame
result to disk, a database, or another external store.

## Next Step

The next tracking implementation should be a basic class-aware IoU association
algorithm with configurable matching thresholds, tentative/confirmed/lost lifecycle
transitions, and bounded expiry for missing tracks.
