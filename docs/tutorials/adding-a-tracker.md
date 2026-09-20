# Adding a tracker

## Responsibility and contract

A tracker receives externally computed detections through
`update(frame_index=..., detections=..., frame=None)` and returns a `TrackingFrameResult`.
`reset()` clears session state. No detector, window, CLI parser, or video decoder belongs inside it.

The existing [DetectionAsTrackAlgorithm](../../mlx/modes/object_detection/tracking/algorithms/placeholder.py)
is the smallest useful lifecycle reference: assign an increasing ID to each observation, and
reset the counter between sessions. It deliberately does not associate objects across frames.
Copy this implementation into your own package for a custom algorithm; no inheritance is required.

## Register, discover, execute, test

```python
from mlx.modes.object_detection.tracking.registry import TrackerRegistry, CreateTrackingAlgorithm, list_trackers
from mlx.modes.object_detection.tracking.models import BoundingBox, TrackingDetection

registry = TrackerRegistry({}).register(
    "observations",
    "mlx.modes.object_detection.tracking.algorithms.placeholder:DetectionAsTrackAlgorithm",
)
assert list_trackers(registry=registry) == ("observations",)
tracker = CreateTrackingAlgorithm(tracker="observations", registry=registry, options={}).execute()
detection = TrackingDetection(BoundingBox(0, 0, 2, 2), 0.9, 0, "object")
first = tracker.update(frame_index=0, detections=[detection])
second = tracker.update(frame_index=1, detections=[detection])
assert (first.tracks[0].track_id, second.tracks[0].track_id) == (1, 2)
tracker.reset()
assert tracker.update(frame_index=0, detections=[detection]).tracks[0].track_id == 1
```

The CLI accepts an installed `package.module:ClassName` through `--tracker`.
Constructor options come from a JSON object via `--tracker-config`; Python callers pass
`options` instead. Do not supply both. Use `--mode track --action ls-trackers` for built-ins.
The normal video command supplies detections and owns lifecycle cleanup.

## Files and expansion

Change your implementation, the optional `BUILTIN_TRACKERS` alias, and tests.
Do not edit video runners or detection providers. Add temporal association later while keeping
the same input/output values and reset semantics.
