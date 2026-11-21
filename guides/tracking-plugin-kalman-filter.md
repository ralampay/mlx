# Build an external Kalman filter plugin for MLX

**You do not need to fork MLX or add files inside its source tree.** In this guide you
create your own Python project, install it into the same environment as MLX, and tell
MLX the import path of your class. MLX already contains the plugin loader, tracking
commands, and output writers. Your project contains the custom tracking implementation.

```text
workspace/
├── mlx/                         # Existing MLX checkout; leave its source unchanged
└── custom-tracking-kalman-filter/       # New, user-owned project created in this guide
```

You will build `KalmanIoUTracker` and select it with
`--tracker custom_tracking.mlx_plugins.kalman:KalmanIoUTracker`. The text before `:` is an importable
Python module; the text after `:` is the class in that module. There is no registration
step inside MLX. The `mlx_plugins` directory below belongs to **your** package; its name
does not mean it belongs in MLX's source tree.

Follow the steps in order. All shell blocks use Bash on Linux/macOS. File blocks are
complete file contents: create the named file in your editor and paste the whole block.
Only replace paths explicitly marked as placeholders. For the other tracker guide,
use its own project directory and virtual environment; both examples use the package
name `custom_tracking`, so do not install both separate projects into one environment.

## Step 1: Create your project outside MLX

Start in the directory containing your existing MLX checkout. The commands below assume
that checkout is named `mlx`. Choose a new, empty project directory so you do not overwrite
an existing plugin. You need Python 3.10 or newer and a supported Python/PyTorch environment
as described in [MLX setup](../README.md).

Run these commands in your terminal:

```bash
mkdir custom-tracking-kalman-filter
cd custom-tracking-kalman-filter
python -m venv .venv
source .venv/bin/activate
```

Keep this terminal open. Run the remaining commands from `custom-tracking-kalman-filter/` with
this virtual environment activated. On another day, return to this directory and run
`source .venv/bin/activate` again before using the plugin.

## Step 2: Create the folders

```bash
mkdir -p src/custom_tracking/components src/custom_tracking/mlx_plugins configs tests examples
```

Once you finish Step 3, your project will contain:

```text
custom-tracking-kalman-filter/
├── pyproject.toml
├── src/custom_tracking/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── observations.py
│   │   ├── kalman.py
│   │   └── kalman_tracker.py
│   └── mlx_plugins/
│       ├── __init__.py
│       ├── boundary.py
│       └── kalman.py
├── configs/kalman.json
├── tests/test_plugin.py
└── examples/
    ├── __init__.py
    ├── run_observations.py
    └── session_factory.py
```

`components/` contains algorithm/backend code. `mlx_plugins/` converts MLX inputs and
outputs. `configs/` holds constructor settings. `tests/` checks the plugin without a
detector, and `examples/` gives you a runnable Python entrypoint.

## Step 3: Add the files

Create **every file below**, relative to the project directory you just created.
Do not run the Python file contents individually in the terminal. Save them in the
named files first. The `__init__.py` files make the directories Python packages.
There are no files to create or edit in the `mlx/` checkout.

**File 1 of 14.** This tells pip how to install your package and its dependencies. MLX itself is installed separately in Step 4.

### File: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "custom-tracking-plugins"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy>=1.10"]

[project.optional-dependencies]
test = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["src"]
```

**File 2 of 14.** Save this file exactly as shown.

### File: `src/custom_tracking/__init__.py`

```python
"""Standalone tracking plugin package."""
```

**File 3 of 14.** Save this file exactly as shown.

### File: `src/custom_tracking/components/__init__.py`

```python
"""Standalone tracking plugin package."""
```

**File 4 of 14.** Save this file exactly as shown.

### File: `src/custom_tracking/mlx_plugins/__init__.py`

```python
"""Standalone tracking plugin package."""
```

**File 5 of 14.** Internal observations and snapshots keep the tracking components independent of MLX.

### File: `src/custom_tracking/components/observations.py`

```python
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Observation:
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str | None = None

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        if not all(math.isfinite(v) for v in (*self.box, self.score)):
            raise ValueError("Boxes and confidence must be finite.")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Tracking boxes must have positive width and height.")
        if not 0 <= self.score <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if type(self.class_id) is not int or self.class_id < 0:
            raise ValueError("Class IDs must be nonnegative integers.")

@dataclass(frozen=True)
class Snapshot:
    identity: int
    observation: Observation
    hits: int
    missing: int
    last_seen: int
    confirmed: bool
```

**File 6 of 14.** This is the shared conversion boundary between your components and MLX data types.

### File: `src/custom_tracking/mlx_plugins/boundary.py`

```python
from mlx.modes.object_detection.tracking import (
    BoundingBox, TrackResult, TrackStatus, TrackingFrameResult,
)
from custom_tracking.components.observations import Observation

def observations(detections):
    return tuple(Observation(d.bounding_box.as_xyxy(), d.confidence,
                             d.class_id, d.label) for d in detections)

def validate_index(index, previous):
    if type(index) is not int or index != previous + 1:
        raise ValueError("Supply consecutive frame indices starting at 1; reset between sequences.")

def frame_result(index, snapshots):
    tracks = []
    for s in snapshots:
        status = (TrackStatus.LOST if s.confirmed and s.missing else
                  TrackStatus.CONFIRMED if s.confirmed else TrackStatus.TENTATIVE)
        tracks.append(TrackResult(
            track_id=s.identity, bounding_box=BoundingBox(*s.observation.box),
            confidence=s.observation.score, class_id=s.observation.class_id,
            label=s.observation.label, status=status, hits=s.hits,
            missing_frames=s.missing, last_seen_frame=s.last_seen,
        ))
    return TrackingFrameResult(frame_index=index, tracks=tuple(tracks))
```

**File 7 of 14.** The filter predicts bounding-box motion and corrects it using matched observations.

### File: `src/custom_tracking/components/kalman.py`

```python
import numpy as np

class BoxFilter:
    """State: cx, cy, width, height and their four velocities; dt=1."""
    def __init__(self, box):
        self.x = np.r_[self.measurement(box), np.zeros(4)]
        self.f = np.eye(8)
        self.f[:4, 4:] = np.eye(4)
        self.h = np.eye(4, 8)
        self.p = np.diag([10.] * 4 + [1000.] * 4)
        self.q = np.diag([1.] * 4 + [0.01] * 4)
        self.r = np.eye(4)

    @staticmethod
    def measurement(box):
        x1, y1, x2, y2 = box
        return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])

    def predict(self):
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.box()

    def update(self, box):
        residual = self.measurement(box) - self.h @ self.x
        innovation = self.h @ self.p @ self.h.T + self.r
        gain = np.linalg.solve(innovation, self.h @ self.p).T
        self.x += gain @ residual
        # Joseph form preserves covariance symmetry more reliably.
        a = np.eye(8) - gain @ self.h
        self.p = a @ self.p @ a.T + gain @ self.r @ gain.T
        return self.box()

    def box(self):
        self.x[2:4] = np.maximum(self.x[2:4], 1e-6)
        cx, cy, w, h = self.x[:4]
        return tuple(float(v) for v in (cx-w/2, cy-h/2, cx+w/2, cy+h/2))
```

**File 8 of 14.** This component matches detections to predictions and manages track lifetimes.

### File: `src/custom_tracking/components/kalman_tracker.py`

```python
from dataclasses import dataclass, replace
from typing import Protocol
import numpy as np
from scipy.optimize import linear_sum_assignment
from custom_tracking.components.kalman import BoxFilter
from custom_tracking.components.observations import Observation, Snapshot

def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(
        0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.

class MotionFilter(Protocol):
    def predict(self) -> tuple[float, float, float, float]: ...
    def update(self, box) -> tuple[float, float, float, float]: ...

@dataclass
class Track:
    identity: int
    filter: MotionFilter
    observation: Observation
    last_seen: int
    hits: int = 1
    streak: int = 1
    missing: int = 0
    confirmed: bool = False

class KalmanTracker:
    def __init__(self, *, iou_threshold=0.3, max_age=30, min_hits=3,
                 filter_factory=BoxFilter):
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1].")
        if type(max_age) is not int or max_age < 0:
            raise ValueError("max_age must be a nonnegative integer.")
        if type(min_hits) is not int or min_hits < 1:
            raise ValueError("min_hits must be a positive integer.")
        self.threshold, self.max_age, self.min_hits = iou_threshold, max_age, min_hits
        if not callable(filter_factory):
            raise ValueError("filter_factory must be callable.")
        self.filter_factory = filter_factory
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def update(self, index, detections):
        for track in self.tracks:
            track.observation = replace(track.observation, box=track.filter.predict())
            track.missing += 1
        n, m = len(self.tracks), len(detections)
        matches = []
        if n and m:
            # Dummy columns let every track remain unmatched. A large penalty
            # maximizes valid match count before minimizing total IoU distance.
            penalty = float(n + m + 1)
            cost = np.full((n, m+n), penalty)
            cost[:, :m] = penalty * 2
            for i, track in enumerate(self.tracks):
                for j, detection in enumerate(detections):
                    overlap = iou(track.observation.box, detection.box)
                    if track.observation.class_id == detection.class_id and overlap >= self.threshold:
                        cost[i, j] = 1 - overlap
            rows, cols = linear_sum_assignment(cost)
            matches = [(i, j) for i, j in zip(rows, cols) if j < m and cost[i, j] < penalty]
        used = set()
        matched_tracks = set()
        for i, j in matches:
            track, detection = self.tracks[i], detections[j]
            track.streak = track.streak + 1 if track.missing == 1 else 1
            track.observation = replace(detection, box=track.filter.update(detection.box))
            track.hits += 1
            track.missing = 0
            track.last_seen = index
            track.confirmed |= track.streak >= self.min_hits
            used.add(j)
            matched_tracks.add(i)
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.streak = 0
        self.tracks = [t for t in self.tracks if t.missing <= self.max_age]
        for j, detection in enumerate(detections):
            if j not in used:
                self.tracks.append(Track(self.next_id, self.filter_factory(detection.box), detection,
                                         index, confirmed=self.min_hits == 1))
                self.next_id += 1
        return tuple(Snapshot(t.identity, t.observation, t.hits, t.missing,
                              t.last_seen, t.confirmed) for t in self.tracks)
```

**File 9 of 14.** This is the `KalmanIoUTracker` class MLX will import. Its constructor accepts the algorithm-specific settings in the JSON file.

### File: `src/custom_tracking/mlx_plugins/kalman.py`

```python
from mlx.core.exceptions import MLXUserError
from custom_tracking.components.kalman_tracker import KalmanTracker
from custom_tracking.components.kalman import BoxFilter
from custom_tracking.mlx_plugins.boundary import observations, frame_result, validate_index

class KalmanIoUTracker:
    def __init__(self, *, iou_threshold=0.3, max_age=30, min_hits=3,
                 filter_factory=BoxFilter):
        self.core = KalmanTracker(iou_threshold=iou_threshold, max_age=max_age,
                                  min_hits=min_hits, filter_factory=filter_factory)
        self.index = 0

    def update(self, *, frame_index, detections, frame=None):
        try:
            validate_index(frame_index, self.index)
            result = self.core.update(frame_index, observations(detections))
            self.index = frame_index
            return frame_result(frame_index, result)
        except ValueError as exc:
            raise MLXUserError(f"Kalman tracker input failed: {exc}") from exc

    def reset(self):
        self.core.reset()
        self.index = 0
```

**File 10 of 14.** These keys are passed to your tracker constructor. Do not put detector paths or video paths here; those are MLX workflow arguments.

### File: `configs/kalman.json`

```json
{
  "iou_threshold": 0.3,
  "max_age": 30,
  "min_hits": 3
}
```

**File 11 of 14.** These synthetic tests exercise identities, missed observations, class isolation, and reset.

### File: `tests/test_plugin.py`

```python
from dataclasses import FrozenInstanceError
import gc
import weakref
import numpy as np
import pytest
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.tracking import (
    BoundingBox, TrackingDetection, TrackStatus, RunTrackByDetectionCommand,
)
from custom_tracking.mlx_plugins.kalman import KalmanIoUTracker

def make_tracker(**options):
    return KalmanIoUTracker(min_hits=2, max_age=1, **options)

def detection(class_id=0):
    return TrackingDetection(BoundingBox(4, 4, 20, 24), 0.9, class_id, "object")

def test_identity_confirmation_loss_expiry_and_reset():
    command = RunTrackByDetectionCommand(algorithm=make_tracker())
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    first = command.execute(detections=[detection()], frame=frame)
    second = command.execute(detections=[detection()], frame=frame)
    assert first.tracks[0].status is TrackStatus.TENTATIVE
    assert second.tracks[0].track_id == first.tracks[0].track_id == 1
    assert second.tracks[0].status is TrackStatus.CONFIRMED
    assert second.tracks[0].hits == 2
    assert second.tracks[0].label == "object"
    with pytest.raises(FrozenInstanceError):
        second.tracks[0].hits = 100
    lost = command.execute(detections=[])
    assert lost.tracks[0].status is TrackStatus.LOST
    assert lost.tracks[0].last_seen_frame == 2
    assert lost.tracks[0].missing_frames == 1
    assert command.execute(detections=[]).tracks == ()
    command.reset()
    restarted = command.execute(detections=[detection()], frame=frame)
    assert restarted.frame_index == restarted.tracks[0].track_id == 1

def test_classes_have_distinct_ids_and_absent_classes_age():
    command = RunTrackByDetectionCommand(algorithm=make_tracker())
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for _ in range(2):
        result = command.execute(detections=[detection(0), detection(1)], frame=frame)
    identities = {t.class_id: t.track_id for t in result.tracks}
    assert len(set(identities.values())) == 2
    for _ in range(2):
        result = command.execute(detections=[detection(1)], frame=frame)
    assert [(t.class_id, t.track_id) for t in result.tracks] == [(1, identities[1])]
    result = command.execute(detections=[detection(0), detection(1)], frame=frame)
    assert next(t.track_id for t in result.tracks if t.class_id == 0) > max(identities.values())

def test_rejects_invalid_input_without_advancing_plugin():
    tracker = make_tracker()
    with pytest.raises(MLXUserError, match="consecutive"):
        tracker.update(frame_index=2, detections=[])
    bad = TrackingDetection(BoundingBox(1, 1, 1, 2), 0.9, 0)
    with pytest.raises(MLXUserError, match="positive"):
        tracker.update(frame_index=1, detections=[bad])
    assert tracker.update(frame_index=1, detections=[]).frame_index == 1

def test_does_not_retain_input_frame():
    tracker = make_tracker()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    reference = weakref.ref(frame)
    tracker.update(frame_index=1, detections=[detection()], frame=frame)
    del frame
    gc.collect()
    assert reference() is None

def test_filter_learns_motion_and_predictions_remain_finite():
    from custom_tracking.components.kalman import BoxFilter
    motion = BoxFilter((0, 0, 10, 10))
    for x in range(1, 8):
        motion.predict()
        motion.update((x, 0, x+10, 10))
    predicted = motion.predict()
    assert predicted[0] > 7
    assert np.isfinite(predicted).all()
    assert np.allclose(motion.p, motion.p.T)
    assert np.linalg.eigvalsh(motion.p).min() >= -1e-10

def test_plugin_keyword_contract_and_frame_alignment():
    from inspect import signature
    from mlx.modes.object_detection.tracking import TrackingFrameResult
    tracker = make_tracker()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for index, detections in ((1, [detection()]), (2, [])):
        arguments = dict(frame_index=index, detections=detections, frame=frame)
        signature(tracker.update).bind(**arguments)
        result = tracker.update(**arguments)
        assert isinstance(result, TrackingFrameResult)
        assert result.frame_index == index
        assert len({t.track_id for t in result.tracks}) == len(result.tracks)
        assert all(type(t.track_id) is int and t.track_id > 0 for t in result.tracks)
        assert all(t.last_seen_frame <= index for t in result.tracks)
    signature(tracker.reset).bind()
    tracker.reset()
    assert tracker.update(frame_index=1, detections=[]).frame_index == 1

def test_session_factory_creates_independent_trackers():
    from examples.session_factory import make_session_factory
    from mlx.modes.object_detection.tracking import TrackingRequest
    factory = make_session_factory()
    first = factory(TrackingRequest())
    second = factory(TrackingRequest())
    assert first.algorithm is not second.algorithm
    # No detector/source is constructed until the session is executed.
    first.algorithm.update(frame_index=1, detections=[])
    assert second.algorithm.update(frame_index=1, detections=[]).frame_index == 1
    first.algorithm.reset()
    second.algorithm.reset()

def test_motion_filter_can_be_replaced_independently():
    class StationaryFilter:
        def __init__(self, box):
            self.box = box
        def predict(self):
            return self.box
        def update(self, box):
            self.box = box
            return box
    tracker = make_tracker(filter_factory=StationaryFilter)
    first = tracker.update(frame_index=1, detections=[detection()])
    second = tracker.update(frame_index=2, detections=[detection()])
    assert first.tracks[0].track_id == second.tracks[0].track_id
    assert second.tracks[0].status is TrackStatus.CONFIRMED
```

**File 12 of 14.** This complete script runs the plugin on synthetic observations without loading a detector. Run it in Step 7.

### File: `examples/run_observations.py`

```python
from pathlib import Path
import numpy as np
from mlx.modes.object_detection.tracking import (
    BoundingBox, TrackingDetection, CreateTrackingAlgorithm, RunTrackByDetectionCommand,
)

algorithm = CreateTrackingAlgorithm(
    tracker="custom_tracking.mlx_plugins.kalman:KalmanIoUTracker",
    config_path=str(Path(__file__).resolve().parents[1] / "configs/kalman.json"),
).execute()
command = RunTrackByDetectionCommand(algorithm=algorithm)
frame = np.zeros((32, 32, 3), dtype=np.uint8)
try:
    for _ in range(3):
        result = command.execute(
            detections=[TrackingDetection(BoundingBox(4, 4, 20, 24), 0.9, 0, "object")],
            frame=frame,
        )
    assert result.tracks[0].track_id == 1
    missing = command.execute(detections=[])
    assert missing.tracks[0].missing_frames == 1
    print(f"Confirmed track ID: {result.tracks[0].track_id}")
    print(f"After an empty frame: {missing.tracks[0].status.value}")
finally:
    command.reset()
```

**File 13 of 14.** This optional-use factory lets dataset benchmarks construct a fresh tracker per sequence. Save it now; it will be exercised by the tests even if you do not run a dataset benchmark.

### File: `examples/session_factory.py`

```python
from mlx.modes.object_detection.tracking import RunTrackingVideo
from custom_tracking.mlx_plugins.kalman import KalmanIoUTracker

def make_session_factory():
    def create_session(request, **collaborators):
        algorithm = KalmanIoUTracker(iou_threshold=0.3, max_age=30, min_hits=3)
        return RunTrackingVideo(request, algorithm=algorithm, **collaborators)

    return create_session
```

**File 14 of 14.** This makes the example helpers importable from the project directory for the benchmark command and tests.

### File: `examples/__init__.py`

```python
"""Runnable examples and composition helpers for this standalone project."""
```

## Step 4: Install MLX and your plugin into the same environment

You are still in `custom-tracking-kalman-filter/`, and `.venv` is active. With the sibling
checkout layout from Step 1, copy and run:

```bash
python -m pip install -e '../mlx[object-detection-libreyolo]'
python -m pip install -e '.[test]'
```

If your MLX checkout is elsewhere, replace only `../mlx` in the first command with its
actual path, keeping the quotes and `[object-detection-libreyolo]` extra. This installs
MLX and its LibreYOLO integration; it does not require editing MLX source. Do not run
`pip install mlx` from PyPI: Apple MLX is a different project with the same package name.

The second command installs **your** package. `-e` means edits to its Python files are
visible without reinstalling. After changing dependencies in `pyproject.toml`, rerun
that second command. No `PYTHONPATH` changes or copying files into site-packages are needed.

## Step 5: Verify MLX can load your class

Copy this complete command into the same terminal:

```bash
python - <<'PYTHON'
from mlx.modes.object_detection.tracking import CreateTrackingAlgorithm

tracker = CreateTrackingAlgorithm(
    tracker="custom_tracking.mlx_plugins.kalman:KalmanIoUTracker",
    config_path="configs/kalman.json",
).execute()
print(type(tracker).__name__)
tracker.reset()
PYTHON
```

Expected output includes `KalmanIoUTracker`. This proves the class can be imported and
constructed through MLX, rather than just imported by a script in your project.
It does not run detection or load video. Construction alone does not validate method
signatures or output semantics; Step 6 checks those contracts explicitly. `--action ls-trackers` lists registered aliases;
your external plugin does not need to appear there.

If you see `No module named custom_tracking`, activate this project's virtual environment
and rerun `python -m pip install -e '.[test]'` from the project directory. If MLX's modules
cannot be imported, check the checkout path used in Step 4. Always use the same `python`
for installation and execution.

## Step 6: Run the synthetic tests

```bash
python -m pytest -q tests
```

Expected result: **8 passed**. These tests require no detector checkpoint, dataset,
GPU, or video. They test the Kalman motion model as well as the tracking lifecycle.
Resolve any failures before continuing to a real video.

## Step 7: Run the Python example without a detector

You already saved the script in Step 3. Run:

```bash
python examples/run_observations.py
```

Expected output includes:

```text
Confirmed track ID: 1
After an empty frame: lost
```

This step needs only bounding-box observations. The optional image is not used by this tracker.

To adapt the script to your application, replace the synthetic observations with your
own observations and call `command.execute(...)` once per frame, including empty frames.
Create the algorithm **once before the loop** so identities persist. The script resets
it in `finally` when the session ends. Detection acquisition remains separate from tracking.

## Step 8: Track a real video through MLX

You now need an existing supported LibreYOLO `.pt` checkpoint and a video. The synthetic
steps do not create either file. In the command below, replace `/path/to/detector.pt`
and `/path/to/video.mp4` with your actual paths. Run from `custom-tracking-kalman-filter/`:

```bash
python -m mlx --mode track --action run \
  --provider libreyolo --model-path /path/to/detector.pt \
  --file-path /path/to/video.mp4 --device cpu --confidence 0.25 \
  --tracker custom_tracking.mlx_plugins.kalman:KalmanIoUTracker \
  --tracker-config configs/kalman.json \
  --output ./tracking-output --no-display
```

The `--tracker` argument is the exact module/class reference you verified in Step 5.
`--tracker-config` supplies algorithm-specific constructor settings. Model, video,
device, detector confidence, and output options belong to MLX's workflow.

On success, look in `tracking-output/` for:

- `tracks.jsonl`: class-aware observations with IDs.
- `tracks.txt`: MOTChallenge-format observations.
- `replay.json` and `replay.html`: open the HTML file in a browser to inspect trajectories.

The first frames may have no saved rows while tracks are tentative. Lost predictions
are not saved as fresh observations. Use a new output directory for another run, or
explicitly add `--overwrite` to replace output. `--no-display` disables the live window.

For evaluation, install the MLX `tracking` extra and supply a matching MOT ground-truth
file with `--ground-truth`. Compare IDF1, identity switches, and runtime on representative
video; the synthetic tests do not establish tracking accuracy.

## Step 9: Change your algorithm without editing MLX

Edit files under your project's `src/custom_tracking/`, adjust `configs/kalman.json`,
rerun the tests, and launch MLX again. Do not edit MLX's registry or CLI. An installed
package containing a compatible class is sufficient. Other users can install your
distributed package and use the same `--tracker` reference without writing plugin code.

Keep algorithm settings in JSON or the factory's Python `options` dictionary. Choose one
of `config_path` or `options`, not both. Injected encoders/backends are Python objects,
so pass them from Python rather than trying to encode callables in JSON.

If launching from a different directory, use absolute paths for `--tracker-config`, the
video, checkpoint, and output. Paths inside JSON are also relative to the process working
directory. Your Python class remains importable because you installed the package.

## How the example works

This guide builds a custom `KalmanIoUTracker` from small NumPy/SciPy components.
It does not wrap MLX's internal tracker classes. A Kalman filter alone is not a complete
multi-object tracker: we add class-aware IoU assignment and identity lifecycle handling.
The result is an educational SORT-style baseline, not an exact reproduction of SORT.
See the [SORT paper](https://arxiv.org/abs/1602.00763).

The filter stores `[cx, cy, width, height, vx, vy, vwidth, vheight]`. Prediction applies
constant velocity and propagates covariance: `x = F x`, `P = F P Fᵀ + Q`.
Correction uses the detection residual and Kalman gain to balance motion and observation
uncertainty. This example uses fixed process/measurement noise; detector confidence is
metadata, not a measurement covariance. Assignment is a separate step after prediction.

### Association and lifecycle policy

Disallow cross-class and below-threshold pairs before assignment. Dummy unmatched columns
avoid forcing invalid matches; costs prioritize the number of valid matches, then IoU.
New tracks confirm after `min_hits` consecutive observations. Confirmed tracks become lost
on a miss and can recover their ID; tracks expire after more than `max_age` misses.
Predicted widths/heights are kept positive. CPU-only NumPy/SciPy work is sufficient.

### MLX contract

The public API is `update(*, frame_index, detections, frame=None)` returning
`TrackingFrameResult`, plus `reset()`. Each detection contains pixel-space `xyxy`,
confidence, integer class ID, and an optional label. These examples additionally require
positive box area and scores in [0, 1]. They deliberately require consecutive indices
starting at 1, matching MLX's commands; the general MLX data model also permits index 0.
Call once per frame, including empty frames. No detections means an empty sequence,
not a skipped update. They assume equal time intervals; indices are not timestamps.

IDs are positive, unique across classes, and never reused within a session. Reset clears
tracks and restarts IDs. Returned snapshots are immutable. Lost tracks retain their last
observed confidence/label and `last_seen_frame`; predictions must not be presented as new
measurements. MLX persists only confirmed tracks observed in the current frame.

If an update fails, stop and reset before reusing the session. Neither MLX's command frame
counter nor arbitrary backend updates are transactional.

## Working with the current MLX boundaries

The following points reflect the tracking structure reviewed for these guides. They are
important when you move beyond the basic CLI example. They do not require a fork of MLX.

### Keep detector configuration separate from tracker configuration

`tracker_config` is passed only to the tracking algorithm. It cannot configure LibreYOLO.
MLX currently reconstructs detector configuration from selected `TrackingRequest` fields;
a separately supplied `adapter` is not forwarded through that path. Do not rely on adding
`adapter` to tracking configuration or `extras`. This does not affect self-contained
adapter detector checkpoints selected through `--model-path`.

For a separate detector adapter, construct the detector with an explicit
`ObjectDetectionRequest(adapter=...)`, then inject it through `RunTrackingVideo(detector=...)`
or `RunObjectDetectionTrackingCommand(detection_model=...)`. Keep the provider, model path,
confidence, image size, and device in the tracking request consistent with that detector:
replay metadata currently comes from the tracking request, not the injected object.
An `adapter` here is a detector feature adapter, not the tracking plugin class.

### Verify the plugin contract, not just imports

The current factory checks protocol membership, which does not validate full signatures.
The frame command also does not reject a returned frame index that differs from its input.
The tests in Step 6 therefore check keyword-call compatibility, result type, frame alignment,
positive unique IDs, and reset. Retain these tests when replacing the example algorithm.
Treat a successful Step 5 as a construction check, not a complete compatibility check.

If a plugin rejects an input or fails mid-update, stop and reset the session. Do not retry
that frame on the same command: its counter may already have advanced, and backend state
may have changed. A wrapper cannot promise rollback of arbitrary algorithm internals.

### Give each resource an explicit owner

In `run_observations.py`, the caller owns the command and resets it in `finally`.
The plugin owns track state; it does not open a camera, release caller frames, close a
writer, or manage a detector. Repeated `reset()` calls on these examples are safe.
Deep SORT reset discards session galleries while retaining reusable encoder weights.

The current video session resets tracking and releases sources/sinks during frame-loop
cleanup. Exceptions before that loop, such as an initial reporter failure, can bypass
cleanup. Dataset benchmarking also attempts to release sources/sinks after the session
has done so. When injecting resource-owning components today, make their `release()` /
`close()` operations idempotent, arrange outer `try/finally` cleanup for resources you
create, and preserve the original error if cleanup also fails. Do not assume reset frees
all model weights, or attach camera/storage ownership to the tracking algorithm.

### Choose the per-frame command for custom storage

`RunTrackingVideo` currently expects file artifacts and always exports replay. Its writer
parameter is not a general database/in-memory sink interface. For a database, queue, or
in-memory consumer, use the per-frame loop from Step 7 and pass each `TrackingFrameResult`
to your own storage component. Keep serialization out of the algorithm. If matching MLX's
saved-observation behavior, store only confirmed tracks whose `last_seen_frame` equals
`result.frame_index`; a lost prediction is not a new observation.

### Use a session factory for dataset benchmarks today

There is currently no `algorithm_factory` argument on `BenchmarkTrackingDataset`.
Use the `examples/session_factory.py` file from Step 3 through its existing `session_factory`
argument. It constructs a new tracker for each sequence. The Deep SORT version shares one
lazy encoder within a sequential benchmark while keeping track IDs and appearance galleries
separate. Do not share that encoder across concurrent workers without establishing its
thread safety; independent workers can each construct their own factory.

From the project directory, after installing MLX's `tracking` extra, the following is a
complete optional command. Replace the prepared dataset and detector checkpoint paths.
The dataset must already follow MLX's tracking-sequence manifest format. A directory of
arbitrary images is not sufficient. Select a fresh output directory.

```bash
python -m pip install -e '../mlx[tracking]'
python - <<'BENCHMARK'
from examples.session_factory import make_session_factory
from mlx.modes.object_detection.tracking.benchmark import BenchmarkTrackingDataset
from mlx.modes.object_detection.tracking.requests import TrackingBenchmarkRequest, TrackingRequest

request = TrackingBenchmarkRequest(
    dataset_path="/path/to/prepared-tracking-dataset",
    tracking=TrackingRequest(
        provider="libreyolo",
        model_path="/path/to/detector.pt",
        output_path="./benchmark-output",
        display=False,
    ),
    real_time_results=False,
)
result = BenchmarkTrackingDataset(
    request, session_factory=make_session_factory(),
).execute()
print(result.output_path)
BENCHMARK
```

The factory explicitly injects an algorithm, so it replaces the normal tracker/config
selection for those sessions. Change its constructor arguments to configure the benchmark;
`TrackingRequest.tracker_config` is not applied to an already constructed algorithm.
For a separately adapted detector, pass the explicitly constructed detector to
`BenchmarkTrackingDataset(detector=...)` as well. `real_time_results=False` avoids video
compilation/rendering but still writes the normal tracking, evaluation, and replay files.

### Compose replaceable algorithm components

The Kalman example accepts `filter_factory` in Python. The factory receives an initial
`xyxy` box and returns an object implementing `predict()` and `update(box)`, each returning
an `xyxy` box. This lets motion estimation change without rewriting association or track
lifecycle. JSON cannot contain this callable; inject it from Python or wrap your selected
factory in another external plugin class for CLI use.

The Deep SORT example already accepts an encoder and backend factory. Keep these integration
details behind `components/`; only `mlx_plugins/` should translate MLX types. Extract shared
lifecycle operations when two local algorithms need them, rather than introducing a common
tracker base class or duplicating the whole implementation.

## Refactor backlog for MLX maintainers

These are proposed MLX runtime changes, **not APIs implemented by this guide**. The files
above work with the current API and apply the improvements possible in an external plugin.

| Priority | Proposed change | Why it helps |
| --- | --- | --- |
| 1 | Compose a typed detector request instead of copying selected fields | Prevent supported detector options, including `adapter`, from being lost. |
| 2 | Give each resource one owner and protect initialization with a cleanup scope | Cover early failures and eliminate duplicate cleanup between session and benchmark. |
| 3 | Validate plugin callable compatibility and execution results at the boundary | Report invalid signatures/results with contextual `MLXUserError` messages. |
| 4 | Introduce a narrow result-sink contract and optional artifact post-processing | Support other storage destinations without forcing MOT files and replay. |
| 5 | Add an optional per-sequence tracker factory to benchmarking | Allow composed trackers without replacing session construction. |
| 6 | Inject motion filters and extract proven shared lifecycle operations in built-ins | Make motion replaceable while preserving distinct association algorithms. |

Preserve current CLI behavior, public imports, the existing registry, and default output
artifacts while making these changes. Add portable regression tests for each boundary.
Do not introduce a second generic `Track` dispatcher or force external plugins to inherit
MLX's internal tracking classes.

## Limits and troubleshooting

- Raising `iou_threshold` makes matching stricter; increasing `max_age` permits longer
  gaps but can increase incorrect reacquisition. `min_hits` trades output latency for
  suppression of one-frame false positives.
- IoU fails when objects jump farther than their predicted boxes overlap. Appearance,
  camera-motion compensation, or a different motion model may be necessary.
- Noise is fixed in pixel units, and elapsed time is one update. Resolution changes,
  variable frame intervals, and long occlusion are not handled automatically.
- `frame` is intentionally unused: the tracker needs only supplied detections. This
  makes offline observation replay possible without decoding images.

See also [the MLX tracker extension tutorial](../docs/tutorials/adding-a-tracker.md) and
[the canonical architecture](../ARCHITECTURE.md).
