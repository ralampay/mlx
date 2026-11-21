# Build an external Deep SORT plugin for MLX

**You do not need to fork MLX or add files inside its source tree.** In this guide you
create your own Python project, install it into the same environment as MLX, and tell
MLX the import path of your class. MLX already contains the plugin loader, tracking
commands, and output writers. Your project contains the custom tracking implementation.

```text
workspace/
├── mlx/                         # Existing MLX checkout; leave its source unchanged
└── custom-tracking-deep-sort/       # New, user-owned project created in this guide
```

You will build `DeepSortTracker` and select it with
`--tracker custom_tracking.mlx_plugins.deep_sort:DeepSortTracker`. The text before `:` is an importable
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
mkdir custom-tracking-deep-sort
cd custom-tracking-deep-sort
python -m venv .venv
source .venv/bin/activate
```

Keep this terminal open. Run the remaining commands from `custom-tracking-deep-sort/` with
this virtual environment activated. On another day, return to this directory and run
`source .venv/bin/activate` again before using the plugin.

## Step 2: Create the folders

```bash
mkdir -p src/custom_tracking/components src/custom_tracking/mlx_plugins configs tests examples
```

Once you finish Step 3, your project will contain:

```text
custom-tracking-deep-sort/
├── pyproject.toml
├── src/custom_tracking/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── observations.py
│   │   └── deep_sort.py
│   └── mlx_plugins/
│       ├── __init__.py
│       ├── boundary.py
│       └── deep_sort.py
├── configs/deep_sort.json
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

**File 1 of 13.** This tells pip how to install your package and its dependencies. MLX itself is installed separately in Step 4.

### File: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "custom-tracking-plugins"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy>=1.10", "deep-sort-realtime==1.3.2", "setuptools>=68,<81"]

[project.optional-dependencies]
test = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["src"]
```

**File 2 of 13.** Save this file exactly as shown.

### File: `src/custom_tracking/__init__.py`

```python
"""Standalone tracking plugin package."""
```

**File 3 of 13.** Save this file exactly as shown.

### File: `src/custom_tracking/components/__init__.py`

```python
"""Standalone tracking plugin package."""
```

**File 4 of 13.** Save this file exactly as shown.

### File: `src/custom_tracking/mlx_plugins/__init__.py`

```python
"""Standalone tracking plugin package."""
```

**File 5 of 13.** Internal observations and snapshots keep the tracking components independent of MLX.

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

**File 6 of 13.** This is the shared conversion boundary between your components and MLX data types.

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

**File 7 of 13.** This component isolates Deep SORT and shares one appearance encoder across class-specific trackers.

### File: `src/custom_tracking/components/deep_sort.py`

```python
from dataclasses import replace
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from custom_tracking.components.observations import Snapshot

class MobileNetEncoder:
    """One lazily loaded encoder shared by all class-specific trackers."""
    def __init__(self, *, gpu=False, weights=None):
        self.gpu, self.weights, self.model = gpu, weights, None

    def __call__(self, crops):
        if self.model is None:
            from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
            self.model = MobileNetv2_Embedder(
                gpu=self.gpu, half=False, bgr=True, model_wts_path=self.weights,
            )
        return self.model.predict(crops)

class DeepSortBackend:
    def __init__(self, *, encoder, max_age=30, n_init=3, nn_budget=100,
                 max_cosine_distance=0.2, backend_factory=DeepSort):
        for name, value, minimum in (("max_age", max_age, 0), ("n_init", n_init, 2),
                                     ("nn_budget", nn_budget, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not 0 < max_cosine_distance <= 2:
            raise ValueError("max_cosine_distance must be in (0, 2].")
        self.encoder, self.factory = encoder, backend_factory
        self.options = dict(max_age=max_age, n_init=n_init, nn_budget=nn_budget,
                            max_cosine_distance=max_cosine_distance,
                            embedder=None, nms_max_overlap=1.0)
        self.reset()

    def reset(self):
        self.groups = {}
        self.previous = {}
        self.next_id = 1

    def update(self, index, detections, frame=None):
        crops = []
        prepared = []
        if detections:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
                raise ValueError("Deep SORT requires a nonempty uint8 BGR frame for detections.")
            height, width = frame.shape[:2]
            for detection in detections:
                x1, y1, x2, y2 = detection.box
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Detection is entirely outside the frame; inspect detector coordinates.")
                detection = replace(detection, box=(float(x1), float(y1), float(x2), float(y2)))
                prepared.append(detection)
                crops.append(frame[int(np.floor(y1)):int(np.ceil(y2)),
                                   int(np.floor(x1)):int(np.ceil(x2))])
        embeddings = np.asarray(self.encoder(crops), dtype=float) if crops else np.empty((0, 0))
        if crops:
            if (embeddings.ndim != 2 or len(embeddings) != len(crops)
                    or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()
                    or np.any(np.linalg.norm(embeddings, axis=1) == 0)):
                raise ValueError("Encoder must return one finite nonzero feature vector per crop.")
        grouped = {}
        for detection, embedding in zip(prepared, embeddings):
            grouped.setdefault(detection.class_id, []).append((detection, embedding))
        for class_id in grouped:
            if class_id not in self.groups:
                self.groups[class_id] = self.factory(**self.options)
        current = {}
        # Advance even classes absent from this frame so tracks age and expire.
        for class_id, backend in list(self.groups.items()):
            pairs = grouped.get(class_id, [])
            raw, features, metadata = [], [], []
            for detection, embedding in pairs:
                x1, y1, x2, y2 = detection.box
                raw.append(([x1, y1, x2-x1, y2-y1], detection.score, str(class_id)))
                features.append(embedding)
                metadata.append(detection)
            tracks = backend.update_tracks(raw, embeds=features, others=metadata)
            for track in tracks:
                key = (class_id, track.track_id)
                previous = self.previous.get(key)
                if previous is None:
                    identity = self.next_id
                    self.next_id += 1
                else:
                    identity = previous.identity
                missing = track.time_since_update
                observation = track.get_det_supplementary() if missing == 0 else previous.observation
                observation = replace(observation, box=tuple(float(v) for v in track.to_ltrb()))
                current[key] = Snapshot(identity, observation, track.hits, missing,
                                        index if missing == 0 else previous.last_seen,
                                        track.is_confirmed())
            if not tracks:
                del self.groups[class_id]
        self.previous = current
        return tuple(sorted(current.values(), key=lambda s: s.identity))
```

**File 8 of 13.** This is the `DeepSortTracker` class MLX will import. Its constructor accepts the algorithm-specific settings in the JSON file.

### File: `src/custom_tracking/mlx_plugins/deep_sort.py`

```python
from mlx.core.exceptions import MLXUserError
from custom_tracking.components.deep_sort import DeepSortBackend, MobileNetEncoder
from custom_tracking.mlx_plugins.boundary import observations, frame_result, validate_index

class DeepSortTracker:
    def __init__(self, *, max_age=30, n_init=3, nn_budget=100,
                 max_cosine_distance=0.2, embedder_gpu=False, embedder_weights=None,
                 encoder=None, backend_factory=None):
        if type(embedder_gpu) is not bool:
            raise ValueError("embedder_gpu must be a JSON boolean.")
        encoder = encoder if encoder is not None else MobileNetEncoder(
            gpu=embedder_gpu, weights=embedder_weights,
        )
        injected = {} if backend_factory is None else {"backend_factory": backend_factory}
        self.core = DeepSortBackend(
            encoder=encoder, max_age=max_age, n_init=n_init, nn_budget=nn_budget,
            max_cosine_distance=max_cosine_distance, **injected,
        )
        self.index = 0

    def update(self, *, frame_index, detections, frame=None):
        try:
            validate_index(frame_index, self.index)
            result = self.core.update(frame_index, observations(detections), frame)
            self.index = frame_index
            return frame_result(frame_index, result)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, AssertionError) as exc:
            raise MLXUserError(
                f"Deep SORT failed: {exc}. Inspect frame format, encoder weights, "
                "and the pinned deep-sort-realtime / PyTorch installation."
            ) from exc

    def reset(self):
        self.core.reset()
        self.index = 0
```

**File 9 of 13.** These keys are passed to your tracker constructor. Do not put detector paths or video paths here; those are MLX workflow arguments.

### File: `configs/deep_sort.json`

```json
{
  "max_age": 30,
  "n_init": 3,
  "nn_budget": 100,
  "max_cosine_distance": 0.2,
  "embedder_gpu": false
}
```

**File 10 of 13.** These synthetic tests exercise identities, missed observations, class isolation, and reset.

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
from custom_tracking.mlx_plugins.deep_sort import DeepSortTracker

class ConstantEncoder:
    def __call__(self, crops):
        return np.tile([1., 0., 0.], (len(crops), 1))

def make_tracker(**options):
    return DeepSortTracker(n_init=2, max_age=1, encoder=ConstantEncoder(), **options)

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

def test_encoder_is_shared_skips_empty_frames_and_survives_reset():
    class CountingEncoder(ConstantEncoder):
        def __init__(self):
            self.batch_sizes = []
        def __call__(self, crops):
            self.batch_sizes.append(len(crops))
            return super().__call__(crops)
    encoder = CountingEncoder()
    tracker = DeepSortTracker(encoder=encoder)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    tracker.update(frame_index=1, detections=[detection(0), detection(1)], frame=frame)
    tracker.update(frame_index=2, detections=[])
    tracker.reset()
    tracker.update(frame_index=1, detections=[detection()], frame=frame)
    assert encoder.batch_sizes == [2, 1]

def test_requires_image_for_appearance_but_not_for_empty_frames():
    tracker = make_tracker()
    with pytest.raises(MLXUserError, match="BGR"):
        tracker.update(frame_index=1, detections=[detection()])
    assert tracker.update(frame_index=1, detections=[]).tracks == ()

def test_rejects_invalid_embeddings():
    tracker = DeepSortTracker(encoder=lambda crops: [[0., 0.] for _ in crops])
    with pytest.raises(MLXUserError, match="nonzero"):
        tracker.update(frame_index=1, detections=[detection()],
                       frame=np.zeros((32, 32, 3), dtype=np.uint8))

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
```

**File 11 of 13.** This complete script runs the plugin on synthetic observations without loading a detector. Run it in Step 7.

### File: `examples/run_observations.py`

```python
from pathlib import Path
import numpy as np
from mlx.modes.object_detection.tracking import (
    BoundingBox, TrackingDetection, CreateTrackingAlgorithm, RunTrackByDetectionCommand,
)

algorithm = CreateTrackingAlgorithm(
    tracker="custom_tracking.mlx_plugins.deep_sort:DeepSortTracker",
    config_path=str(Path(__file__).resolve().parents[1] / "configs/deep_sort.json"),
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

**File 12 of 13.** This optional-use factory lets dataset benchmarks construct a fresh tracker per sequence. Save it now; it will be exercised by the tests even if you do not run a dataset benchmark.

### File: `examples/session_factory.py`

```python
from mlx.modes.object_detection.tracking import RunTrackingVideo
from custom_tracking.components.deep_sort import MobileNetEncoder
from custom_tracking.mlx_plugins.deep_sort import DeepSortTracker

def make_session_factory():
    # Shared weights, but fresh identities and appearance history per sequence.
    encoder = MobileNetEncoder(gpu=False)

    def create_session(request, **collaborators):
        algorithm = DeepSortTracker(encoder=encoder, n_init=3, nn_budget=100)
        return RunTrackingVideo(request, algorithm=algorithm, **collaborators)

    return create_session
```

**File 13 of 13.** This makes the example helpers importable from the project directory for the benchmark command and tests.

### File: `examples/__init__.py`

```python
"""Runnable examples and composition helpers for this standalone project."""
```

## Step 4: Install MLX and your plugin into the same environment

You are still in `custom-tracking-deep-sort/`, and `.venv` is active. With the sibling
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
    tracker="custom_tracking.mlx_plugins.deep_sort:DeepSortTracker",
    config_path="configs/deep_sort.json",
).execute()
print(type(tracker).__name__)
tracker.reset()
PYTHON
```

Expected output includes `DeepSortTracker`. This proves the class can be imported and
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

Expected result: **9 passed**. These tests require no detector checkpoint, dataset,
GPU, or video. They inject deterministic features into the real pinned Deep SORT backend, so they do not load neural weights.
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

This step loads the bundled MobileNet weights and runs the actual appearance encoder on CPU. The black synthetic image verifies execution, not appearance accuracy. The tests in Step 6 deliberately avoid this heavyweight path.

To adapt the script to your application, replace the synthetic observations with your
own observations and call `command.execute(...)` once per frame, including empty frames.
Create the algorithm **once before the loop** so identities persist. The script resets
it in `finally` when the session ends. Detection acquisition remains separate from tracking.

## Step 8: Track a real video through MLX

You now need an existing supported LibreYOLO `.pt` checkpoint and a video. The synthetic
steps do not create either file. In the command below, replace `/path/to/detector.pt`
and `/path/to/video.mp4` with your actual paths. Run from `custom-tracking-deep-sort/`:

```bash
python -m mlx --mode track --action run \
  --provider libreyolo --model-path /path/to/detector.pt \
  --file-path /path/to/video.mp4 --device cpu --confidence 0.25 \
  --tracker custom_tracking.mlx_plugins.deep_sort:DeepSortTracker \
  --tracker-config configs/deep_sort.json \
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

Edit files under your project's `src/custom_tracking/`, adjust `configs/deep_sort.json`,
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

This guide wraps `deep-sort-realtime==1.3.2` with a custom `DeepSortTracker`.
Deep SORT combines Kalman motion prediction, appearance descriptors, motion gating,
and association. Appearance can help through crossings and occlusion, but encoder quality
and domain matter. This is an adapter tutorial, not a claim to reproduce paper benchmarks.
See the [paper](https://arxiv.org/abs/1703.07402) and
[backend source](https://github.com/levan92/deep_sort_realtime).

The default is a shared, lazily loaded MobileNet encoder using BGR pixels, CPU, and
full precision. Version 1.3.2 ships MobileNet weights; `embedder_weights` can select an
explicit compatible checkpoint. General ImageNet descriptors are not automatically a
person- or vehicle-reidentification model. Replace the injected encoder for your domain.
The package uses legacy `pkg_resources` in its encoder, so this project also constrains
its runtime setuptools dependency below 81. The class-specific backends use `embedder=None` and receive precomputed features, so
there is only one encoder. `nn_budget` bounds per-identity appearance history.

### Backend policy

A separate Deep SORT instance per class prevents cross-class matches, which passing a
class label to the backend alone does not guarantee. Local backend IDs are remapped to
session-wide IDs; expired mappings and empty class pools are removed. All pools advance
on every frame. Reset discards the pools and their appearance galleries but retains the
encoder object and its loaded weights.

`n_init` is at least 2 in this example, reflecting the backend's confirmation-on-update
behavior. Tentative tracks can be deleted after one miss. Confirmed tracks survive up to
`max_age` missed frames. Boxes are clipped before cropping; fully out-of-frame boxes are
rejected. Empty frames need no image and never invoke the encoder. With detections, even
an injected encoder receives crops from a nonempty uint8 BGR image.

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

- Set `embedder_gpu: true` only when using a compatible CUDA/PyTorch installation.
  This is separate from MLX's detector `--device`. The backend may fall back to CPU.
- Missing weights, torchvision/PyTorch incompatibility, and invalid frame shape are
  reported at the plugin boundary. Verify the pinned package and your matching PyTorch
  and torchvision installation; do not hide these errors with empty tracking results.
- For a real-encoder smoke test, run the direct Python example above (it loads the bundled
  encoder), then the CLI on a short real video. Synthetic black crops only test execution.
  Record whether CPU encoding succeeds; no accuracy claim follows from that smoke test.
- Supply a callable `encoder(crops) -> finite nonzero feature vectors` to swap appearance
  models. It must be stateless across sequences, apart from reusable weights. A learned
  stateful encoder would need its own explicit reset lifecycle.
- Deep SORT uses pixel features; there is no detector-free appearance signal in a box
  alone. Empty updates are supported, but nonempty observations need frames here.
- No camera-motion compensation or cross-camera identity persistence is implemented.

See also [the MLX tracker extension tutorial](../docs/tutorials/adding-a-tracker.md) and
[the canonical architecture](../ARCHITECTURE.md).
