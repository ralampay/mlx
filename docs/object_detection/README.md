# Object Detection

Mode: `object-detection`

Package: `mlx.modes.object_detection`

## Overview

This mode provides provider-backed object-detection workflows. Ultralytics is the
default provider and LibreYOLO is selected explicitly:

```bash
python -m mlx --mode object-detection --provider ultralytics
python -m mlx --mode object-detection --provider libreyolo
```

The repository-level `requirements.txt` installs both providers. Create and activate
the virtual environment by following the main [installation guide](../../README.md#installation),
then run:

```bash
python -m pip install -r requirements.txt
```

Package consumers installing MLX directly may instead install one provider or both:

```bash
python -m pip install ".[object-detection-ultralytics]"
python -m pip install ".[object-detection-libreyolo]"
python -m pip install ".[object-detection]"
```

The LibreYOLO extra follows the `release` branch of
[`ralampay/libreyolo`](https://github.com/ralampay/libreyolo). It does not install
LibreYOLO from PyPI or the upstream repository.

The neutral source is split by responsibility:

- `runner.py`: action dispatch and CLI presentation wiring.
- `commands.py`: provider-neutral training, model creation, conversion, listing, and streaming.
- `models.py`: normalized detection records and the detector protocol.
- `providers.py`: lazy provider registry and provider protocol.
- `streaming.py`: frame-source and frame-sink ports plus OpenCV adapters.
- `artifacts.py`: provider-shared checkpoint discovery and ONNX destination rules.
- `libreyolo/`: Ralampay LibreYOLO training, conversion, model resolution, and decoding.
- `ultralytics/`: Ultralytics training, conversion, model resolution, and decoding.
- `tracking/`: detector-neutral online tracking types, protocol, and per-frame command.

## Generic Tracking by Detection

The first tracking layer keeps frame acquisition, detection, and tracking separate:

```text
video or camera source
    ↓
detection adapter
    ↓
normalized detections
    ↓
tracking command
    ↓
immutable per-frame track results
```

This boundary lets the same tracking API work with different detector formats, video
sources, and future association algorithms. A tracking algorithm receives detections
that have already been computed; it never invokes YOLO itself.

The included `DetectionAsTrackAlgorithm` is only an executable skeleton. It assigns a
new, confirmed ID to every detection and deliberately performs no temporal
association. The integration command accepts the same detection adapter returned by
the current `build_detection_adapter(...)` factory, including both the Ultralytics
`.pt` and ONNX Runtime `.onnx` implementations:

```python
from mlx.modes.object_detection.tracking.algorithms import DetectionAsTrackAlgorithm
from mlx.modes.object_detection import RunObjectDetectionTrackingCommand

# `detector` is the existing adapter created by build_detection_adapter(...).
command = RunObjectDetectionTrackingCommand(
    detection_model=detector,
    algorithm=DetectionAsTrackAlgorithm(),
)

while True:
    ok, frame = capture.read()
    if not ok:
        break

    tracking_result = command.execute(frame=frame)
    write_or_render(tracking_result)

command.reset()  # Call before processing a new video or camera session.
```

For callers that already perform detection separately,
`RunTrackByDetectionCommand.execute(detections=..., frame=...)` remains available as
the lower-level detector-neutral API.

The conversion is kept at the neutral detection/tracking boundary. `Detection` uses
integer `xyxy` coordinates, while generic tracking state uses
immutable `BoundingBox` values backed by ordinary Python floats and does not retain
an Ultralytics result object.

Future algorithms implement the structural `TrackingAlgorithm` protocol without
inheriting from a project base class:

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
        # Future class-aware association logic belongs here.
        ...

    def reset(self) -> None:
        ...
```

The command retains only its current frame index, and algorithms must retain only
active or temporarily lost track state. Frames, detector result objects, full
trajectories, and prior `TrackingFrameResult` objects are not stored, so tracking
memory stays proportional to active tracks plus current-frame detections rather than
video duration. Applications that need session history should stream each result to
disk or another external store as it is produced.

## Dataset Format

Both providers accept any of these dataset sources:

- a local YOLO dataset root containing `data.yaml`
- a direct dataset YAML path
- a provider-supplied dataset alias such as `coco8` or `coco128`

For this repository, `coco8` is the best default example dataset. It is small, is
available through both providers, and is fast enough for smoke-testing `yolo26`,
both DraxNet YOLO26 variants, and `yolo9-t`.

If you want a slightly less trivial quick-start dataset, use `coco128`. For real training, point `--dataset` to your own YOLO-format dataset root.

Local dataset example:

Minimal example:

```yaml
path: ../
train: images/train
val: images/val
names:
  0: class-a
  1: class-b
```

Pass `--dataset-path` as the directory that contains `data.yaml`, or pass `--dataset coco8` / `--dataset coco128` to use a built-in dataset YAML.

### Expected Local Directory Structure

When `--dataset` points to a local directory, `mlx` expects that directory itself to be the YOLO dataset root. In practice, this means `data.yaml` must live directly inside the path you pass on the command line.

Example:

```text
my-detection-dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/          # optional
└── labels/
    ├── train/
    ├── val/
    └── test/          # optional
```

Typical command:

```bash
python -m mlx \
    --mode object-detection \
    --action train \
    --dataset /path/to/my-detection-dataset \
    --model draxnet-yolo26 \
    --epochs 100 \
    --batch-size 16 \
    --device cuda:0
```

That command assumes:

- `/path/to/my-detection-dataset/data.yaml` exists
- `data.yaml` references paths relative to that dataset root, such as `images/train` and `images/val`
- each image in `images/<split>/` has a matching YOLO label file in `labels/<split>/`

Example `data.yaml` for that layout:

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: class-a
  1: class-b
```

The important rule is simple: pass the directory that contains `data.yaml`, not the `images/` directory and not the `labels/` directory.

## Model Selection

With `--provider ultralytics`, `--model` accepts a YAML path or one of these aliases:

- `yolo26`
- `yolov26`
- `draxnet-ave-yolo26`
- `draxnet-sknet-yolo26`
- `draxnet-yolo26`

The two explicit DraxNet aliases select fixed-average and SKNet-style adaptive fusion,
respectively. The legacy `draxnet-yolo26` alias remains supported and maps to
`draxnet-ave-yolo26`.

With `--provider libreyolo`, first-class training and listing use these aliases:

- `yolo9-t`
- `yolo9-s`
- `yolo9-m`
- `yolo9-c`
- `yolo9-s-drax-b5`

LibreYOLO inference and conversion load the local artifact supplied through
`--model-path`. Other axis-aligned detection checkpoints understood by the fork may
work through this passthrough path, but only the configurations above are part of
MLX's tested training and listing surface. `yolo9-s-drax-b5` matches LibreYOLO's first
supported Drax experiment: size S, Drax at B5, attention and efficient mode enabled,
average fusion, and zero drop path.

List the canonical project models and their total parameter counts:

```bash
python -m mlx --mode object-detection --action ls-models
python -m mlx --mode object-detection --provider libreyolo --action ls-models
```

This action constructs `yolo26`, `draxnet-ave-yolo26`, and `draxnet-sknet-yolo26`
from their architecture YAML files without loading pretrained weights.

If a DraxNet model YAML is missing, update the Ultralytics dependency from the
`ralampay/ultralytics` fork. Installed distributions of the fork include both DraxNet `.yaml`
files under `ultralytics/cfg/models/ext/`.

## End-to-End Workflow

The typical object-detection deployment path in this repository is:

1. Train with Ultralytics using a `.yaml` model definition and produce a `.pt` checkpoint.
2. Convert that trained `.pt` checkpoint to `.onnx`.
3. Run camera or video inference from the exported `.onnx` model through ONNX Runtime.

The same action sequence works with LibreYOLO by adding `--provider libreyolo` to
every command and using `--model yolo9-t` for training. Provider artifacts are not
interchangeable, so the provider flag must remain consistent across the workflow.

Concrete example:

### Step 1: Train with Ultralytics

```bash
python -m mlx \
    --mode object-detection \
    --action train \
    --dataset coco8 \
    --model draxnet-yolo26 \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0 \
    --output ./runs/draxnet-yolo26
```

Expected result:

- Ultralytics writes a training run under `./runs/draxnet-yolo26/...`
- the trained checkpoint is typically available at `./runs/draxnet-yolo26/<run-name>/weights/best.pt`

### Step 2: Convert the trained checkpoint to ONNX

```bash
python -m mlx \
    --mode object-detection \
    --action convert \
    --model-path ./runs/draxnet-yolo26/exp/weights/best.pt \
    --output ./exports \
    --device cpu
```

Expected result:

- MLX loads the Ultralytics `.pt` checkpoint
- Ultralytics exports an ONNX model
- the final ONNX file is written to `./exports/best.onnx`

### Step 3: Use the exported ONNX model for inference

Camera inference:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-camera \
    --model-path ./exports/best.onnx \
    --device cpu \
    --confidence 0.35 \
    --camera-index 0
```

Video inference:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-video \
    --model-path ./exports/best.onnx \
    --file-path ~/videos/sample.mp4 \
    --device cpu \
    --confidence 0.35
```

Important notes for this workflow:

- `--model` is required for training because Ultralytics needs the model YAML or alias.
- `--model` is not required for `.onnx` inference; MLX switches to ONNX Runtime when `--model-path` ends in `.onnx`.
- if you want the ONNX export beside the checkpoint instead of under `./exports`, omit `--output`.

## Training

### LibreYOLO YOLOv9

```bash
python -m mlx \
    --mode object-detection \
    --provider libreyolo \
    --action train \
    --dataset coco8 \
    --model yolo9-t \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0 \
    --output ./runs/libreyolo-yolo9
```

For LibreYOLO, `--model-path` is an optional local warm-start/resume checkpoint and
`--pretrained` requests the matching published initialization from the fork.
`--loss-clip` is not supported for LibreYOLO YOLOv9 and fails with a user-facing
message instead of being ignored.

Use `--model yolo9-s-drax-b5` to train the Drax-enabled configuration documented by
the fork. MLX passes the same Drax configuration used by `ls-models`, so the listed
parameter count represents the architecture selected for scratch training.

### Ultralytics

Baseline YOLO26 example:

```bash
python -m mlx \
    --mode object-detection \
    --action train \
    --dataset coco8 \
    --model yolo26 \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0 \
    --output ./runs/yolo26
```

DraxNet-backed YOLO26 example:

```bash
python -m mlx \
    --mode object-detection \
    --action train \
    --dataset coco8 \
    --model draxnet-yolo26 \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0 \
    --output ./runs/draxnet-yolo26
```

Important arguments:

- `--dataset` / `--dataset-path`: required dataset source. Use `coco8` for the documented smoke-test path.
- `--model`: required architecture YAML or alias such as `yolo26`, `draxnet-ave-yolo26`,
  or `draxnet-sknet-yolo26`.
- `--model-path`: optional checkpoint for warm-start training.
- `--epochs`, `--batch-size`, `--device`: core training controls.
- `--use-best` / `--no-use-best`: select `weights/best.pt` after training by default. Use `--no-use-best` to prefer `weights/last.pt` instead.
- `--pretrained`: enable Ultralytics pretrained initialization.
- `--lr0`, `--optimizer`, `--nbs`, `--warmup-epochs`, `--loss-clip`, `--amp`: trainer overrides.
- `--output`: optional Ultralytics project directory. If omitted for a local dataset root, runs go under `<dataset>/runs`. Otherwise they default to `./runs/object_detection`.
- `--run-name`: output folder name inside the Ultralytics project directory.

If `--model-path` is not provided and the chosen `--output` directory already contains checkpoints, MLX now reuses them automatically. It resumes from the newest `last.pt` it finds under that output tree; if only other `.pt` files are present, it warm-starts from the newest one instead of starting from random initialization.

When a resumable `last.pt` is found, the trainer explicitly reports that it is continuing the existing run and shows the checkpoint path. The training summary also marks the run as `continue existing run` instead of `new run`.

After training completes, MLX selects the checkpoint to use downstream. By default `--use-best` is enabled, so object detection selects `weights/best.pt` when Ultralytics writes it, stores that path in the returned training result as `model_path` / `checkpoint_path`, and prints the selected checkpoint. If `--no-use-best` is passed, MLX prefers `weights/last.pt`.

After training completes, MLX now prints a final validation-metrics table when Ultralytics exposes the values. For detection runs, the most important metrics are typically precision, recall, `mAP@0.50`, `mAP@0.50:0.95`, and fitness; train/validation loss terms may also appear. ROC/AUC is only shown when the underlying metrics object reports it, which is uncommon for standard object-detection validation.

MLX also now writes extra training graphs into the resolved run directory, alongside Ultralytics artifacts. When available, this includes:

- `loss_curves.png`: train/validation loss curves from `results.csv`
- `detection_metrics.png`: precision, recall, F1, `mAP@0.50`, and `mAP@0.50:0.95`
- `learning_rate.png`: learning-rate schedule curves
- `speed_metrics.png`: per-epoch speed metrics if Ultralytics records them
- `per_class_map.csv`: per-class `mAP@0.50` and `mAP@0.50:0.95` values
- `per_class_map50.csv`: per-class `mAP@0.50` values
- `per_class_map50.png`: bar chart of per-class `mAP@0.50`
- `per_class_map50_95.csv`: per-class `mAP@0.50:0.95` values
- `per_class_map50_95.png`: bar chart of per-class `mAP@0.50:0.95`

Ultralytics plotting is also explicitly enabled for training runs, so native artifacts such as PR/F1/P/R curves and confusion-matrix plots should continue to land in the same run directory when supported by the installed Ultralytics version.

## Webcam Inference

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-camera \
    --model-path ./exports/best.onnx \
    --device cpu \
    --confidence 0.35 \
    --camera-index 0
```

Important arguments:

- `--model`: required only when `--model-path` points to a PyTorch checkpoint (`.pt`).
- `--model-path`: required trained checkpoint or exported `.onnx` model.
- `--confidence`: minimum confidence threshold to render.
- `--camera-index`: OpenCV camera device index.

## Video Inference

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-video \
    --model-path ./exports/best.onnx \
    --file-path ~/videos/sample.mp4 \
    --device cpu \
    --confidence 0.35
```

Additional arguments:

- `--file-path`: required video path.
- `--device`: inference backend. `.pt` uses Ultralytics; `.onnx` uses ONNX Runtime.
- `--confidence`: rendered detection threshold.

## ONNX Conversion

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action convert \
    --model-path ./runs/draxnet-yolo26/exp/weights/best.pt \
    --output ./exports \
    --device cpu
```

Important arguments:

- `--model-path`: required Ultralytics PyTorch checkpoint (`.pt`) to export.
- `--output`: optional destination directory or explicit `.onnx` file path. If omitted, MLX writes the ONNX file beside the checkpoint.
- `--height`, `--width`: optional export image size. If equal, MLX passes a square `imgsz`; otherwise it passes the `(height, width)` tuple to Ultralytics.
- `--device`: export backend such as `cpu` or `cuda:0`.

LibreYOLO conversion uses the same flags:

```bash
python -m mlx \
    --mode object-detection \
    --provider libreyolo \
    --action convert \
    --model-path ./runs/libreyolo-yolo9/mlx-libreyolo/weights/best.pt \
    --output ./exports/libreyolo
```

## Dependencies

- `object-detection-ultralytics` installs `ultralytics` from the
  `ralampay/ultralytics` fork plus ONNX Runtime.
- `object-detection-libreyolo` installs `libreyolo[onnx]` from the `release` branch
  of `ralampay/libreyolo`.
- `object-detection` installs both providers.
- `opencv-python` supplies webcam/video input and display for either provider.

The LibreYOLO fork requires Python 3.10 or newer, which is also the minimum supported
Python version for MLX.

Run `python -m mlx --help` for the full CLI reference.
