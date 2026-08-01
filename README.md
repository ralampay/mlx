# MLX

MLX is a command-line toolkit for machine-learning workflows. It provides a shared
CLI and project conventions while keeping object detection, image classification,
segmentation, NLP, and tracking logic in focused modules.

## Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Command-line interface](#command-line-interface)
- [Object detection and tracking](#object-detection-and-tracking)
- [Image classification](#image-classification)
- [Segmentation](#segmentation)
- [NLP embeddings](#nlp-embeddings)
- [Documentation](#documentation)

## Architecture

The CLI dispatches each `--mode` directly to its package. Shared exceptions, random
seed handling, model summaries, and terminal presentation utilities live in
`mlx.core`; mode-specific data preparation, models, training, inference, evaluation,
and presentation remain under `mlx.modes`.

```text
mlx/
├── core/                              shared infrastructure
└── modes/
    ├── object_detection/
    │   ├── ultralytics/               detection training and inference
    │   └── tracking/                  detector-neutral online tracking
    ├── image_classification/          standard and one-shot classification
    ├── segmentation/                  semantic segmentation
    └── nlp/                           CSV embedding workflows
```

Each workflow follows the same project pattern:

```text
CLI configuration
    ↓
thin mode runner
    ↓
command-style workflow
    ↓
mode-specific models, data, and presentation
```

### Module map

| Module | Package | Primary interface | Detailed documentation |
| --- | --- | --- | --- |
| Shared core | `mlx.core` | Exceptions, UI, seeds, model summaries | This README |
| Object detection | `mlx.modes.object_detection.ultralytics` | CLI and detection adapters | [Object detection](./docs/object_detection/README.md) |
| Tracking | `mlx.modes.object_detection.tracking` | Python API | [Tracking](./TRACKING.md) |
| Image classification | `mlx.modes.image_classification` | CLI and Python workflows | [Image classification](./docs/image_classification/README.md) |
| Segmentation | `mlx.modes.segmentation` | CLI and Python workflows | [Segmentation](./docs/segmentation/README.md) |
| NLP embeddings | `mlx.modes.nlp` | CLI | This README |

## Installation

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependency set includes NumPy, pandas, PyTorch, torchvision, OpenCV, Rich,
scikit-learn, ONNX Runtime, llama-cpp-python, and the repository's pinned
Ultralytics fork.

## Command-line interface

All CLI modules share one command shape:

```bash
python -m mlx --mode <mode> --action <action> [options]
```

Available CLI modes are:

| Mode | Actions |
| --- | --- |
| `object_detection` | `train`, `infer-camera`, `infer-video`, `convert`, `ls-models` |
| `image_classification` | `train`, `test`, `benchmark`, `infer-image`, `cam`, `build-dataset`, `ls-models` |
| `segmentation` | `train`, `test`, `benchmark`, `infer-image`, `infer-camera`, `infer-video`, `build-dataset`, `ls-models` |
| `nlp` | `embed` |

Hyphenated mode names such as `object-detection` and `image-classification` are also
accepted. Run the following command for the complete option reference:

```bash
python -m mlx --help
```

## Object detection and tracking

### Object detection

Package: `mlx.modes.object_detection.ultralytics`

The object-detection module trains Ultralytics models, converts PyTorch checkpoints
to ONNX, and runs camera or video inference through normalized detection adapters.
Model aliases include `yolo26`, `yolov26`, and `draxnet-yolo26`. Dataset input may
be a local YOLO dataset root, a dataset YAML, or an Ultralytics alias such as `coco8`
or `coco128`.

```bash
# Inspect available models.
python -m mlx --mode object_detection --action ls-models

# Train an Ultralytics model.
python -m mlx --mode object_detection --action train \
    --dataset coco8 \
    --model draxnet-yolo26 \
    --output ./runs/draxnet

# Convert the selected checkpoint to ONNX.
python -m mlx --mode object_detection --action convert \
    --model-path ./runs/draxnet/exp/weights/best.pt \
    --output ./exports

# Run video inference with the exported model.
python -m mlx --mode object_detection --action infer-video \
    --model-path ./exports/best.onnx \
    --file-path ~/videos/sample.mp4
```

Training reuses compatible checkpoints found under `--output` when no explicit
`--model-path` is supplied. By default, the best checkpoint is selected for
downstream use; pass `--no-use-best` to prefer the last checkpoint.

The intended deployment flow is:

```text
Ultralytics training → .pt checkpoint → ONNX conversion → camera/video inference
```

See [the object-detection guide](./docs/object_detection/README.md) for dataset
layout, model resolution, training artifacts, inference, and conversion details.

### Tracking by detection

Package: `mlx.modes.object_detection.tracking`

Tracking is a Python API layered on normalized object detections rather than a
separate CLI mode. The integrated command accepts the detector produced by the
current object-detection adapter factory:

```python
from mlx.modes.object_detection.tracking.algorithms import DetectionAsTrackAlgorithm
from mlx.modes.object_detection.ultralytics import RunObjectDetectionTrackingCommand

tracking = RunObjectDetectionTrackingCommand(
    detection_model=detector,
    algorithm=DetectionAsTrackAlgorithm(),
)

while True:
    ok, frame = capture.read()
    if not ok:
        break
    tracking_result = tracking.execute(frame=frame)

tracking.reset()
```

Detector-specific inference and generic tracking remain separate:

```text
frame source → detection adapter → normalized detections
             → tracking algorithm → immutable track results
```

The included algorithm is an architectural placeholder, not a temporal association
tracker. See [TRACKING.md](./TRACKING.md) for the complete integration flow, public
types, memory guarantees, lower-level API, and extension protocol.

## Image classification

Package: `mlx.modes.image_classification`

This module supports standard classifiers and Siamese one-shot models. Standard
families include ResNet, DenseNet, MobileNet, EfficientNet, ConvNeXt, DraxNet, and
Drax MobileNet variants. Its workflows cover dataset construction, training,
checkpoint resume, benchmarking, image inference, and CAM visualization.
Standard classifiers can optionally add a joint Deep SVDD projection head for
validation-calibrated out-of-distribution rejection. This is opt-in with
`--ood-method deep-svdd`; ordinary classification remains the default.

```bash
# List supported models.
python -m mlx --mode image_classification --action ls-models

# Train a standard classifier.
python -m mlx --mode image_classification --action train \
    --model resnet18 \
    --dataset ./dataset \
    --output ./artifacts/resnet18 \
    --seed 42

# Train a Siamese one-shot model.
python -m mlx --mode image_classification --action train \
    --model siamese-le-net \
    --dataset ./omniglot \
    --output ./artifacts/siamese \
    --seed 42

# Jointly train classification and Deep SVDD OOD rejection.
python -m mlx --mode image_classification --action train \
    --model resnet18 \
    --dataset ./dataset \
    --output ./artifacts/resnet18-svdd \
    --ood-method deep-svdd \
    --svdd-weight 0.05 \
    --svdd-quantile 0.95

# Generate class-activation maps.
python -m mlx --mode image_classification --action cam \
    --model resnet18 \
    --model-path ./artifacts/resnet18/resnet18.pth \
    --dataset ./dataset \
    --output ./cam-results \
    --cam-method gradcam
```

Training writes the selected `{model}.pth`, resumable `{model}.last.pth`, and
`training.csv` into the artifact directory. Benchmarking can export aggregate and
class-level metrics, confusion matrices, ROC curves, and one-shot-specific pair and
threshold analyses. CAM supports Grad-CAM, AblationCAM, and ScoreCAM.

During Deep SVDD inference, inspect the returned `accepted` field. An image is
treated as OOD when its squared-Euclidean `ood_score` is greater than the calibrated
`ood_threshold`; rejected results set `predicted_label` and `confidence` to `None`.

See the [image-classification guide](./docs/image_classification/README.md) for model
families, dataset layouts, joint Deep SVDD behavior, evaluation artifacts, and
explainability workflows.

## Segmentation

Package: `mlx.modes.segmentation`

The segmentation module provides semantic-segmentation dataset preparation,
training, resume support, benchmarking, and image, camera, or video inference. It
supports U-Net variants with native and registered backbone configurations.

```bash
# List supported segmentation models.
python -m mlx --mode segmentation --action ls-models

# Train a baseline U-Net.
python -m mlx --mode segmentation --action train \
    --dataset ./dataset \
    --model unet \
    --output ./artifacts/unet

# Train a pretrained-backbone variant.
python -m mlx --mode segmentation --action train \
    --dataset ./dataset \
    --model unet-resnet18 \
    --pretrained \
    --output ./artifacts/unet-resnet18

# Run image inference.
python -m mlx --mode segmentation --action infer-image \
    --model-path ./artifacts/unet/unet.pth \
    --input-img ./sample.jpg
```

Training stores best-loss, best-Dice, and resumable-last checkpoints together with
CSV and plot research history. Benchmarking exports aggregate, per-class, per-image,
probability, calibration, boundary, threshold, timing, and prediction artifacts.

See the [segmentation guide](./docs/segmentation/README.md) for dataset format,
models, metrics, artifacts, and inference workflows.

## NLP embeddings

Package: `mlx.modes.nlp`

The NLP module currently provides one command-style workflow: generating embeddings
for a text column in a CSV file with a compatible GGUF embedding model through
llama-cpp-python.

```bash
python -m mlx --mode nlp --action embed \
    --model-file ./models/embedding-model.gguf \
    --input-file ./data/documents.csv \
    --column-name content \
    --output-file ./data/document-embeddings.csv
```

The input column must contain non-empty text in every row. The output CSV contains
the source content and generated embedding values. When `--output-file` is omitted,
MLX derives an output name beside the input CSV.

## Documentation

- [Documentation index](./docs/README.md)
- [Object detection](./docs/object_detection/README.md)
- [Tracking by detection](./TRACKING.md)
- [Image classification](./docs/image_classification/README.md)
- [Segmentation](./docs/segmentation/README.md)
