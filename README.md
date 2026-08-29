# MLX

MLX is a command-line toolkit for machine-learning workflows. It provides a shared
CLI and project conventions while keeping object detection, image classification,
segmentation, NLP, and tracking logic in focused modules.

## Contents

- [Architecture](./ARCHITECTURE.md)
- [Installation](#installation)
- [Command-line interface](#command-line-interface)
- [Object detection and tracking](#object-detection-and-tracking)
- [Image classification](#image-classification)
- [Segmentation](#segmentation)
- [NLP embeddings](#nlp-embeddings)
- [Documentation](#documentation)

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for package ownership, command conventions,
dependency direction, presentation boundaries, and the object-detection provider API.

## Installation

MLX requires Python 3.10 or newer. Create an isolated virtual environment from the
repository root so its packages do not conflict with system Python or other projects.

### Linux and macOS

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If `python3` or `py -3` reports a version older than 3.10, install a supported Python
release and use that interpreter to create `.venv`. Activate the environment again in
each new terminal before running MLX commands.

The requirements file installs the project dependencies used for development,
notebooks, tests, and both object-detection providers. Confirm the environment after
installation:

```bash
python -m mlx --help
python -m pytest -q
```

Leave the environment when finished with:

```bash
deactivate
```

The development dependency set includes NumPy, pandas, SciPy, motmetrics, PyTorch,
torchvision, OpenCV, Rich, scikit-learn, ONNX Runtime, llama-cpp-python, the Ralampay
Ultralytics fork, and the `release` branch of the Ralampay LibreYOLO fork.

Package consumers who do not use `requirements.txt` can install the project with only
the object-detection provider they need:

```bash
python -m pip install ".[object-detection-ultralytics]"
python -m pip install ".[object-detection-libreyolo]"
python -m pip install ".[object-detection]"  # both providers
python -m pip install ".[aws,object-detection]"  # SageMaker detection lifecycle
python -m pip install ".[aws]"                   # SageMaker classification lifecycle
```

For local AWS credentials, least-privilege IAM policies, S3 preparation, and SageMaker lifecycle
commands, see the [AWS object-detection training guide](docs/object_detection/aws-sagemaker-training.md).

## Command-line interface

All CLI modules share one command shape:

```bash
python -m mlx --mode <mode> --action <action> [options]
```

Available CLI modes are:

| Mode | Actions |
| --- | --- |
| `object_detection` | `train`, `benchmark`, AWS `resume`/`status`/`stop`, `infer-camera`, `infer-video`, `convert`, `ls-models` |
| `track` | `run`, `export-mot`, `ls-trackers` |
| `image_classification` | `train`, `test`, `benchmark`, AWS `resume`/`status`/`stop`, `infer-image`, `cam`, `build-dataset`, `ls-models` |
| `segmentation` | `train`, `test`, `benchmark`, `infer-image`, `infer-camera`, `infer-video`, `build-dataset`, `ls-models` |
| `nlp` | `embed` |

Hyphenated mode names such as `object-detection` and `image-classification` are also
accepted. Run the following command for the complete option reference:

```bash
python -m mlx --help
```

## Object detection and tracking

### Object detection

Package: `mlx.modes.object_detection`

The default `ultralytics` provider and alternative `libreyolo` provider both train
models, convert PyTorch checkpoints to ONNX, and run camera or video inference through
normalized detection adapters. Select LibreYOLO explicitly with `--provider libreyolo`.

Ultralytics aliases include `yolo26`, `yolov26`, `draxnet-ave-yolo26`, and
`draxnet-sknet-yolo26`. The legacy `draxnet-yolo26` alias selects the fixed-average
variant. First-class LibreYOLO training/listing aliases are `yolo9-t`, `yolo9-s`,
`yolo9-m`, `yolo9-c`, and `yolo9-s-drax-b5`. The Drax alias matches LibreYOLO's
documented YOLOv9-S experiment with Drax enabled at B5.
Dataset input may be a local YOLO dataset root, a dataset YAML, or an alias such as
`coco8` or `coco128`.

```bash
# Inspect available models.
python -m mlx --mode object_detection --action ls-models

# Inspect the first-class LibreYOLO models.
python -m mlx --mode object_detection --provider libreyolo --action ls-models

# Train an Ultralytics model.
python -m mlx --mode object_detection --action train \
    --dataset coco8 \
    --model draxnet-yolo26 \
    --output ./runs/draxnet

# Train with the LibreYOLO fork.
python -m mlx --mode object_detection --provider libreyolo --action train \
    --dataset coco8 \
    --model yolo9-t \
    --output ./runs/libreyolo

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

Provider checkpoints are not interchangeable. Always select the same provider that
created the `.pt` or `.onnx` artifact.

The intended deployment flow is:

```text
selected provider training → .pt checkpoint → ONNX conversion → camera/video inference
```

See [the object-detection guide](./docs/object_detection/README.md) for dataset
layout, model resolution, training artifacts, inference, and conversion details.

### Tracking by detection

Package: `mlx.modes.object_detection.tracking`

Tracking is exposed as a separate CLI mode while remaining layered on normalized
object detections. Select the detection provider and either the built-in `sort` or
`bytetrack` algorithm:

```bash
python -m mlx --mode track --tracker bytetrack \
    --provider ultralytics \
    --model yolo26 \
    --model-path ./runs/yolo/weights/best.pt \
    --file-path ./video.mp4 \
    --ground-truth ./gt.txt \
    --output ./tracking-run
```

Detector-specific inference and generic tracking remain separate:

```text
frame source → detection adapter → normalized detections
             → tracking algorithm → immutable track results
```

The run writes class-aware `tracks.jsonl`, MOTChallenge-compatible `tracks.txt`,
portable `replay.json`, and a self-contained `replay.html` 2D player that does not
require the source video. `track --action export-mot` can extract all classes or a
selected class from the JSONL into a strict MOT file. When ground truth is provided,
the run also writes standard MOT metrics in `metrics.json` and includes GT boxes in
the replay. External algorithms can be selected with
`--tracker package.module:ClassName`. See [TRACKING.md](./TRACKING.md) for provider
examples, output details, configuration, lower-level APIs, and see
[CUSTOM_TRACKING.md](./CUSTOM_TRACKING.md) for the complete tracker contract and a
copyable custom tracker.

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

Image-classification training can run asynchronously on SageMaker with Managed Spot
recovery. See the [AWS image-classification guide](./docs/image_classification/aws-sagemaker-training.md).

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
