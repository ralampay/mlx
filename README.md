# MLX

MLX is a command-line toolkit for machine-learning workflows. It provides a shared
CLI and project conventions while keeping object detection, image classification,
one-class image recognition, video anomaly detection, segmentation, saliency mapping,
text embedding, vector autoencoders, and tracking logic in focused modules.

## Table of contents

- [Module documentation](#module-documentation)
- [Architecture](./ARCHITECTURE.md)
- [Installation](#installation)
- [Command-line interface](#command-line-interface)
- [Training from an S3 dataset ZIP](#training-from-an-s3-dataset-zip)
- [Object detection and tracking](#object-detection-and-tracking)
- [Image classification](#image-classification)
- [One-class image recognition](#one-class-image-recognition)
- [Video anomaly detection](#video-anomaly-detection)
- [Segmentation](#segmentation)
- [Saliency mapping](#saliency-mapping)
- [Text embeddings](#text-embeddings)
- [Vector autoencoders](#vector-autoencoders)
- [Documentation](#documentation)
- [Development and extensions](#extending-mlx)

## Module documentation

Each guide covers the module's purpose, inputs, outputs, entrypoints, and related workflows.
The [documentation index](docs/README.md) also links setup, development, AWS, and research guides.

| Main module | Guide |
| --- | --- |
| `mlx.cli`, `mlx.cli_config`, `mlx.cli_options`, `mlx.cli_routing` | [CLI and routing](docs/cli.md) |
| `mlx.core` | [Shared core infrastructure](docs/core.md) |
| `mlx.modes` | [Workflow modes and task selection](docs/modes.md) |
| `mlx.modes.object_detection` | [Object detection](docs/object_detection/README.md) |
| `mlx.modes.object_detection.tracking` | [Tracking by detection](TRACKING.md) |
| `mlx.modes.image_classification` | [Image classification](docs/image_classification/README.md) |
| `mlx.modes.image_recognition_oc` | [One-class image recognition](docs/image_recognition_oc/README.md) |
| `mlx.modes.video_anomaly_detection` | [Video anomaly detection](docs/video_anomaly_detection/README.md) |
| `mlx.modes.segmentation` | [Semantic segmentation](docs/segmentation/README.md) |
| `mlx.modes.saliency_mapping` | [Saliency mapping](docs/saliency-mapping/README.md) |
| `mlx.modes.text_embedding` | [Text embedding and retrieval](docs/text-embedding.md) |
| `mlx.modes.autoencoder` | [Vector autoencoders](docs/autoencoder.md) |
| `mlx.modes.nlp` | [Legacy NLP CSV embedding](docs/nlp.md) |

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for package ownership, command conventions,
dependency direction, presentation boundaries, object-detection providers, and the video-anomaly
3D-backbone/checkpoint compatibility boundary.

## Installation

MLX requires Python 3.10 or newer. Create an isolated virtual environment from the
repository root so its packages do not conflict with system Python or other projects.

### Linux and macOS

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

### Windows PowerShell

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If `python3` or `py -3` reports a version older than 3.10, install a supported Python
release and use that interpreter to create `.venv`. Activate the environment again in
each new terminal before running MLX commands.

The editable install makes both `mlx` and `python -m mlx` available and reflects local
source changes. For a non-editable installation, use `python -m pip install .` instead.
Install from this checkout; the examples below use the local project's extras.

Verify the base install without a dataset, pretrained weights, or detector provider:

```bash
python -m mlx --help
python -m mlx --mode image-classification --action ls-models --names-only --format json
python -m mlx --mode segmentation --action ls-models --names-only --format json
```

### Optional workflows

Choose the extra needed for your workflow; these are alternatives, not a sequence of
required installations. The base package already includes PyTorch, torchvision, NumPy,
OpenCV, matplotlib, scikit-learn, Rich, and tqdm.

| Workflow | Install from the repository root |
| --- | --- |
| Ultralytics detection | `python -m pip install -e ".[object-detection-ultralytics]"` |
| LibreYOLO detection | `python -m pip install -e ".[object-detection-libreyolo]"` |
| Both detection providers | `python -m pip install -e ".[object-detection]"` |
| Detection and MOT tracking/evaluation | `python -m pip install -e ".[object-detection,tracking]"` |
| S3 dataset staging or SageMaker lifecycle | `python -m pip install -e ".[aws]"` |
| SageMaker detection lifecycle | `python -m pip install -e ".[aws,object-detection]"` |
| Legacy GGUF CSV embedding | `python -m pip install -e ".[nlp]"` |
| GGUF embedding with Chroma retrieval | `python -m pip install -e ".[text-embedding]"` |
| Retrieval experiments and dataset acquisition | `python -m pip install -e ".[retrieval-benchmark]"` |
| Class-activation maps | `python -m pip install -e ".[image-explainability]"` |

The retrieval-benchmark extra does not include Chroma; add `text-embedding` when using
`--vector-store chroma`. Provider extras that install Git URLs require Git. Native
provider builds such as llama-cpp-python may require additional build tools; a base
installation does not require those optional providers.

LibreYOLO follows the `release` branch of the Ralampay fork; this is a moving branch,
not a pinned commit. Record resolved dependency versions for reproducible experiments.

### Development dependencies

For the broad development/notebook environment, install the requirements file in
addition to the editable package:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pytest -q
```

`requirements.txt` includes tests, notebooks, and both detector providers, but does not
install the MLX package itself or include every optional extra. For example, add
`.[retrieval-benchmark]` for the Hub-backed dataset acquisition workflows. See the
[development guide](docs/development.md) for focused validation and contribution steps.

Leave the virtual environment with `deactivate`.

For local AWS credentials, least-privilege IAM policies, S3 preparation, and SageMaker lifecycle
commands, see the [AWS object-detection training guide](docs/object_detection/aws-sagemaker-training.md).
For sequential SageMaker training across every video-anomaly 3D configuration, see the
[AWS video-anomaly training guide](docs/video_anomaly_detection/aws-sagemaker-training.md).
For single/all-backbone Deep SVDD training and benchmarking, see the
[AWS one-class image guide](docs/image_recognition_oc/aws-sagemaker.md).

## Command-line interface

All CLI modules share one command shape:

```bash
python -m mlx --mode <mode> --action <action> [options]
```

The table shows default actions and common workflows. Run `python -m mlx --help`
for the complete registered action inventory; action availability can depend on
`--platform` and provider. Pass `--action` explicitly in scripts.

| Mode | Default action | Common actions |
| --- | --- | --- |
| `object_detection` | `train` | `train`, `fine-tune`, `benchmark`, `convert`, `infer-video`, `ls-models`; adapter research and AWS lifecycle actions |
| `track` | `run` | `run`, `benchmark`, `build-dataset`, `export-mot`, `ls-trackers` |
| `image_classification` | `test` | `train`, `benchmark`, `infer-image`, `cam`, `build-dataset`, `ls-models`, `ls-losses`, `ls-cam-methods`; AWS lifecycle actions |
| `image_recognition_oc` | `ls-models` | `train`, `infer-image`, `benchmark`, `ls-models`; AWS `train-all` and lifecycle actions |
| `video_anomaly_detection` | `ls-models` | `train`, `benchmark`, `infer-video`, `ls-models`; AWS `train-all`, `status`, `resume` |
| `segmentation` | `test` | `train`, `benchmark`, `infer-image`, `infer-camera`, `infer-video`, `build-dataset`, `ls-models`, `ls-losses` |
| `text_embedding` (`text-embedding`, `nlp`) | `embed` | `embed`, `benchmark`, `ls-metrics`, `ls-vector-stores`, `ls-embedding-backends`, `prepare-datasets`, `benchmark-autoencoders`, `select-autoencoder-settings` |
| `autoencoder` | `ls-models` | `train`, `embed`, `ls-models`, `ls-loss-functions` (`ls-losses`) |
| `saliency_mapping` (`saliency-mapping`) | `test` | `train`, `benchmark`, `infer-image`, `build-dataset`, `ls-models` |

Hyphenated mode names such as `object-detection`, `image-classification`,
`image-recognition-oc`, and `video-anomaly-detection` are also accepted. Run the following command
for the complete option reference:

```bash
python -m mlx --help
```

Mode aliases, default actions, and action inventories come from one routing registry. Use
`--format json` for a single machine-readable local result with progress and Rich presentation
suppressed. Recoverable workflow errors are emitted as JSON on standard error;
argument-parsing errors still use the CLI usage display. AWS watch commands have
streaming status output rather than the local single-result contract.

`--help` displays parser defaults. Modes may resolve omitted options differently;
consult the mode guide for effective defaults. For example, detection uses 640 × 640
inputs when dimensions are omitted, and autoencoder training uses a batch size of 64.
Use `--width`, `--height`, `--batch-size`, and `--workers` explicitly when reproducing a run.

For model-name discovery, use `ls-models --names-only`; detailed `ls-models` may construct
models to count parameters. Classification and segmentation name listing does not load
training, inference, OpenCV, or plotting modules.

### Training from an S3 dataset ZIP

Object detection, image classification, one-class image recognition, segmentation,
saliency mapping, and video anomaly detection can stage a dataset ZIP from S3 for local
training. For example:

```bash
python -m mlx --mode video_anomaly_detection --action train \
  --dataset-s3-uri s3://my-datasets/avenue-prepared.zip \
  --output ./artifacts/avenue --model resnet18 --profile mlx-training
```

Vector autoencoders use CSV input through `--input` rather than this ZIP-staging workflow.

Install the integration with `python -m pip install ".[aws]"`. The ZIP may contain the dataset
at its root or inside one wrapper directory, but its extracted contents must follow the selected
mode's documented dataset contract. MLX downloads with Boto3's normal credential chain, safely
extracts the archive, and reuses a content-identity cache under `~/.cache/mlx/datasets`. Override
that location with `--dataset-cache-dir`. A changed S3 VersionId or ETag creates a new cache entry.
The selected AWS identity needs `s3:GetObject`; MLX also uses `s3:HeadObject` semantics to inspect
the object before download (authorized through `s3:GetObject` in IAM policies).

`--dataset-s3-uri` is training-only and cannot be combined with an explicitly supplied
`--dataset`. Local S3 training requires `--output` and writes `dataset_source.json` beside the
training artifacts with the URI, object identity, size, SHA-256, cache identity, and resolved
archive root. It never records credentials or the selected profile. Cache eviction is manual.

For SageMaker object-detection or image-classification training, the existing YAML
`aws.dataset_s3_uri` remains supported. An explicit `--dataset-s3-uri` overrides it for a new
training submission; resumed runs must retain their original dataset URI.

See [Training from an S3 Dataset ZIP](docs/s3-dataset-training.md) for all archive layouts,
AWS credentials and IAM, cache behavior, provenance schema, safe-extraction rules, SageMaker
differences, resume guidance, and troubleshooting.

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
documented YOLOv9-S experiment with Drax enabled at B5. LibreYOLO also supports
`yolox-{n,t,s,m,l,x}`, `yolo9-drax-mobilenet-v3-large-{t,s,m,c}`, and
`yolox-drax-mobilenet-v3-large-{n,t,s,m,l,x}`; use an explicit size suffix.
See the [object-detection guide](docs/object_detection/README.md) for training examples
and pretrained backbone behavior.
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

To run a YOLOX foundation with an adapter-only checkpoint from an MLX adapter
experiment, pass `--adapter`. Its saved configuration supplies the adapter method
and hyperparameters; the foundation checksum must match. This works with camera
and video inference using LibreYOLO and a standard YOLOX `.pt` foundation:

```bash
python -m mlx --mode object_detection --provider libreyolo --action infer-video \
    --model-path ~/workspace/object-detection-models/foundational-yolox-l.pt \
    --adapter ~/workspace/object-detection-models/yolox-l-lora-dawn-best.pt \
    --file-path ~/videos/sample.mp4
```

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

## One-class image recognition

Package: `mlx.modes.image_recognition_oc`

This mode trains a normal-only still-image recognizer. The initial `deep-svdd` algorithm uses every
standard image-classification model as a replaceable feature backbone; Siamese variants are
excluded. Training and validation contain normal images only, while benchmark data contains both
normal and anomalous images.

```bash
python -m mlx --mode image_recognition_oc --action ls-models

python -m mlx --mode image_recognition_oc --action train \
    --model deep-svdd --backbone resnet18 \
    --dataset ./one-class-dataset --output ./artifacts/one-class

python -m mlx --mode image_recognition_oc --action infer-image \
    --model-path ./artifacts/one-class/resnet18-deep-svdd.pth \
    --input-img ./sample.jpg

python -m mlx --mode image_recognition_oc --action benchmark \
    --model-path ./artifacts/one-class/resnet18-deep-svdd.pth \
    --dataset ./one-class-dataset --output ./benchmark

python -m mlx --mode image_recognition_oc --platform aws --action train-all \
    --config ./aws-svdd.yaml
```

See the [one-class image recognition guide](./docs/image_recognition_oc/README.md) for dataset,
checkpoint, inference, resume, and benchmark contracts, and the
[AWS guide](./docs/image_recognition_oc/aws-sagemaker.md) for SageMaker execution.

## Video anomaly detection

Package: `mlx.modes.video_anomaly_detection`

This mode trains a normal-only Deep SVDD model over fixed-length video clips. By default it
inflates every standard image-classification architecture into a dedicated 3D CNN, preserves time
through the backbone, globally pools a clip embedding, and calibrates an anomaly threshold only
from held-out normal validation clips. `DraxNet3D` and `DraxMobileNetV3Large3D` support both
`average` and `sknet` fusion. Siamese/one-shot models are excluded by model-family capability. Legacy
frame-2D+TCN checkpoints remain loadable with `--backbone-mode frame-2d`.

The training frame buffer contains `--clip-length` sampled RGB frames. With frame stride `S`, one
window spans `(clip_length - 1) * S + 1` source frames. Training and validation data must contain
normal examples only; test data may contain both normal and anomalous sequences.

```bash
python -m mlx --mode video_anomaly_detection --action train \
    --model resnet18 --backbone-mode 3d --backbone-temporal-kernel-size 3 \
    --dataset ./ped2-prepared --output ./artifacts/ped2 \
    --clip-length 16 --frame-stride 1 --height 224 --width 224

python -m mlx --mode video_anomaly_detection --action ls-models

python -m mlx --mode video_anomaly_detection --action benchmark \
    --model-path ./artifacts/ped2/resnet18-3d-svdd.pth \
    --dataset ./ped2-prepared/test --output ./artifacts/ped2/benchmark

python -m mlx --mode video-anomaly-detection --action infer-video \
    --model-path ./artifacts/ped2/resnet18-3d-svdd.pth \
    --file-path ./sample.mp4 --output ./inference
```

See the [video anomaly detection guide](./docs/video_anomaly_detection/README.md) for the dataset
contract, UCSD Ped2 mapping, checkpoint semantics, research artifacts, and limitations.

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
CSV and plot research history. When a paired test partition exists, it also writes
up to 16 original, ground-truth, prediction, overlay, and combined-panel samples.
Benchmarking exports aggregate, per-class, per-image, probability, calibration,
boundary, threshold, timing, and prediction artifacts.

Use `--model all` with `--action train` to train every registered segmentation model
from scratch against one dataset and benchmark each best-validation-loss checkpoint
on the required test split. Local or S3 ZIP datasets are supported; S3 data is staged
once for the complete batch. Per-model artifacts live below the output directory and
root `all-models.json`/`leaderboard.csv` files summarize the comparison.

Use `--model all-small` for the original four-model comparison group, whose members
are below 10 million parameters. Segmentation training also accepts `--transform resize` (the default),
`--transform random-crop`, or `--transform center-crop`; random training crops use
deterministic center crops for validation, test benchmarking, and samples.

See the [segmentation guide](./docs/segmentation/README.md) for dataset format,
models, metrics, artifacts, and inference workflows.

## Saliency mapping

Package: `mlx.modes.saliency_mapping`

This still-image salient-object-detection mode has its own model registry. Its built-in
U-Net models reuse segmentation encoders and decoders with a one-channel logits head;
custom saliency models register directly with the saliency mode. Sigmoid probabilities and
grayscale targets remain continuous in `[0, 1]` rather than becoming class IDs.

```bash
python -m mlx --mode saliency-mapping --action ls-models
python -m mlx --mode saliency-mapping --action test --model all-small
python -m mlx --mode saliency-mapping --action train \
  --model unet-drax_mobilenet_v3_large-average \
  --dataset ./data/duts --output ./artifacts/saliency
```

Training uses weighted BCE-with-logits + SSIM + IoU and selects the best checkpoint
by lowest validation MAE. Benchmarks export per-image and aggregate MAE, max/mean
F-beta (`beta²=0.3`), threshold and precision-recall data, runtime measurements,
continuous predictions, heatmaps, overlays, and sample panels. The built-in `all-small` group matches segmentation; custom saliency registrations
can extend its membership independently. `all` selects every registered saliency architecture. See the [saliency-mapping guide](./docs/saliency-mapping/README.md).

## Text embeddings

Package: `mlx.modes.text_embedding`

The canonical workflow embeds BEIR-format retrieval datasets with a compatible GGUF
sequence-embedding model, persists caller-generated vectors in Chroma, and benchmarks
retrieval from those artifacts without loading the model again.

```bash
python -m mlx --mode text-embedding --action embed \
    --model ./models/multilingual-e5-small-q8_0.gguf \
    --input ./datasets/scifact --output ./artifacts/scifact-e5-small \
    --query-prefix "query: " --document-prefix "passage: "

python -m mlx --mode text-embedding --action benchmark \
    --input ./artifacts/scifact-e5-small \
    --output ./results/scifact-e5-small
```

Install with `python -m pip install ".[text-embedding]"`. The historic
`--mode nlp --model-file ... --input-file ...` CSV workflow remains supported.
See the [text-embedding research guide](docs/text-embedding.md) for dataset layout,
prefixes, normalization, artifacts, metrics, and extension boundaries.

## Vector autoencoders

Package: `mlx.modes.autoencoder`

Train a one-dimensional reconstruction autoencoder from an MLX embedding CSV, then export its
bottleneck representation or apply it directly during text embedding:

```bash
python -m mlx --mode autoencoder --action train \
    --model simple --input ./artifacts/scifact/corpus_embeddings.csv \
    --output ./artifacts/autoencoder-128 --input-dim 384 \
    --hidden-dim 256 --bottleneck-dim 128 --loss mse

python -m mlx --mode autoencoder --action embed \
    --model simple --model-path ./artifacts/autoencoder-128/autoencoder.pth \
    --input ./artifacts/scifact/query_embeddings.csv \
    --output ./artifacts/scifact/query_embeddings_ae128.csv

python -m mlx --mode text-embedding --action embed \
    --model ./models/embedding-model.gguf \
    --adapter ./artifacts/autoencoder-128/autoencoder.pth \
    --input ./datasets/scifact --output ./artifacts/scifact-ae128
```

See the [autoencoder guide](docs/autoencoder.md) for checkpoints, preprocessing, custom model and
loss contracts, and the built-in MSE, MAE, and Smooth L1 losses.

## Documentation

- [Documentation index](./docs/README.md)
- [Object detection](./docs/object_detection/README.md)
- [Tracking by detection](./TRACKING.md)
- [Image classification](./docs/image_classification/README.md)
- [One-class image recognition](./docs/image_recognition_oc/README.md)
- [Segmentation](./docs/segmentation/README.md)
- [Text embedding and retrieval](./docs/text-embedding.md)
- [Vector autoencoders](./docs/autoencoder.md)
- [Saliency mapping](./docs/saliency-mapping/README.md)
## Extending MLX

See the [extension inventory](docs/extensions.md), [executable tutorials](docs/tutorials/README.md),
and canonical [architecture](ARCHITECTURE.md) for adding models, trackers, losses, metrics, and adapters.
The [development guide](docs/development.md) covers setup, new CLI options/actions, Python
workflow composition, and the inspect/validate/review cycle.
