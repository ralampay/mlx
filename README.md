# MLX

Machine-learning workflow runner for computer-vision tasks.

## Table of Contents

- [Overview](#overview)
- [Project Layout](#project-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Modes](#modes)
- [Documentation](#documentation)

## Overview

MLX provides a CLI for running mode-specific workflows behind a shared interface:

```bash
python -m mlx --mode object_detection --action train
```

The codebase is organized around mode packages:

- `mlx.core`: shared exceptions and terminal UI helpers.
- `mlx.modes.image_classification`: image-classification workflows for both one-shot and standard classifiers.
- `mlx.modes.object_detection.ultralytics`: object detection on Ultralytics.
- `mlx.modes.segmentation`: semantic segmentation workflows for U-Net style models.

## Project Layout

```text
mlx/
├── core/
├── modes/
│   ├── object_detection/
│   │   └── ultralytics/
│   ├── image_classification/
│   └── segmentation/
```

The CLI now dispatches directly by `--mode`, so there is no separate platform abstraction.

## Installation

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Current runtime dependencies:

- `numpy`
- `opencv-python`
- `python-dotenv`
- `rich`
- `scikit-learn`
- `torch`
- `torchvision`
- `tqdm`
- `ultralytics` from the pinned Git repository in `requirements.txt`

## Usage

All commands share the same high-level signature:

```bash
python -m mlx --mode <mode-name> --action <action-name>
```

Examples:

```bash
python -m mlx --mode object_detection --action train --dataset-path ./dataset --model ultralytics/cfg/models/ext/cad_yolo12.yaml
python -m mlx --mode object_detection --action infer-camera --model ultralytics/cfg/models/ext/cad_yolo12.yaml --model-path ./runs/train/weights/best.pt
python -m mlx --mode image_classification --action train --output ./artifacts/resnet18 --dataset ./dataset --model resnet18 --seed 42
python -m mlx --mode image_classification --action train --output ./artifacts/siamese --dataset ./omniglot --model siamese-le-net --seed 42
python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset
python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --output ./dataset --train-count 100 --val-count 20 --test-count 20 --overwrite --seed 42
python -m mlx --mode segmentation --action train --dataset ./dataset --model unet --output ./unet-seg.pt
python -m mlx --mode segmentation --action infer-image --model-path ./unet-seg.pt --input-img ./sample.jpg
```

Run `python -m mlx --help` for the complete CLI reference.

## Modes

| Mode | Package | Actions | Docs |
| --- | --- | --- | --- |
| `object_detection` | `mlx.modes.object_detection.ultralytics` | `train`, `infer-camera`, `infer-video` | [Object detection](./docs/object_detection/README.md) |
| `image_classification` | `mlx.modes.image_classification` | `train`, `test`, `benchmark`, `infer-image`, `build-dataset` | [Image classification](./docs/image_classification/README.md) |
| `segmentation` | `mlx.modes.segmentation` | `train`, `test`, `infer-image`, `infer-camera`, `infer-video` | [Segmentation](./docs/segmentation/README.md) |

`image_classification` supports both Siamese one-shot models and standard classifiers such as `resnet18`, `resnet50`, `densenet121`, `mobilenet_v3_large`, `efficientnet_b0`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `draxnet`, and `drax_mobilenet_v3_large`.

For image-classification training, `--output` is an artifact directory. Training writes `{model}.pth` and `training.csv` inside that directory.
For image-classification benchmarking, `--output` can also be used to store `metrics.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, and `roc_curve.png`.

## Documentation

- [Documentation index](./docs/README.md)
- [Object detection mode docs](./docs/object_detection/README.md)
- [Image classification docs](./docs/image_classification/README.md)
- [Segmentation mode docs](./docs/segmentation/README.md)
