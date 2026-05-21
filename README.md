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
python -m mlx --mode object_detection --action train --dataset coco8 --model draxnet-yolo26 --output ./runs/draxnet
python -m mlx --mode object_detection --action convert --model-path ./runs/draxnet/exp/weights/best.pt --output ./exports
python -m mlx --mode object_detection --action infer-camera --model-path ./exports/best.onnx
python -m mlx --mode object_detection --action infer-video --model-path ./exports/best.onnx --file-path ~/videos/sample.mp4
python -m mlx --mode image_classification --action train --output ./artifacts/resnet18 --dataset ./dataset --model resnet18 --seed 42
python -m mlx --mode image_classification --action train --output ./artifacts/siamese --dataset ./omniglot --model siamese-le-net --seed 42
python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset
python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --output ./dataset --train-count 100 --val-count 20 --test-count 20 --overwrite --seed 42
python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --split-mode ratios --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --output ./dataset --overwrite --seed 42
python -m mlx --mode segmentation --action train --dataset ./dataset --model unet --output ./unet-seg.pt
python -m mlx --mode segmentation --action infer-image --model-path ./unet-seg.pt --input-img ./sample.jpg
```

Run `python -m mlx --help` for the complete CLI reference.

For object detection, `--model` now accepts built-in aliases such as `yolo26`, `yolov26`, and `draxnet-yolo26`. `--dataset` or `--dataset-path` accepts a local YOLO dataset root, a dataset YAML, or built-in Ultralytics aliases such as `coco8` and `coco128`.

Object-detection training also reuses checkpoints already present under the selected `--output` directory when `--model-path` is omitted. If a resumable `last.pt` is found, MLX continues the existing run and says so in the training output; otherwise it warm-starts from the newest `.pt` it finds there. `--use-best` is enabled by default, so after training MLX selects `weights/best.pt` as the checkpoint for downstream use when Ultralytics writes it; pass `--no-use-best` to prefer `weights/last.pt`.
For object-detection inference, `.pt` checkpoints continue to use Ultralytics, while `.onnx` model paths now run through ONNX Runtime without requiring `--model`.
The documented deployment path is now: train with Ultralytics, convert the resulting `.pt` checkpoint to `.onnx`, then run inference against that `.onnx` model.

## Modes

| Mode | Package | Actions | Docs |
| --- | --- | --- | --- |
| `object_detection` | `mlx.modes.object_detection.ultralytics` | `train`, `infer-camera`, `infer-video`, `convert` | [Object detection](./docs/object_detection/README.md) |
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
