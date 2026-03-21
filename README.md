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
python -m mlx --mode object-detection --action train
```

The codebase is organized around mode packages:

- `mlx.core`: shared exceptions and terminal UI helpers.
- `mlx.modes.one_shot`: image-classification workflows, including one-shot models.
- `mlx.modes.object_detection.ultralytics`: object detection on Ultralytics.

## Project Layout

```text
mlx/
├── core/
├── modes/
│   ├── object_detection/
│   │   └── ultralytics/
│   └── one_shot/
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
- `tqdm`
- `ultralytics` from the pinned Git repository in `requirements.txt`

## Usage

All commands share the same high-level signature:

```bash
python -m mlx --mode <mode-name> --action <action-name>
```

Examples:

```bash
python -m mlx --mode object-detection --action train --dataset-path ./dataset --model ultralytics/cfg/models/ext/cad_yolo12.yaml
python -m mlx --mode object-detection --action infer-camera --model ultralytics/cfg/models/ext/cad_yolo12.yaml --model-path ./runs/train/weights/best.pt
python -m mlx --mode image-classification --action train --dataset-path ./omniglot
python -m mlx --mode image-classification --action build-dataset --dataset-path ./raw-dataset
```

Run `python -m mlx --help` for the complete CLI reference.

## Modes

| Mode | Package | Actions | Docs |
| --- | --- | --- | --- |
| `object-detection` | `mlx.modes.object_detection.ultralytics` | `train`, `infer-camera`, `infer-video` | [Object detection](./docs/object_detection/README.md) |
| `image-classification` | `mlx.modes.one_shot` | `train`, `test`, `benchmark`, `infer-image`, `build-dataset` | [Image classification](./docs/image_classification/README.md) |

`image-classification` is the public mode name even when using one-shot functionality internally.

## Documentation

- [Documentation index](./docs/README.md)
- [Object detection mode docs](./docs/object_detection/README.md)
- [Image classification docs](./docs/image_classification/README.md)
