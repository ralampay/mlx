# MLX

Machine-learning workflow runner for computer-vision tasks.

## Table of Contents

- [Overview](#overview)
- [Project Layout](#project-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Environment](#environment)
- [Documentation](#documentation)

## Overview

MLX provides a CLI for running feature-specific workflows behind a shared interface:

```bash
mlx --module system --action ls-env
```

The codebase is organized around explicit namespaces:

- `mlx.core`: shared exceptions, types, and terminal UI helpers.
- `mlx.features.one_shot`: one-shot image classification on Torch.
- `mlx.features.object_detection.ultralytics`: object detection on Ultralytics.
- `mlx.platforms`: runtime module registry plus generic system actions.

## Project Layout

```text
mlx/
├── core/
├── features/
│   ├── object_detection/
│   │   └── ultralytics/
│   └── one_shot/
└── platforms/
```

Feature code now lives under `mlx.features.*`. `mlx.platforms` is intentionally thin and only handles dispatch.

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
mlx --module <module-name> --platform <platform-name> --action <action-name>
```

Examples:

```bash
mlx --module system --action ls-env
mlx --module obj-detect --platform ultralytics --action train --dataset-path ./dataset --model ultralytics/cfg/models/ext/cad_yolo12.yaml
mlx --module ic-one-shot --platform torch --action train --dataset-path ./omniglot
```

Run `mlx --help` for the complete CLI reference.

## Modules

| Module | Platform | Namespace | Docs |
| --- | --- | --- | --- |
| `system` | `generic` | `mlx.platforms.system` | [System and project overview](./docs/README.md) |
| `obj-detect` | `ultralytics` | `mlx.features.object_detection.ultralytics` | [Object detection](./docs/object_detection/README.md) |
| `ic-one-shot` | `torch` | `mlx.features.one_shot` | [One-shot image classification](./docs/image_classification/README.md) |

Segmentation is not implemented in the current codebase.

## Environment

Copy the template and populate any required variables:

```bash
cp .env.dist .env
```

Supported environment variables:

- `ROBOFLOW_API_KEY`: optional key for Roboflow-backed dataset workflows.

The CLI loads `.env` automatically on startup. Inspect the current environment with:

```bash
mlx --module system --action ls-env
```

## Documentation

- [Documentation index](./docs/README.md)
- [Object detection namespace docs](./docs/object_detection/README.md)
- [One-shot image classification namespace docs](./docs/image_classification/README.md)
