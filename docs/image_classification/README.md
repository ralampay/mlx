# One-Shot Image Classification

Namespace: `mlx.features.one_shot`

## Overview

This namespace contains the Torch-backed one-shot classification workflow exposed by:

```bash
mlx --module ic-one-shot --platform torch
```

The source is organized by responsibility:

- `runner.py`: default config and action dispatch.
- `actions.py`: train, test, benchmark, and inference actions.
- `data.py`: dataset loading, dataset building, and shared image preprocessing.
- `models/`: one-shot model definitions.
- `presentation.py`: rich tables and OpenCV result rendering.

## Dataset Expectations

### Training

Training expects a dataset root with at least:

```text
<dataset-path>/
├── train/
│   └── <label>/
└── val/
    └── <label>/
```

Each label directory must contain at least two images so positive pairs can be generated.

### Dataset Builder

`build-dataset` expects a flat label-organized source dataset:

```text
<source-dataset>/
└── <label>/
    ├── image-1.png
    └── image-2.png
```

It interactively creates `train/`, `val/`, and `test/` splits in a new output directory.

## Training

Example:

```bash
mlx --module ic-one-shot \
    --platform torch \
    --model siamese-le-net \
    --action train \
    --dataset-path ~/datasets/omniglot \
    --epochs 50 \
    --batch-size 8 \
    --device cuda:0
```

Important arguments:

- `--model`: model name, currently `siamese-le-net`.
- `--dataset-path`: dataset root containing `train/` and `val/`.
- `--embedding-size`: Siamese embedding width.
- `--epochs`, `--batch-size`, `--device`: training controls.
- `--height`, `--width`: input dimensions used to build `input_size`.

## Available Actions

- `train`: train the network and save best checkpoints under `<dataset-path>/checkpoints`.
- `test`: run a random-tensor smoke test for the configured model.
- `benchmark`: evaluate a trained checkpoint against a dataset directory.
- `infer-image`: compare one input image against a reference dataset and show the best matches.
- `build-dataset`: interactively create train/val/test splits from a label-organized source dataset.

## Benchmarking

Example:

```bash
mlx --module ic-one-shot \
    --platform torch \
    --action benchmark \
    --dataset-path ~/datasets/omniglot/test \
    --model-path ~/datasets/omniglot/checkpoints/best_epoch_10.pt \
    --device cpu
```

`benchmark` requires `--model-path` and uses `--dataset-path` as the directory to evaluate.

## Image Inference

Example:

```bash
mlx --module ic-one-shot \
    --platform torch \
    --action infer-image \
    --dataset-path ~/datasets/omniglot/test \
    --input-img ~/datasets/query/sample.png \
    --model siamese-le-net \
    --device cpu
```

`infer-image` loads the query image, computes its embedding, compares it against images under `--dataset-path`, and renders the top matches.
