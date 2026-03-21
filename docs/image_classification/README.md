# Image Classification

Mode: `image-classification`

Package: `mlx.modes.one_shot`

## Overview

This mode provides the image-classification workflow exposed by:

```bash
python -m mlx --mode image-classification
```

The source is organized by responsibility:

- `runner.py`: default config and action dispatch.
- `actions.py`: train, test, benchmark, one-shot inference, and dataset actions.
- `data.py`: dataset loading, dataset building, and shared image preprocessing.
- `models/`: one-shot model definitions used by the mode.
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
python -m mlx \
    --mode image-classification \
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

## Build Dataset

`build-dataset` is used to convert a label-organized source dataset into a split dataset with `train/`, `val/`, and `test/` directories.

Required flag:

- `--dataset-path`: source dataset root. This must point to the unsplit label-organized dataset you want to process.

Behavior:

- The command scans each label directory under `--dataset-path`.
- It prints a label summary showing how many images were found per label.
- It then prompts for three values: images per label for `TRAIN`, `VAL`, and `TEST`.
- It finally prompts for the output path where the split dataset should be created.
- If the output directory already exists, MLX asks for confirmation before overwriting it.
- If a label has fewer images than the requested total, MLX prints a warning before continuing.

Expected input layout:

```text
<source-dataset>/
├── cats/
│   ├── cat-1.jpg
│   ├── cat-2.jpg
│   └── ...
├── dogs/
│   ├── dog-1.jpg
│   ├── dog-2.jpg
│   └── ...
└── horses/
    ├── horse-1.jpg
    ├── horse-2.jpg
    └── ...
```

Example command:

```bash
python -m mlx \
    --mode image-classification \
    --action build-dataset \
    --dataset-path ~/datasets/animals-raw
```

Example interactive flow:

```text
How many images per label for TRAIN? 20
How many images per label for VAL? 5
How many images per label for TEST? 5
Enter output path for split dataset ~/datasets/animals-split
```

This creates a dataset like:

```text
~/datasets/animals-split/
├── train/
│   ├── cats/
│   ├── dogs/
│   └── horses/
├── val/
│   ├── cats/
│   ├── dogs/
│   └── horses/
└── test/
    ├── cats/
    ├── dogs/
    └── horses/
```

Another example:

```bash
python -m mlx \
    --mode image-classification \
    --action build-dataset \
    --dataset-path ./data/omniglot-raw
```

Then provide prompts such as:

```text
How many images per label for TRAIN? 12
How many images per label for VAL? 4
How many images per label for TEST? 4
Enter output path for split dataset ./data/omniglot-split
```

Use the generated output directory as `--dataset-path` for `train`, and use its `test/` directory for `benchmark` or `infer-image` when needed.

## Benchmarking

Example:

```bash
python -m mlx \
    --mode image-classification \
    --action benchmark \
    --dataset-path ~/datasets/omniglot/test \
    --model-path ~/datasets/omniglot/checkpoints/best_epoch_10.pt \
    --device cpu
```

`benchmark` requires `--model-path` and uses `--dataset-path` as the directory to evaluate.

## Image Inference

Example:

```bash
python -m mlx \
    --mode image-classification \
    --action infer-image \
    --dataset-path ~/datasets/omniglot/test \
    --input-img ~/datasets/query/sample.png \
    --model siamese-le-net \
    --device cpu
```

`infer-image` loads the query image, computes its embedding, compares it against images under `--dataset-path`, and renders the top matches.
