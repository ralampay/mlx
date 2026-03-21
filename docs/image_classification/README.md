# Image Classification

Mode: `image_classification`

Package: `mlx.modes.image_classification`

## Overview

This mode provides the image-classification workflow exposed by:

```bash
python -m mlx --mode image_classification
```

The source is organized by responsibility:

- `runner.py`: default config and action dispatch.
- `train.py`: training loops and smoke tests for one-shot and standard classifiers.
- `evaluation.py`: benchmark flows for both model families.
- `inference.py`: image inference for both model families.
- `data.py`: dataset loading, dataset building, and shared image preprocessing.
- `models/`: Siamese and standard classification model builders.
- `utils.py`: model resolution and checkpoint metadata helpers.
- `presentation.py`: rich tables and one-shot match rendering.
- `data.py` exposes `ImageClassificationDataset`, which expects a dataset root containing `train/` and `val/`.

## Supported Models

This mode supports two training setups:

- One-shot similarity models: `siamese-le-net`
- Standard classification models: `resnet18`, `resnet50`

The selected `--model` determines which training, benchmarking, and inference path is used. Torchvision-backed models are loaded by name for `resnet18` and `resnet50`. Additional custom standard classifiers can be plugged in later through the model registry.

## Dataset Expectations

### Training

Training expects a dataset root with at least:

```text
<dataset-root>/
├── train/
│   └── <label>/
└── val/
    └── <label>/
```

Each label directory must contain at least two images so positive pairs can be generated.

This same split layout is used by both model families:

- Standard classifiers use `train/` and `val/` as supervised class datasets.
- One-shot models use `train/` and `val/` to generate positive and negative image pairs.

The standard classification path uses `ImageClassificationDataset(dataset_path, split="train" | "val", transform=...)`, which returns `(x, y)` pairs where `x` is the transformed image tensor and `y` is the label index.

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

### Standard Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --model resnet18 \
    --action train \
    --dataset ~/datasets/animals \
    --output ./model.pth \
    --epochs 50 \
    --batch-size 16 \
    --device cuda:0 \
    --lr 0.001
```

Important arguments:

- `--model`: standard classifier such as `resnet18` or `resnet50`.
- `--dataset`: dataset root containing `train/` and `val/`.
- `--output`: output checkpoint file, for example `model.pth`.
- `--epochs`, `--batch-size`, `--device`, `--lr`: standard training controls.
- `--pretrained`: enable pretrained initialization for supported torchvision backbones.
- `--height`, `--width`: input dimensions used to build `input_size`.

### One-Shot Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --model siamese-le-net \
    --action train \
    --dataset ~/datasets/omniglot \
    --output ./siamese.pth \
    --epochs 50 \
    --batch-size 8 \
    --device cuda:0
```

Important arguments:

- `--model`: one-shot model name, currently `siamese-le-net`.
- `--dataset`: dataset root containing `train/` and `val/`.
- `--output`: output checkpoint file, for example `siamese.pth`.
- `--embedding-size`: Siamese embedding width.
- `--epochs`, `--batch-size`, `--device`, `--lr`: training controls.
- `--height`, `--width`: input dimensions used to build `input_size`.

## Available Actions

- `train`: train the selected model and write the best checkpoint to `--output`.
- `test`: run a random-tensor smoke test for the configured model.
- `benchmark`: evaluate a trained checkpoint against a dataset directory. Standard models classify labels directly; one-shot models evaluate pair similarity.
- `infer-image`: run inference for one input image. Standard models output class probabilities; one-shot models compare against a reference dataset and show the best matches.
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
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw
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
    --mode image_classification \
    --action build-dataset \
    --dataset ./data/omniglot-raw
```

Then provide prompts such as:

```text
How many images per label for TRAIN? 12
How many images per label for VAL? 4
How many images per label for TEST? 4
Enter output path for split dataset ./data/omniglot-split
```

Use the generated output directory as `--dataset` for `train`, and use its `test/` directory for `benchmark` or `infer-image` when needed.

## Benchmarking

### Standard Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action benchmark \
    --model-path ~/datasets/animals/checkpoints/best_epoch_12.pt \
    --dataset ~/datasets/animals/test \
    --device cpu
```

For standard classifiers, `benchmark` loads class labels from the checkpoint metadata and evaluates accuracy, precision, recall, and F1 against the labelled images in `--dataset`. If `--dataset` points to the dataset root and a `test/` directory exists, MLX evaluates that `test/` directory automatically.

### One-Shot Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action benchmark \
    --dataset ~/datasets/omniglot/test \
    --model-path ~/datasets/omniglot/checkpoints/best_epoch_10.pt \
    --device cpu
```

For one-shot models, `benchmark` requires `--model-path` and evaluates similarity pairs built from the provided dataset directory.

## Image Inference

### Standard Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action infer-image \
    --model-path ~/datasets/animals/checkpoints/best_epoch_12.pt \
    --input-img ~/datasets/query/cat.jpg \
    --device cpu
```

For standard classifiers, `infer-image` predicts the top classes for `--input-img`. `--dataset` is not required for this path because labels are loaded from the checkpoint metadata.

### One-Shot Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action infer-image \
    --dataset ~/datasets/omniglot/test \
    --input-img ~/datasets/query/sample.png \
    --model siamese-le-net \
    --device cpu
```

`infer-image` loads the query image, computes its embedding, compares it against images under `--dataset`, and renders the top matches.
