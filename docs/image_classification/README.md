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
- Standard classification models: `resnet18`, `resnet50`, `densenet121`, `mobilenet_v3_large`, `efficientnet_b0`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `draxnet`, `drax_mobilenet_v3_large`

The selected `--model` determines which training, benchmarking, and inference path is used. Torchvision-backed standard models are loaded by name and their classifier heads are adapted to the dataset class count. Additional custom standard classifiers can be plugged in later through the model registry.

All standard classifiers share the same preprocessing family:

- Training: `Resize`, `ToTensor`, `Normalize`. Add `--apply-transformations` to include `RandomHorizontalFlip` and `RandomRotation(10)`.
- Validation / benchmark / inference: `Resize`, `ToTensor`, `Normalize`
- RGB normalization uses ImageNet mean/std: `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`
- Grayscale normalization uses mean/std: `(0.5,)` / `(0.5,)`

Experimental custom-block documentation:

- [Experimental Blocks](./experimental_blocks.md)

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

For image-classification training, `--output` is treated as an artifact directory. MLX creates the directory if needed and writes:

```text
<output>/
├── {model}.pth
└── training.csv
```

`training.csv` contains one row per epoch with this schema:

```text
epoch,loss,metric
```

Metric semantics depend on the model family:

- Standard classifiers: `metric` is validation accuracy.
- One-shot classifiers: `metric` is validation loss.

### Standard Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --model resnet18 \
    --action train \
    --dataset ~/datasets/animals \
    --output ./artifacts/resnet18 \
    --epochs 50 \
    --batch-size 16 \
    --device cuda:0 \
    --lr 0.001
```

Important arguments:

- `--model`: standard classifier such as `resnet18` or `resnet50`.
- `--dataset`: dataset root containing `train/` and `val/`.
- `--output`: output artifact directory. Training writes `{model}.pth` and `training.csv` inside it.
- `--epochs`, `--batch-size`, `--device`, `--lr`: standard training controls.
- `--seed` / `--random-seed`: optional integer seed applied globally across Python, NumPy, and PyTorch for reproducible runs.
- `--pretrained`: enable pretrained initialization for supported torchvision backbones.
- `--use-best`: save only the best validation-loss checkpoint. By default, training saves every epoch to the final model path, so the last epoch is the final model.
- `--height`, `--width`: input dimensions used to build `input_size`.
- The terminal UI prints one completed epoch per line above the training progress bars, including training loss, validation loss, accuracy, precision, recall, and F1.

Supported standard models:

- `resnet18`
- `resnet50`
- `densenet121`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `convnext_tiny`
- `convnext_small`
- `convnext_base`
- `convnext_large`
- `draxnet`
- `drax_mobilenet_v3_large`

Parameter counts for the available standard classifiers, using the current implementations with a 1000-class classifier head:

| Model | Parameters | Special Properties |
| --- | ---: | --- |
| `efficientnet_b0` | 5,288,548 | <ul><li>Smallest standard backbone in this repo</li><li>Compound-scaled EfficientNet family</li><li>Good baseline for efficiency-focused runs</li></ul> |
| `mobilenet_v3_large` | 5,483,032 | <ul><li>Mobile-oriented architecture</li><li>Uses inverted residual blocks</li><li>Good low-parameter benchmark</li></ul> |
| `drax_mobilenet_v3_large` | 6,058,232 | <ul><li>MobileNetV3 Large backbone with a bottlenecked late-stage `DraxBlock` refiner</li><li>Preserves the pretrained MobileNet feature extractor and classifier path</li><li>Adds moderate capacity with much lower overhead than full-width Drax insertion</li></ul> |
| `densenet121` | 7,978,856 | <ul><li>Feature reuse through dense connections</li><li>Lower parameter count than ResNet-18</li><li>Strong classical CNN baseline</li></ul> |
| `resnet18` | 11,689,512 | <ul><li>Smallest ResNet variant available here</li><li>Clean apples-to-apples baseline for `draxnet`</li><li>Standard residual basic blocks</li></ul> |
| `draxnet` | 16,994,856 | <ul><li>Local `ResNet-18`-style implementation</li><li>Current default uses `DraxResidualBlock` in `layer4`</li><li>Designed for custom block experimentation</li></ul> |
| `resnet50` | 25,557,032 | <ul><li>Deeper ResNet with bottleneck blocks</li><li>Common strong baseline</li><li>Larger than `draxnet` and `resnet18`</li></ul> |
| `convnext_tiny` | 28,589,128 | <ul><li>Smallest ConvNeXt variant available here</li><li>Modern conv backbone</li><li>Larger than `resnet50` in parameter count</li></ul> |
| `convnext_small` | 50,223,688 | <ul><li>Mid-sized ConvNeXt variant</li><li>Substantially larger than `convnext_tiny`</li><li>Useful for capacity scaling comparisons</li></ul> |
| `convnext_base` | 88,591,464 | <ul><li>Large ConvNeXt backbone</li><li>High-capacity benchmark</li><li>Much heavier training/inference footprint</li></ul> |
| `convnext_large` | 197,767,336 | <ul><li>Largest backbone currently exposed</li><li>Very high parameter count</li><li>Best suited for heavyweight benchmarking</li></ul> |

### DraxNet Notes

`draxnet` is currently a local `ResNet-18`-style backbone with a configurable per-stage block layout.

`drax_mobilenet_v3_large` keeps the torchvision `mobilenet_v3_large` backbone and inserts a bottlenecked `DraxBlock` refiner after the final feature stage. This is intended for late-stage feature mixing experiments while preserving MobileNet's efficient scaffold and pretrained initialization path.

Current default:

```text
basic,basic,basic,drax
```

That means the first three stages use plain residual blocks and `layer4` uses `DraxResidualBlock`.

`Drax` stands for `Dynamic Residual Attention eXchange`.

Detailed architecture, naming, and benchmarking notes are documented in:

- [Experimental Blocks](./experimental_blocks.md)

### One-Shot Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --model siamese-le-net \
    --action train \
    --dataset ~/datasets/omniglot \
    --output ./artifacts/siamese \
    --epochs 50 \
    --batch-size 8 \
    --device cuda:0
```

Important arguments:

- `--model`: one-shot model name.
- `--dataset`: dataset root containing `train/` and `val/`.
- `--output`: output artifact directory. Training writes `{model}.pth` and `training.csv` inside it.
- `--embedding-size`: Siamese embedding width.
- `--epochs`, `--batch-size`, `--device`, `--lr`: training controls.
- `--seed` / `--random-seed`: optional integer seed applied globally across Python, NumPy, and PyTorch for reproducible runs.
- `--height`, `--width`: input dimensions used to build `input_size`.
- The terminal UI prints one completed epoch per line above the training progress bars, including training loss, validation loss, accuracy, precision, recall, and F1.

Supported one-shot models:

- `siamese-le-net`

## Available Actions

- `train`: train the selected model and write artifacts to `--output`, including `{model}.pth` and `training.csv`.
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
- In count mode, it prompts for three values: images per label for `TRAIN`, `VAL`, and `TEST`.
- In ratio mode, it prompts for three values: ratios for `TRAIN`, `VAL`, and `TEST`, then splits each label independently using those ratios.
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

To build the split dataset in one command without prompts, pass the split counts and output path:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw \
    --output ~/datasets/animals \
    --train-count 100 \
    --val-count 20 \
    --test-count 20 \
    --overwrite \
    --seed 42
```

To build a split from ratios, use `--split-mode ratios`. MLX applies the ratios within each label independently:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw \
    --split-mode ratios \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --output ~/datasets/animals \
    --overwrite \
    --seed 42
```

You can also pass only some ratio values on the command line and let MLX prompt for the rest:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw \
    --split-mode ratios \
    --train-ratio 0.7 \
    --output ~/datasets/animals \
    --seed 42
```

Non-interactive build-dataset flags:

- `--train-count`: images per label copied into `train/`
- `--val-count`: images per label copied into `val/`
- `--test-count`: images per label copied into `test/`
- `--train-ratio`: train split ratio applied within each label
- `--val-ratio`: validation split ratio applied within each label
- `--test-ratio`: test split ratio applied within each label
- `--split-mode`: `counts` or `ratios`; ratio mode splits each label independently using the provided ratios
- `--output`: destination directory for the split dataset
- `--overwrite`: replace an existing output directory without prompting
- `--seed` / `--random-seed`: global seed value; dataset splitting uses it for deterministic shuffling

Behavior notes:

- If any of `--train-count`, `--val-count`, `--test-count`, or `--output` are omitted, MLX prompts only for the missing values.
- If `--split-mode ratios` is used and any of `--train-ratio`, `--val-ratio`, `--test-ratio`, or `--output` are omitted, MLX prompts only for the missing values.
- Count mode and ratio mode are mutually exclusive.
- If `--output` already exists in non-interactive mode, MLX raises an error unless `--overwrite` is set.

Example interactive flow:

```text
How many images per label for TRAIN? 20
How many images per label for VAL? 5
How many images per label for TEST? 5
Enter output path for split dataset ~/datasets/animals-split
```

Example interactive flow in ratio mode:

```text
Train ratio? 0.7
Validation ratio? 0.15
Test ratio? 0.15
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
    --output ./benchmark-results \
    --device cpu
```

For standard classifiers, `benchmark` loads class labels from the checkpoint metadata and evaluates accuracy, average precision, average recall, F1, and ROC AUC against the labelled images in `--dataset`. It also reports per-class AUC, sensitivity, and specificity. If `--dataset` points to the dataset root and a `test/` directory exists, MLX evaluates that `test/` directory automatically.

If `--output` is set for `benchmark`, MLX writes benchmark artifacts to that directory:

- `metrics.csv`: aggregate metrics including accuracy, average precision, average recall, F1, available ROC AUC values, plus per-class AUC, sensitivity, and specificity
- `confusion_matrix.csv`: raw confusion-matrix counts with multi-class support
- `confusion_matrix.png`: rendered confusion-matrix heatmap
- `roc_curve.png`: ROC curve plot. For multi-class classification, MLX renders one-vs-rest curves per class.

### One-Shot Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action benchmark \
    --dataset ~/datasets/omniglot/test \
    --model-path ~/datasets/omniglot/checkpoints/best_epoch_10.pt \
    --output ./benchmark-results \
    --device cpu
```

For one-shot models, `benchmark` requires `--model-path` and evaluates similarity pairs built from the provided dataset directory. When `--output` is set, MLX writes the same benchmark artifacts, using binary confusion-matrix and ROC outputs.

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
