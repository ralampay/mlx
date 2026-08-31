# One-Class Image Recognition

Mode: `image_recognition_oc` (alias: `image-recognition-oc`)

Package: `mlx.modes.image_recognition_oc`

## Overview

This mode learns a normal-image distribution and classifies individual images as `normal` or
`anomaly`. The initial one-class model is `deep-svdd`. It uses any standard image-classification
model as a feature backbone; Siamese/one-shot variants are intentionally excluded. The one-class
algorithm and backbone are selected separately so later algorithms do not need to redefine the
backbone inventory.

```text
image -> standard 2D feature backbone -> Deep SVDD projection
      -> squared distance from fixed normal center -> anomaly score
```

## Dataset contract

Training and validation contain normal images only. Benchmarking requires both labels:

```text
dataset/
├── train/
│   └── normal/
├── val/
│   └── normal/
└── test/
    ├── normal/
    └── anomaly/
```

Images may be nested recursively below each label directory. Supported extensions are PNG, JPEG,
BMP, TIFF, and TIF. A non-empty `train/anomaly` or `val/anomaly` directory is rejected instead of
being silently ignored. The same layout can be supplied as an S3 ZIP for local training through
`--dataset-s3-uri`.

## Models and training

List every compatible configuration:

```bash
python -m mlx --mode image_recognition_oc --action ls-models
```

Train Deep SVDD with a standard classifier backbone:

```bash
python -m mlx --mode image_recognition_oc --action train \
  --model deep-svdd --backbone resnet18 \
  --dataset ./dataset --output ./artifacts/one-class \
  --epochs 50 --batch-size 16 --lr 0.001 \
  --svdd-dim 128 --svdd-hidden-dim 256 --svdd-quantile 0.95 \
  --pretrained --seed 42
```

The entire backbone and SVDD projection are optimized. Before the first epoch, MLX initializes a
fixed center from all unaugmented normal training embeddings. Training minimizes mean squared
distance to that center. Best-checkpoint selection uses mean normal-validation score. After
training, the best and resumable checkpoints receive thresholds calibrated independently from the
normal validation-score quantile.

Artifacts use the one-class model and backbone names:

```text
artifacts/one-class/
├── resnet18-deep-svdd.pth
├── resnet18-deep-svdd.last.pth
├── training.csv
├── training_history.png
└── run_metadata.json
```

Resume by increasing the total epoch target and supplying the `.last.pth` file to the normal
training action:

```bash
python -m mlx --mode image_recognition_oc --action train \
  --model deep-svdd --backbone resnet18 \
  --dataset ./dataset --output ./artifacts/one-class \
  --model-path ./artifacts/one-class/resnet18-deep-svdd.last.pth \
  --epochs 100
```

Resume requires matching algorithm, backbone, dimensions, image size, color mode, and Drax fusion
configuration. It restores optimizer, history, best objective, fixed center, and random state.

## Image inference

```bash
python -m mlx --mode image_recognition_oc --action infer-image \
  --model-path ./artifacts/one-class/resnet18-deep-svdd.pth \
  --input-img ./sample.jpg
```

Checkpoint metadata controls the algorithm, backbone, preprocessing, center, and threshold.
Explicit `--model` or `--backbone` values are compatibility assertions. The returned result contains
`predicted_label`, `is_anomaly`, `anomaly_score`, and `threshold`. A score equal to the threshold is
normal; a strictly greater score is anomalous.

## Benchmarking

```bash
python -m mlx --mode image_recognition_oc --action benchmark \
  --model-path ./artifacts/one-class/resnet18-deep-svdd.pth \
  --dataset ./dataset --output ./benchmark
```

Benchmarking uses the stored threshold and never recalibrates on test images. It reports AUROC,
AUPRC, precision, recall, specificity, F1, balanced accuracy, acceptance/detection rates, confusion
counts, and score statistics. The output directory contains metrics and predictions in JSON/CSV,
ROC and precision-recall data, plots, a score histogram, confusion matrix, checkpoint provenance,
and a Markdown report. Use `--no-plots` to retain machine-readable artifacts without PNG files.

## Current boundaries

- Execution is local only; SageMaker lifecycle actions are not yet implemented.
- Inference accepts one image at a time.
- Deep SVDD is the only registered one-class algorithm in this release.
- Checkpoints are mode-specific and cannot be loaded as image-classification or video-anomaly
  checkpoints.
