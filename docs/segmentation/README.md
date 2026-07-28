# Segmentation

Mode: `segmentation`

Package: `mlx.modes.segmentation`

## Overview

This mode provides semantic segmentation workflows exposed by:

```bash
python -m mlx --mode segmentation
```

The source is organized by responsibility:

- `runner.py`: default config and action dispatch.
- `train.py`: training loop and smoke test.
- `inference.py`: single-image, webcam, and video inference.
- `data.py`: paired image/mask dataset loading and preprocessing.
- `data.py` also contains the interactive dataset split builder.
- `models/`: segmentation model registry, U-Net decoders, and classification
  backbone adapters.
- `utils.py`: checkpoint metadata, metrics, and shared helpers.
- `presentation.py`: rich tables and OpenCV visualization helpers.

## Supported Models

- `unet`: basic U-Net for semantic segmentation
- `unet-resnet18`, `unet-resnet50`
- `unet-densenet121`
- `unet-mobilenet_v3_large`, `unet-efficientnet_b0`
- `unet-convnext_tiny`, `unet-convnext_small`, `unet-convnext_base`,
  `unet-convnext_large`
- `unet-draxnet-average`, `unet-draxnet-sknet`
- `unet-drax_mobilenet_v3_large-average`,
  `unet-drax_mobilenet_v3_large-sknet`

The backbone models reuse the corresponding image-classification feature
extractors and replace their pooling/classification heads with a common U-Net
decoder. Average and SKNet DRAX fusion are separate model identifiers so runs,
checkpoints, and parameter counts remain unambiguous.

## Dataset Format

Training expects a paired image/mask dataset root:

```text
<dataset-root>/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Requirements:

- image and mask filenames must match by stem
- masks must be single-channel class-index masks
- binary segmentation uses `0` for background and any nonzero value for foreground
- multiclass segmentation expects values in `0..num_classes-1`

Image preprocessing:

- read as RGB
- resize to `--width` x `--height`
- convert to `float32`
- divide by `255.0`

Mask preprocessing:

- read as single-channel grayscale
- resize with nearest-neighbor interpolation
- keep as integer class indices
- do not normalize mask values

## Available Actions

- `train`: train the selected segmentation model and write the best checkpoint to `--output`
- `benchmark`: evaluate a trained checkpoint and optionally write research artifacts
- `test`: run a random-tensor smoke test for the configured model
- `infer-image`: run inference for one image and display original, predicted mask, and overlay
- `infer-camera`: run webcam inference with segmentation overlay
- `infer-video`: run file-based video inference with segmentation overlay
- `build-dataset`: interactively split a flat paired image/mask dataset into `train`, `val`, and `test`
- `ls-models`: list registered segmentation models and their total parameter counts

## Build Dataset

`build-dataset` is used to convert a flat paired segmentation dataset into a split dataset with `train/`, `val/`, and `test/`.

Required flag:

- `--dataset-path`: source dataset root. This must point to the unsplit dataset you want to process.

Expected input layout:

```text
<source-dataset>/
├── images/
│   ├── sample_001.png
│   ├── sample_002.tiff
│   └── ...
└── masks/
    ├── sample_001.png
    ├── sample_002.tiff
    └── ...
```

Rules:

- image and mask filenames must match by stem
- image and mask extensions may differ as long as the stem matches
- supported file extensions include `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`

Behavior:

- the command scans the source `images/` and `masks/` directories
- it validates that pairs exist for every image and mask by stem
- it prints a pair summary
- it prompts for the number of samples to place in `TRAIN`, `VAL`, and `TEST`
- it prompts for the output path where the split dataset should be created
- if the output directory already exists, MLX asks for confirmation before overwriting it

Example command:

```bash
python -m mlx \
    --mode segmentation \
    --action build-dataset \
    --dataset ./data/kvasir-seg-raw
```

Example interactive flow:

```text
How many paired samples for TRAIN? 800
How many paired samples for VAL? 100
How many paired samples for TEST? 100
Enter output path for split dataset ./data/kvasir-seg-split
```

This creates a dataset like:

```text
./data/kvasir-seg-split/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Training

Example:

```bash
python -m mlx \
    --mode segmentation \
    --action train \
    --model unet \
    --dataset ./data/kvasir-seg \
    --output ./results/unet-seg \
    --width 256 \
    --height 256 \
    --batch-size 4 \
    --epochs 50 \
    --device cpu \
    --num-classes 2 \
    --class-names background,foreground
```

Important arguments:

- `--model`: one of the segmentation model names listed above
- `--pretrained`: initialize supported classification backbones with torchvision
  weights; DraxNet variants require non-pretrained initialization, while DRAX
  MobileNet can initialize its MobileNet backbone from pretrained weights
- `--dataset`: dataset root containing `train/` and `val/`
- `--output`: artifact directory, or a legacy `.pt`/`.pth` checkpoint path
- `--num-classes`: number of output classes expected in the masks
- `--class-names`: optional comma-separated names matching `--num-classes`
- `--epochs`, `--batch-size`, `--device`, `--lr`: training controls
- `--width`, `--height`: input dimensions used to build `input_size`

List every comparison model and its total parameter count for the selected
class count:

```bash
python -m mlx \
    --mode segmentation \
    --action ls-models \
    --num-classes 2
```

Train an explicit SKNet fusion variant:

```bash
python -m mlx \
    --mode segmentation \
    --action train \
    --model unet-draxnet-sknet \
    --dataset ./data/kvasir-seg \
    --output ./results/unet-draxnet-sknet \
    --num-classes 2
```

Directory output writes:

- `unet.pth`: best validation-loss checkpoint
- `unet.best-dice.pth`: best non-background validation-Dice checkpoint
- `unet.last.pth`: resumable model, optimizer, random state, and history
- `training.csv`: one row per epoch with loss, aggregate overlap, and per-class metrics
- `training_curves.png`: loss and validation metric curves
- `training_config.json`: effective configuration and dataset sizes

For compatibility, a file output such as `./checkpoints/unet-seg.pt` remains the
best-loss checkpoint. Its best-Dice and last checkpoints are written beside it,
and training research files are written under `./checkpoints/unet-seg-research/`.

Resume by passing the generated last checkpoint through `--model-path`. `--epochs`
is the total target epoch count:

```bash
python -m mlx \
    --mode segmentation \
    --action train \
    --dataset ./data/kvasir-seg \
    --output ./results/unet-seg \
    --model-path ./results/unet-seg/unet.last.pth \
    --epochs 100
```

## Benchmarking

```bash
python -m mlx \
    --mode segmentation \
    --action benchmark \
    --model-path ./results/unet-seg/unet.pth \
    --dataset ./data/kvasir-seg \
    --split test \
    --output ./results/unet-seg/benchmark \
    --device cpu
```

The dataset may be a split root containing `train/`, `val/`, and `test/`, or a
direct paired directory containing `images/` and `masks/`. Root evaluation uses
the explicit `--split`, which defaults to `test`.

The benchmark reports and saves:

- aggregate pixel accuracy, macro/micro/weighted precision, recall, specificity,
  F1/Dice, IoU/Jaccard, foreground Dice/IoU, frequency-weighted IoU,
  generalized Dice, Cohen kappa, and multiclass MCC
- per-class confusion counts, support, predictive values, sensitivity,
  specificity, error rates, balanced accuracy, Dice, IoU, MCC, area error,
  ROC AUC, average precision, and PR AUC
- cross-entropy/NLL, Brier scores, confidence, entropy, ECE, and MCE
- boundary precision/recall/F1, surface Dice, Hausdorff distance, HD95, and
  average symmetric surface distance in resized-image pixels
- wall/forward timing, throughput, latency percentiles, and accelerator memory
- binary threshold analysis controlled by `--threshold-steps` and
  `--mask-threshold`

Research output includes `metrics.csv`, `metrics.json`, `class_metrics.csv`,
`image_metrics.csv`, `timing.csv`, `run_metadata.json`, confusion matrices,
ROC/PR/calibration curve data and plots, threshold artifacts for binary models,
metric distributions, and `worst_cases.csv`. By default it also writes predicted
class-ID masks, overlays, and error maps under `predictions/`; pass
`--no-save-images` for metrics-only runs.

Class ratios with no valid denominator are saved as `nan` and excluded from
macro averages. Both-empty boundary masks score perfectly, while a one-sided
empty boundary has zero overlap and undefined distance metrics.

## Smoke Test

Example:

```bash
python -m mlx \
    --mode segmentation \
    --action test \
    --model unet \
    --width 256 \
    --height 256 \
    --batch-size 2 \
    --num-classes 2
```

## Single Image Inference

Example:

```bash
python -m mlx \
    --mode segmentation \
    --action infer-image \
    --model-path ./checkpoints/unet-seg.pt \
    --input-img ./samples/image.jpg
```

This opens a window with three panels:

- original image
- predicted segmentation mask
- overlay of the mask on the original image

## Camera Inference

Example:

```bash
python -m mlx \
    --mode segmentation \
    --action infer-camera \
    --model-path ./checkpoints/unet-seg.pt \
    --camera-index 0 \
    --device cpu
```

Important arguments:

- `--model-path`: required trained checkpoint
- `--camera-index`: OpenCV camera device index
- `--overlay-alpha`: blend strength for the mask overlay

## Video Inference

Example:

```bash
python -m mlx \
    --mode segmentation \
    --action infer-video \
    --model-path ./checkpoints/unet-seg.pt \
    --file-path ./samples/video.mp4 \
    --device cpu
```

Additional arguments:

- `--file-path`: required video path
- `--overlay-alpha`: blend strength for the mask overlay

## Dependencies

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tqdm`

Run `python -m mlx --help` for the full CLI reference.
