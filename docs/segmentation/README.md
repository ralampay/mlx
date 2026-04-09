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
- `models/`: segmentation model registry and the basic `unet`.
- `utils.py`: checkpoint metadata, metrics, and shared helpers.
- `presentation.py`: rich tables and OpenCV visualization helpers.

## Supported Models

- `unet`: basic U-Net for semantic segmentation

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
- `test`: run a random-tensor smoke test for the configured model
- `infer-image`: run inference for one image and display original, predicted mask, and overlay
- `infer-camera`: run webcam inference with segmentation overlay
- `infer-video`: run file-based video inference with segmentation overlay
- `build-dataset`: interactively split a flat paired image/mask dataset into `train`, `val`, and `test`

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
    --output ./checkpoints/unet-seg.pt \
    --width 256 \
    --height 256 \
    --batch-size 4 \
    --epochs 50 \
    --device cpu \
    --num-classes 2
```

Important arguments:

- `--model`: segmentation model name, currently `unet`
- `--dataset`: dataset root containing `train/` and `val/`
- `--output`: output checkpoint path
- `--num-classes`: number of output classes expected in the masks
- `--epochs`, `--batch-size`, `--device`, `--lr`: training controls
- `--width`, `--height`: input dimensions used to build `input_size`

The checkpoint stores segmentation metadata, so inference can reconstruct the model without re-supplying the model type.

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

Run `python -m mlx --help` for the full CLI reference.
