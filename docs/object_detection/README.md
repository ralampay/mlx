# Object Detection

Mode: `object-detection`

Package: `mlx.modes.object_detection.ultralytics`

## Overview

This mode provides the Ultralytics-backed object-detection workflow exposed by:

```bash
python -m mlx --mode object-detection
```

The source is split by responsibility:

- `runner.py`: action dispatch.
- `training.py`: training workflow.
- `inference.py`: webcam and video inference.
- `utils.py`: model-path resolution, model initialization, and annotation helpers.

## Dataset Format

Training accepts any of these dataset sources:

- a local YOLO dataset root containing `data.yaml`
- a direct dataset YAML path
- a built-in Ultralytics dataset alias such as `coco8` or `coco128`

For this repository, `coco8` is the best default example dataset. It is small, ships with an auto-download manifest in Ultralytics, and is fast enough for smoke-testing both `yolo26` and `draxnet-yolo26`.

If you want a slightly less trivial quick-start dataset, use `coco128`. For real training, point `--dataset` to your own YOLO-format dataset root.

Local dataset example:

Minimal example:

```yaml
path: ../
train: images/train
val: images/val
names:
  0: class-a
  1: class-b
```

Pass `--dataset-path` as the directory that contains `data.yaml`, or pass `--dataset coco8` / `--dataset coco128` to use a built-in dataset YAML.

### Expected Local Directory Structure

When `--dataset` points to a local directory, `mlx` expects that directory itself to be the YOLO dataset root. In practice, this means `data.yaml` must live directly inside the path you pass on the command line.

Example:

```text
my-detection-dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/          # optional
└── labels/
    ├── train/
    ├── val/
    └── test/          # optional
```

Typical command:

```bash
python -m mlx \
    --mode object-detection \
    --action train \
    --dataset /path/to/my-detection-dataset \
    --model draxnet-yolo26 \
    --epochs 100 \
    --batch-size 16 \
    --device cuda:0
```

That command assumes:

- `/path/to/my-detection-dataset/data.yaml` exists
- `data.yaml` references paths relative to that dataset root, such as `images/train` and `images/val`
- each image in `images/<split>/` has a matching YOLO label file in `labels/<split>/`

Example `data.yaml` for that layout:

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: class-a
  1: class-b
```

The important rule is simple: pass the directory that contains `data.yaml`, not the `images/` directory and not the `labels/` directory.

## Model Selection

`--model` accepts either a YAML path or one of the friendly aliases resolved by `mlx`:

- `yolo26`
- `yolov26`
- `draxnet-yolo26`

`draxnet-yolo26` maps to the custom DraxNet backbone YAML added in the `ralampay/ultralytics` fork.

## Training

Baseline YOLO26 example:

```bash
python -m mlx \
    --mode object-detection \
    --action train \
    --dataset coco8 \
    --model yolo26 \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0 \
    --output ./runs/yolo26
```

DraxNet-backed YOLO26 example:

```bash
python -m mlx \
    --mode object-detection \
    --action train \
    --dataset coco8 \
    --model draxnet-yolo26 \
    --epochs 10 \
    --batch-size 8 \
    --device cuda:0 \
    --output ./runs/draxnet-yolo26
```

Important arguments:

- `--dataset` / `--dataset-path`: required dataset source. Use `coco8` for the documented smoke-test path.
- `--model`: required architecture YAML or alias such as `yolo26` or `draxnet-yolo26`.
- `--model-path`: optional checkpoint for warm-start training.
- `--epochs`, `--batch-size`, `--device`: core training controls.
- `--pretrained`: enable Ultralytics pretrained initialization.
- `--lr0`, `--optimizer`, `--nbs`, `--warmup-epochs`, `--loss-clip`, `--amp`: trainer overrides.
- `--output`: optional Ultralytics project directory. If omitted for a local dataset root, runs go under `<dataset>/runs`. Otherwise they default to `./runs/object_detection`.
- `--run-name`: output folder name inside the Ultralytics project directory.

## Webcam Inference

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-camera \
    --model draxnet-yolo26 \
    --model-path ./runs/draxnet-yolo26/exp/weights/best.pt \
    --device cpu \
    --confidence 0.35 \
    --camera-index 0
```

Important arguments:

- `--model`: required architecture YAML or alias for rebuilding the network.
- `--model-path`: required trained checkpoint.
- `--confidence`: minimum confidence threshold to render.
- `--camera-index`: OpenCV camera device index.

## Video Inference

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-video \
    --model draxnet-yolo26 \
    --model-path ./runs/draxnet-yolo26/exp/weights/best.pt \
    --file-path ~/videos/sample.mp4 \
    --device cpu \
    --confidence 0.35
```

Additional arguments:

- `--file-path`: required video path.
- `--device`: inference backend.
- `--confidence`: rendered detection threshold.

## Dependencies

- `ultralytics` from the `ralampay/ultralytics` fork, because `draxnet-yolo26` is defined there
- `opencv-python` for webcam or video inference

Run `python -m mlx --help` for the full CLI reference.
