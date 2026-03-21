# Object Detection

Namespace: `mlx.features.object_detection.ultralytics`

## Overview

This namespace contains the Ultralytics-backed object-detection workflow exposed by:

```bash
mlx --module obj-detect --platform ultralytics
```

The source is split by responsibility:

- `runner.py`: action dispatch.
- `training.py`: training workflow.
- `inference.py`: webcam and video inference.
- `utils.py`: model-path resolution, model initialization, and annotation helpers.

## Dataset Format

Training expects a YOLO-style dataset root containing `data.yaml` plus the usual image and label folders referenced by that manifest.

Minimal example:

```yaml
path: ../
train: images/train
val: images/val
names:
  0: class-a
  1: class-b
```

Pass `--dataset-path` as the directory that contains `data.yaml`.

## Training

Example:

```bash
mlx --module obj-detect \
    --platform ultralytics \
    --action train \
    --dataset-path ~/datasets/roboflow-yolo \
    --model ultralytics/cfg/models/ext/cad_yolo12.yaml \
    --epochs 100 \
    --batch-size 16 \
    --device cuda:0
```

Important arguments:

- `--dataset-path`: required YOLO dataset root.
- `--model`: required architecture YAML.
- `--model-path`: optional checkpoint for warm-start training.
- `--epochs`, `--batch-size`, `--device`: core training controls.
- `--pretrained`: enable Ultralytics pretrained initialization.
- `--lr0`, `--optimizer`, `--nbs`, `--warmup-epochs`, `--loss-clip`, `--amp`: trainer overrides.
- `--run-name`: output folder name under `<dataset-path>/runs`.

## Webcam Inference

Example:

```bash
mlx --module obj-detect \
    --platform ultralytics \
    --action infer-camera \
    --model ultralytics/cfg/models/ext/cad_yolo12.yaml \
    --model-path ./runs/train/weights/best.pt \
    --device cpu \
    --confidence 0.35 \
    --camera-index 0
```

Important arguments:

- `--model`: required architecture YAML for rebuilding the network.
- `--model-path`: required trained checkpoint.
- `--confidence`: minimum confidence threshold to render.
- `--camera-index`: OpenCV camera device index.

## Video Inference

Example:

```bash
mlx --module obj-detect \
    --platform ultralytics \
    --action infer-video \
    --model ultralytics/cfg/models/ext/cad_yolo12.yaml \
    --model-path ./runs/train/weights/best.pt \
    --file-path ~/videos/sample.mp4 \
    --device cpu \
    --confidence 0.35
```

Additional arguments:

- `--file-path`: required video path.
- `--device`: inference backend.
- `--confidence`: rendered detection threshold.

## Dependencies

- `ultralytics`
- `opencv-python` for webcam or video inference

Run `mlx --help` for the full CLI reference.
