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
- `adapters.py`: runtime adapters for Ultralytics `.pt` and ONNX Runtime `.onnx` inference.
- `conversion.py`: Ultralytics `.pt` to ONNX export.
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

If you see `Model YAML not found: draxnet-yolo26`, your installed `ultralytics` package does not
currently expose `draxnet-yolo26.yaml`. In that case, reinstall the pinned dependency for this
repo, pass a direct filesystem path to that YAML, or switch to `--model yolo26`.

## End-to-End Workflow

The typical object-detection deployment path in this repository is:

1. Train with Ultralytics using a `.yaml` model definition and produce a `.pt` checkpoint.
2. Convert that trained `.pt` checkpoint to `.onnx`.
3. Run camera or video inference from the exported `.onnx` model through ONNX Runtime.

Concrete example:

### Step 1: Train with Ultralytics

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

Expected result:

- Ultralytics writes a training run under `./runs/draxnet-yolo26/...`
- the trained checkpoint is typically available at `./runs/draxnet-yolo26/<run-name>/weights/best.pt`

### Step 2: Convert the trained checkpoint to ONNX

```bash
python -m mlx \
    --mode object-detection \
    --action convert \
    --model-path ./runs/draxnet-yolo26/exp/weights/best.pt \
    --output ./exports \
    --device cpu
```

Expected result:

- MLX loads the Ultralytics `.pt` checkpoint
- Ultralytics exports an ONNX model
- the final ONNX file is written to `./exports/best.onnx`

### Step 3: Use the exported ONNX model for inference

Camera inference:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-camera \
    --model-path ./exports/best.onnx \
    --device cpu \
    --confidence 0.35 \
    --camera-index 0
```

Video inference:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-video \
    --model-path ./exports/best.onnx \
    --file-path ~/videos/sample.mp4 \
    --device cpu \
    --confidence 0.35
```

Important notes for this workflow:

- `--model` is required for training because Ultralytics needs the model YAML or alias.
- `--model` is not required for `.onnx` inference; MLX switches to ONNX Runtime when `--model-path` ends in `.onnx`.
- if you want the ONNX export beside the checkpoint instead of under `./exports`, omit `--output`.

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
- `--use-best` / `--no-use-best`: select `weights/best.pt` after training by default. Use `--no-use-best` to prefer `weights/last.pt` instead.
- `--pretrained`: enable Ultralytics pretrained initialization.
- `--lr0`, `--optimizer`, `--nbs`, `--warmup-epochs`, `--loss-clip`, `--amp`: trainer overrides.
- `--output`: optional Ultralytics project directory. If omitted for a local dataset root, runs go under `<dataset>/runs`. Otherwise they default to `./runs/object_detection`.
- `--run-name`: output folder name inside the Ultralytics project directory.

If `--model-path` is not provided and the chosen `--output` directory already contains checkpoints, MLX now reuses them automatically. It resumes from the newest `last.pt` it finds under that output tree; if only other `.pt` files are present, it warm-starts from the newest one instead of starting from random initialization.

When a resumable `last.pt` is found, the trainer explicitly reports that it is continuing the existing run and shows the checkpoint path. The training summary also marks the run as `continue existing run` instead of `new run`.

After training completes, MLX selects the checkpoint to use downstream. By default `--use-best` is enabled, so object detection selects `weights/best.pt` when Ultralytics writes it, stores that path in the returned training result as `model_path` / `checkpoint_path`, and prints the selected checkpoint. If `--no-use-best` is passed, MLX prefers `weights/last.pt`.

After training completes, MLX now prints a final validation-metrics table when Ultralytics exposes the values. For detection runs, the most important metrics are typically precision, recall, `mAP@0.50`, `mAP@0.50:0.95`, and fitness; train/validation loss terms may also appear. ROC/AUC is only shown when the underlying metrics object reports it, which is uncommon for standard object-detection validation.

MLX also now writes extra training graphs into the resolved run directory, alongside Ultralytics artifacts. When available, this includes:

- `loss_curves.png`: train/validation loss curves from `results.csv`
- `detection_metrics.png`: precision, recall, F1, `mAP@0.50`, and `mAP@0.50:0.95`
- `learning_rate.png`: learning-rate schedule curves
- `speed_metrics.png`: per-epoch speed metrics if Ultralytics records them
- `per_class_map.csv`: per-class `mAP@0.50` and `mAP@0.50:0.95` values
- `per_class_map50.csv`: per-class `mAP@0.50` values
- `per_class_map50.png`: bar chart of per-class `mAP@0.50`
- `per_class_map50_95.csv`: per-class `mAP@0.50:0.95` values
- `per_class_map50_95.png`: bar chart of per-class `mAP@0.50:0.95`

Ultralytics plotting is also explicitly enabled for training runs, so native artifacts such as PR/F1/P/R curves and confusion-matrix plots should continue to land in the same run directory when supported by the installed Ultralytics version.

## Webcam Inference

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-camera \
    --model-path ./exports/best.onnx \
    --device cpu \
    --confidence 0.35 \
    --camera-index 0
```

Important arguments:

- `--model`: required only when `--model-path` points to a PyTorch checkpoint (`.pt`).
- `--model-path`: required trained checkpoint or exported `.onnx` model.
- `--confidence`: minimum confidence threshold to render.
- `--camera-index`: OpenCV camera device index.

## Video Inference

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action infer-video \
    --model-path ./exports/best.onnx \
    --file-path ~/videos/sample.mp4 \
    --device cpu \
    --confidence 0.35
```

Additional arguments:

- `--file-path`: required video path.
- `--device`: inference backend. `.pt` uses Ultralytics; `.onnx` uses ONNX Runtime.
- `--confidence`: rendered detection threshold.

## ONNX Conversion

Example:

```bash
python -m mlx \
    --mode object-detection \
    --action convert \
    --model-path ./runs/draxnet-yolo26/exp/weights/best.pt \
    --output ./exports \
    --device cpu
```

Important arguments:

- `--model-path`: required Ultralytics PyTorch checkpoint (`.pt`) to export.
- `--output`: optional destination directory or explicit `.onnx` file path. If omitted, MLX writes the ONNX file beside the checkpoint.
- `--height`, `--width`: optional export image size. If equal, MLX passes a square `imgsz`; otherwise it passes the `(height, width)` tuple to Ultralytics.
- `--device`: export backend such as `cpu` or `cuda:0`.

## Dependencies

- `ultralytics` from the `ralampay/ultralytics` fork, because `draxnet-yolo26` is defined there
- `onnxruntime` for `.onnx` inference
- `opencv-python` for webcam or video inference

Run `python -m mlx --help` for the full CLI reference.
