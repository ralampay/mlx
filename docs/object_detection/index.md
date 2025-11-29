# Object Detection

## Ultralytics

### Overview

The Ultralytics workflow is powered by the `ralampay/ultralytics` fork that is already pinned in this repository's `requirements.txt`/`pyproject.toml`. Object detection runs via `mlx --module obj-detect --platform ultralytics` and expects datasets that follow the YOLO format. Before every run, make sure you have the dependencies installed (`pip install -r requirements.txt` will pull in `ultralytics` plus the optional OpenCV dependency for streaming inference).

### Preparing a YOLO dataset

Every training run points at a root dataset directory containing a `data.yaml` manifest plus the usual `images/` and `labels/` subfolders. A minimal `data.yaml` looks like:

```yaml
path: ../
train: images/train
val: images/val
names:
  0: class-a
  1: class-b
```

The `--dataset-path` argument should reference the folder that contains this manifest. During training, the module writes Ultralytics checkpoints and logs under `<dataset-path>/runs/<run-name>`, so make sure that directory is writable.

### Training

Point `--model` at the architecture YAML you want to train (you can reuse one of the YAML files shipped with Ultralytics, for example `ultralytics/cfg/models/ext/cad_yolo12.yaml`, or supply your own). Pass the dataset path via `--dataset-path` and adjust device/epochs/etc. as needed:

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

Key arguments:

- `--dataset-path`: Directory that includes `data.yaml`, `images/`, and `labels/`.
- `--model`: Required architecture YAML. Paths can reference relative files, absolute paths, or files under the Ultralytics package (e.g., `ultralytics/cfg/models/ext/cad_yolo12.yaml`).
- `--model-path`: Optional `.pt` checkpoint to warm-start training; omit to train from scratch.
- `--pretrained` / `--no-pretrained`: Toggle Ultralytics’ built-in pretrained weights (default: disabled when starting from scratch).
- `--lr0`: Override the initial learning rate defined in the YAML, useful when training from a random initialization.
- `--optimizer`, `--nbs`, `--warmup-epochs`, `--loss-clip`: Adjust optimizer selection (auto/adamw/sgd/…), nominal batch size scaling, warmup length, and gradient clipping.
- `--amp` / `--no-amp`: Enable or disable mixed-precision (Ultralytics defaults to `amp` on when available).
- `--height` / `--width`: Control the effective `imgsz` (the higher of `height`/`width` is used).
- `--epochs`, `--batch-size`, `--device`: Standard training controls.
- `--run-name`: Override the default `mlx-ultralytics` run folder under `<dataset-path>/runs`.

### Inference on Webcam

Camera inference requires OpenCV (`pip install opencv-python`) and a set of trained weights (`--model-path`). Use the `infer-camera` action to stream detections directly from your webcam:

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

What to set:

- `--model`: Architecture YAML that produced the weights (required so Ultralytics can rebuild the network).
- `--model-path`: Path to the trained `.pt` checkpoint (`./runs/train/weights/best.pt` is the default output of Ultralytics training).
- `--device`: Device for inference (`cpu` is acceptable for webcams, CUDA for faster throughput).
- `--confidence`: Minimum detection confidence to annotate.
- `--camera-index`: OpenCV camera ID (0 for built-in webcams, up to your system's camera count).

### Inference on Video

For file-based inference, switch the action to `infer-video` and provide a `--file-path`:

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

Additional considerations:

- `--file-path`: Required video to process.
- `--confidence`: Controls which boxes are rendered.
- `--device`: The backend that Ultralytics uses for inference; keep it consistent with the weights.
- `--model` and `--model-path` behave the same as in camera mode.

Run `mlx --help` to list every available option for the object detection module.
