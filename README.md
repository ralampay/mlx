# MLX (Machine Learning eXecutor)

A CLI that wraps several ML/DL workflows behind a consistent interface.

## Usage

All commands share a common signature. Pick the `--module` you want to run, select a `--platform`, and supply any module-specific arguments. Example:

```bash
mlx --module chat --platform openai --model gpt-4o-mini
```

The sections below detail every built-in module, their supported platforms, and the key parameters you can tweak.

## Modules

### Chat (OpenAI platform)

```bash
mlx --module chat \
    --platform openai \
    --model gpt-4o-mini \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 40
```

- `--module chat`: Selects the conversational agent.
- `--platform openai`: Routes to the OpenAI SDK integration.
- `--model`: OpenAI chat model name.
- `--temperature`: Controls response creativity.
- `--top-p`: Nucleus sampling probability cutoff.
- `--top-k`: Retained token count (ignored by OpenAI but accepted for parity).

### One-Shot Image Classification (Torch platform)

```bash
mlx --module ic-one-shot \
    --platform torch \
    --model siamese-le-net \
    --action train \
    --dataset-path ~/datasets/omniglot \
    --epochs 50 \
    --batch-size 8 \
    --device cuda:0
```

- `--module ic-one-shot`: Runs the Siamese network pipeline.
- `--platform torch`: Uses the PyTorch-backed implementation.
- `--model`: Model architecture (currently `siamese-le-net`).
- `--action`: Workflow to execute (`train`, `test`, `benchmark`, `infer-image`, `build-dataset`).
- `--dataset-path`: Root folder for the dataset (supports build/test modes).
- `--epochs`, `--batch-size`: Training loop controls.
- `--device`: Target device (`cpu`, `cuda:0`, etc.).
- Additional options (`--embedding-size`, `--input-img`, etc.) propagate to the trainer.

### Object Detection (Ultralytics platform)

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

- `--module obj-detect`: Activates the Ultralytics YOLO trainer.
- `--platform ultralytics`: Uses the `ralampay/ultralytics` fork.
- `--action`: Currently supports `train`.
- `--dataset-path`: Directory containing the YOLO-formatted dataset (`data.yaml`, images, labels).
- `--model`: Architecture YAML to instantiate the network (required when starting from scratch).
- `--model-path`: Optional weights (`.pt`) to warm start training; omit to train from random init.
- `--pretrained/--no-pretrained`: Whether Ultralytics should fetch pretrained weights for the provided YAML (default: disabled).
- `--lr0`: (Optional) Override the initial learning rate if the default is too high when training from scratch.
- `--optimizer`: Optimizer selection (`auto`, `adamw`, `sgd`, `adam`, `rmsprop`, etc.).
- `--nbs`: Nominal batch size used for Ultralytics’ LR scaling (default `64`).
- `--warmup-epochs`: Control the length of the LR warmup phase (default `3`).
- `--amp/--no-amp`: Toggle mixed-precision training.
- `--loss-clip`: Optional gradient clipping threshold.
- `--epochs`, `--batch-size`, `--device`: Standard training controls.
- `--height`, `--width`: Influence the YOLO `imgsz` (largest dimension wins).

For live camera inference with trained weights:

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

- `--model`: Path to the model YAML (architecture definition).
- `--model-path`: Trained weights (`.pt`) to load.
- `--confidence`: Detection confidence threshold.
- `--camera-index`: OpenCV camera index (`0` for default webcam).
- `--file-path`: Required for `infer-video`; supplies the video source.

For running inference on a video file:

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

For more options, run:

```bash
mlx --help
```

## Environment Variables

Some integrations expect API credentials to be present in your shell environment:

- `OPENAI_API_KEY`: Required for chat sessions via the OpenAI platform. Generate a key from the OpenAI dashboard and export it, e.g. `export OPENAI_API_KEY=sk-...`.
- `ROBOFLOW_API_KEY`: Needed when building or downloading datasets from Roboflow for the one-shot and object-detection workflows. Set it with `export ROBOFLOW_API_KEY=...`.

Ensure these are configured (or provided through your secrets manager) before invoking the corresponding modules.
