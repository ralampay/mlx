# One-Shot Image Classification

## Overview

The `ic-one-shot` module runs the Siamese network pipeline on Torch and expects a dataset with a typical `images/` and `labels/` layout (a dataset builder is available via `--action build-dataset`). You must point the command to the dataset root with `--dataset-path` and then trigger the `train` action.

## Sample Training Command

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

Key arguments:

- `--module ic-one-shot`: Selects the one-shot classification workflow.
- `--platform torch`: Runs on the PyTorch-backed backend.
- `--model`: Neural architecture to instantiate (`siamese-le-net` is the default).
- `--action train`: Switches the workflow to training mode (other actions include `test`, `benchmark`, `infer-image`, `build-dataset`).
- `--dataset-path`: Path to your dataset root (used for all actions that read or build the data).
- `--epochs`, `--batch-size`: Control the training loop.
- `--device`: Target compute device (`cpu`, `cuda:0`, etc.).
- Additional trainer-specific flags (like `--embedding-size` or `--input-img`) are accepted and forwarded transparently.
