# Documentation

## Modes

The repository documentation is split by CLI mode and backing package:

| Mode Package | Purpose | Docs |
| --- | --- | --- |
| `mlx.modes.object_detection.ultralytics` | Ultralytics training and inference workflows | [Object detection](./object_detection/README.md) |
| `mlx.modes.one_shot` | Image-classification workflows, including one-shot models | [Image classification](./image_classification/README.md) |
| `mlx.core` | Shared exceptions and terminal UI helpers | Documented in the main [README](../README.md) |

## CLI Mapping

| CLI Mode | Backing Package |
| --- | --- |
| `object-detection` | `mlx.modes.object_detection.ultralytics` |
| `image-classification` | `mlx.modes.one_shot` |
