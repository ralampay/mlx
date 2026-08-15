# Documentation

## Modes

The repository documentation is split by CLI mode and backing package:

| Mode Package | Purpose | Docs |
| --- | --- | --- |
| `mlx.modes.object_detection` | Provider-neutral detection, Ultralytics integration, and tracking | [Object detection](./object_detection/README.md) |
| `mlx.modes.image_classification` | Image-classification workflows for one-shot and standard classifiers | [Image classification](./image_classification/README.md) |
| `mlx.modes.segmentation` | Semantic segmentation workflows for U-Net style models | [Segmentation](./segmentation/README.md) |
| `mlx.core` | Shared exceptions and terminal UI helpers | Documented in the main [README](../README.md) |

## CLI Mapping

| CLI Mode | Backing Package |
| --- | --- |
| `object_detection` | `mlx.modes.object_detection` |
| `image_classification` | `mlx.modes.image_classification` |
| `segmentation` | `mlx.modes.segmentation` |
