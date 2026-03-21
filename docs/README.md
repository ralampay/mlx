# Documentation

## Namespaces

The repository documentation is split by feature namespace:

| Namespace | Purpose | Docs |
| --- | --- | --- |
| `mlx.features.object_detection.ultralytics` | Ultralytics training and inference workflows | [Object detection](./object_detection/README.md) |
| `mlx.features.one_shot` | Torch-based one-shot classification workflows | [One-shot image classification](./image_classification/README.md) |
| `mlx.core` | Shared exceptions, types, and terminal UI helpers | Documented in the main [README](../README.md) |
| `mlx.platforms` | Module registry and generic system actions | Documented in the main [README](../README.md) |

## CLI Modules

| CLI Module | Platform | Source Namespace |
| --- | --- | --- |
| `system` | `generic` | `mlx.platforms.system` |
| `obj-detect` | `ultralytics` | `mlx.features.object_detection.ultralytics` |
| `ic-one-shot` | `torch` | `mlx.features.one_shot` |
