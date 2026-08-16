# Documentation

## Environment Setup

Use Python 3.10 or newer and install MLX inside a virtual environment. From the
repository root on Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

See the main [installation guide](../README.md#installation) for version checks,
verification, deactivation, and provider-specific installation alternatives.

## Modes

The repository documentation is split by CLI mode and backing package:

| Mode Package | Purpose | Docs |
| --- | --- | --- |
| `mlx.modes.object_detection` | Provider-neutral detection, Ultralytics/LibreYOLO integrations, and tracking | [Object detection](./object_detection/README.md) |
| `mlx.modes.image_classification` | Image-classification workflows for one-shot and standard classifiers | [Image classification](./image_classification/README.md) |
| `mlx.modes.segmentation` | Semantic segmentation workflows for U-Net style models | [Segmentation](./segmentation/README.md) |
| `mlx.core` | Shared exceptions and terminal UI helpers | Documented in the main [README](../README.md) |

## CLI Mapping

| CLI Mode | Backing Package |
| --- | --- |
| `object_detection` | `mlx.modes.object_detection` |
| `track` | `mlx.modes.object_detection.tracking` |
| `image_classification` | `mlx.modes.image_classification` |
| `segmentation` | `mlx.modes.segmentation` |
