# MLX (Machine Learning eXecutor)

A CLI for computer-vision workflows.

The terminal experience uses the Python `rich` text UI library for interactive prompts, status panels, tables, and runtime summaries.

## Usage

All commands share a common signature. Pick the `--module` you want to run, select a `--platform`, and supply any module-specific arguments. Example:

```bash
mlx --module system --action ls-env
```

The sections below detail the built-in modules, their supported platforms, and the key parameters you can tweak.

## Modules

* [Object Detection](./docs/object_detection/index.md)
* [One-Shot Image Classification (Torch platform)](./docs/image_classification/index.md)

Segmentation is not implemented in the current codebase yet.

Run `mlx --help` to view the rich help screen and available options.

## Environment Setup

Copy the provided template and populate the values required for your workspace:

```bash
cp .env.dist .env
```

- `ROBOFLOW_API_KEY`: Optional key for Roboflow-backed dataset workflows.

The CLI loads `.env` automatically on startup. You can confirm the current values (masked where appropriate) with:

```bash
mlx --module system --action ls-env
```

Set any additional variables you rely on in the same `.env` file or through your preferred secrets manager.

## Packages

Install the dependencies listed below (package names target PyPI):

- beautifulsoup4 (`bs4`)
- matplotlib
- numpy
- opencv-python (`cv2`)
- Pillow (`PIL`)
- python-dotenv (`dotenv`)
- requests
- rich
- scikit-learn (`sklearn`)
- torch
- torchvision
- tqdm
- ultralytics (`https://github.com/ralampay/ultralytics`)
