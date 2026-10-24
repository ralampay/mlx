# Object-detection adapters

MLX owns adapter implementations, registration, injection and serialization under
`mlx.modes.object_detection.feature_adapters`. Detector-specific placement and
checkpoint reconstruction live at the LibreYOLO integration boundary. LibreYOLO's
legacy adapter package remains compatibility support. See
[adapter ownership](adapter-ownership.md) and [architecture](../../ARCHITECTURE.md).

Adapter workflows accept explicit checkpoint, dataset and output paths. No action
selects a machine-specific Desktop path. Reusable commands remain available from
Python; fixed DAWN priority/paired study recipes and their launchers have been removed.

## Prepare and verify inputs

Verify a compatible foundation checkpoint without supplying training paths:

```bash
python -m mlx --mode object-detection --action adapter-verify \
  --model yolox-l --checkpoint /path/to/foundation.pt
```

The dataset-specific DAWN converter remains available:

```bash
python -m mlx --mode object-detection --action adapter-prepare \
  --dataset /path/to/dawn/original --output /path/to/dawn/processed --seed 42
```

For other prepared datasets, supply a directory containing `manifest.json`,
`data.yaml`, and train/val/test image and label directories. The manifest's class
order must agree with the YAML and, for same-taxonomy training, the foundation.
Taxonomy transfer uses the separate transfer workflow and explicit head-reset policy.

## Train adapters

Choose methods, seeds and training settings for your workload:

```bash
python -m mlx --mode object-detection --action adapter-experiment \
  --model yolox-l --checkpoint /path/to/foundation.pt \
  --dataset /path/to/prepared-data --output /path/to/results \
  --methods lora --experiment-seeds 42 --epochs 12 --batch-size 1 --device cuda
```

`adapter-calibrate` accepts the same explicit input paths. Supported adapter names
come from the feature-adapter registry. `frozen`, `head-only` and `full-finetune`
are training strategies. Adapter rank, reduction, alpha and placement remain
explicit options. Existing initialization defaults and checkpoint formats are preserved.

Completed compatible runs can be reused; incompatible configuration or provenance
must use a separate output directory. `--baseline-study` selects an explicit compatible
baseline. Model selection uses validation metrics, not test metrics.

## Reports and inference

Generate a report from completed runs without the training dataset or checkpoint:

```bash
python -m mlx --mode object-detection --action adapter-report \
  --output /path/to/results --comparison-method lora
```

The existing weather-slice analysis is specialized to DAWN-style filenames.
`adapter-slice-predict` requires explicit checkpoint, dataset and output paths.
After prediction caching, `adapter-slice-report` requires only `--output`; it reads
cached ground truth and predictions without reopening the foundation or dataset.

Adapter-only checkpoints can be loaded with a compatible foundation and `--adapter`.
Self-contained exported adapter detectors load through `--model-path` without a
separate foundation. See [adapter ownership](adapter-ownership.md) for Python export
and loading commands, and [residual fusion](drax-residual-fusion.md) for that adapter's
configuration and reconstruction contract.
