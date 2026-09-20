# Saliency Mapping

Mode: `saliency_mapping` (alias: `saliency-mapping`)

Package: `mlx.modes.saliency_mapping`

## Task definition

This mode performs still-image salient object detection (SOD):

```text
RGB image → U-Net encoder/decoder → [B, 1, H, W] logits → sigmoid → [0, 1]
```

Zero means low saliency and one means high saliency. Unlike semantic segmentation,
targets are continuous grayscale maps, not class IDs; the model is not configured
as two classes and does not use softmax or argmax. This mode does not perform
fixation prediction, eye tracking, video saliency, or optical flow.

The formulation follows the SOD setting used by *BASNet: Boundary-Aware Salient
Object Detection* and *U²-Net: Going Deeper with Nested U-Structure for Salient
Object Detection*. Those works provide literature grounding only. MLX intentionally
tests its existing U-Net and Drax families and does not implement BASNet or U²-Net.

## Models and model groups

The mode calls the segmentation U-Net factory with one output channel. It does not
copy the encoder, decoder, skip connections, backbone adapters, or Drax blocks.
Supported identifiers are:

- `unet`
- `unet-resnet18`, `unet-resnet50`
- `unet-densenet121`
- `unet-mobilenet_v3_large`, `unet-efficientnet_b0`
- `unet-convnext_tiny`, `unet-convnext_small`, `unet-convnext_base`, `unet-convnext_large`
- `unet-draxnet-average`, `unet-draxnet-sknet`
- `unet-drax_mobilenet_v3_large-average`, `unet-drax_mobilenet_v3_large-sknet`

Drax average and SKNet fusion retain their existing mathematical behavior. Standard
torchvision backbones and Drax-MobileNet retain existing `--pretrained` semantics.
DraxNet is reported as not supporting pretrained initialization for its default
Drax stage.

`all` resolves every registered architecture. `all-small` is deliberately identical
to segmentation's group:

- `unet-mobilenet_v3_large`
- `unet-efficientnet_b0`
- `unet-drax_mobilenet_v3_large-average`
- `unet-drax_mobilenet_v3_large-sknet`

The selectors work for smoke tests, benchmarks, and sequential grouped training.

```bash
python -m mlx --mode saliency-mapping --action ls-models
python -m mlx --mode saliency-mapping --action ls-models --format json
python -m mlx --mode saliency-mapping --action test --model all-small
python -m mlx --mode saliency-mapping --action test --model all
```

`ls-models` reports name, parameter count, encoder/backbone, pretrained support,
group membership, and the one-channel output contract. Smoke tests assert finite
`[B, 3, H, W] → [B, 1, H, W]` logits.

## Dataset and builder

Training expects files matched by stem:

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Images become `FloatTensor[C,H,W]`. Targets are read as grayscale, divided by 255,
and returned as `FloatTensor[1,H,W]`. Continuous gray levels are preserved and are
never converted to class IDs. Resize uses linear interpolation for both continuous
images and targets. Paired random/center crops share coordinates and zero padding.

`build-dataset` copies an unsplit `images/` + `masks/` source into partitions without
rewriting or quantizing saliency maps:

```bash
python -m mlx --mode saliency-mapping --action build-dataset \
  --dataset ./data/duts-raw --output ./data/duts \
  --train-count 8000 --val-count 1000 --test-count 1000 --seed 42
```

## Loss, training, and checkpoints

The BASNet-style hybrid objective is:

```text
loss = bce_weight × BCEWithLogitsLoss
     + ssim_weight × SSIM loss(sigmoid(logits), target)
     + iou_weight × IoU loss(sigmoid(logits), target)
```

The model itself never embeds sigmoid. Configure components with `--bce-weight`,
`--ssim-weight`, and `--iou-weight`; each defaults to 1.0. Training history exposes
the combined loss and all three components for train and validation.

```bash
python -m mlx --mode saliency-mapping --action train \
  --model unet-drax_mobilenet_v3_large-average \
  --dataset ./data/duts --output ./artifacts/saliency \
  --width 320 --height 320 --batch-size 8 --epochs 50 --device cuda
```

Training writes the lowest-validation-MAE checkpoint, resumable `.last` checkpoint,
`training.csv`, `training_curves.png`, and `training_config.json`. The last checkpoint
stores optimizer state, random state, history, model identity, input/transform
metadata, and best validation MAE. Resume with `--model-path` and set `--epochs` to
the new total. A valid test split also produces up to 16 qualitative sample panels.

Grouped training is sequential, scratch-only, fail-fast, and requires an empty
output plus a test split. It writes per-model artifacts and root `all-models.json`
and `leaderboard.csv`, ranked by test MAE.

## Metrics and benchmark artifacts

The mode reports these metrics per image and in aggregate:

- MAE on continuous probabilities and continuous targets
- max F-beta and its best threshold
- mean F-beta across the threshold sweep
- precision and recall at the best-F-beta threshold
- complete threshold rows and precision-recall curve data

F-measure uses conventional SOD `beta² = 0.3`. Threshold metrics binarize targets
at 0.5; MAE retains soft targets. `--threshold-steps` controls the inclusive 0-to-1
sweep.

```bash
python -m mlx --mode saliency-mapping --action benchmark \
  --model unet --model-path ./artifacts/unet/unet.pth \
  --dataset ./data/duts --split test --output ./artifacts/unet/benchmark
```

The benchmark writes:

- `metrics.json`, `metrics.csv`, `image_metrics.csv`
- `threshold_metrics.csv`, `precision_recall.csv`
- `timing.csv`, `run_metadata.json`
- threshold/PR plots and per-image distributions when plots are enabled
- `predictions/` grayscale probability maps, `heatmaps/`, and `overlays/`
- `samples/{original,ground_truth,prediction,overlay,panels}/`

Runtime fields include parameter count, wall/forward throughput, latency statistics,
and peak allocated memory on CUDA. FLOPs are not reported because segmentation has
no current FLOP counter.

For group comparison, a directory passed as `--model-path` is searched using
`<root>/<model>/<model>.pth`, with `<root>/<model>.pth` also accepted. Without a
model path, group benchmarking is explicitly an untrained architecture comparison.
Root `comparison.csv` and `comparison.json` contain MAE, F-beta, parameters, latency,
throughput, and memory fields for each model.

```bash
python -m mlx --mode saliency-mapping --action benchmark \
  --model all-small --model-path ./artifacts/all-small \
  --dataset ./data/duts --output ./artifacts/all-small-benchmark
```

## Image inference

```bash
python -m mlx --mode saliency-mapping --action infer-image \
  --model-path ./artifacts/unet/unet.pth --input-img ./sample.jpg \
  --output ./saliency-output
```

Inference is headless and writes `probability.npy` (float32 continuous values),
`probability.png`, `heatmap.png`, and `overlay.png`. The Python result also returns
the probability array directly, keeping it usable by future patch-ranking workflows.

## Known limitations and deliberate differences

- S-measure, E-measure, and weighted F-measure are not yet supported. No fragile
  approximation is published under those standard names.
- Camera/video inference is intentionally absent; this mode covers still images.
- Saliency uses one-channel BCE/SSIM/IoU and lowest validation MAE instead of
  segmentation cross-entropy, class metrics, or best foreground Dice.
- Semantic-segmentation confusion, class-ID, calibration, and boundary metrics are
  not copied because they do not represent the continuous SOD contract.
- BASNet and U²-Net architectures, fixation prediction, SAM, DINO, and patch
  ranking are not implemented.
