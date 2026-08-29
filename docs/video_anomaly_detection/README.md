# Video Anomaly Detection

Mode: `video_anomaly_detection` (alias: `video-anomaly-detection`)

Package: `mlx.modes.video_anomaly_detection`

## Conceptual architecture and tensor shapes

The default architecture is scene/clip-level Deep SVDD over a clip-native 3D inflation of an
existing image-recognition family:

```text
[B,T,C,H,W]
    ↓ transpose
[B,C,T,H,W]
    ↓ inflated Conv3d image-recognition backbone (temporal stride 1)
[B,D,T,H',W']
    ↓ AdaptiveAvgPool3d(1)
[B,D]
    ↓ Deep SVDD projection
[B,S]
    ↓ sum((embedding - center) ** 2, dim=1)
[B] anomaly score
```

Spatial convolutions larger than 1x1 are inflated to `K×H×W`; pointwise convolutions use a
temporal kernel of one. Inflated kernels divide repeated 2D weights by `K`, and every learned
temporal stride is one, so the backbone preserves temporal resolution until global 3D pooling.
Configure `K` with `--backbone-temporal-kernel-size` (default `3`); it must be positive, odd, and
no larger than `--clip-length`.

Version-1 checkpoints remain supported through `--backbone-mode frame-2d`. That compatibility
path batches frames through the shared 2D feature backbone and uses the registered `tcn` temporal
encoder. For CLI compatibility, explicitly passing a legacy `--temporal-*` option without an
explicit backbone mode selects `frame-2d`. New runs otherwise default to `--backbone-mode 3d`.

## Normal-only training and threshold calibration

Training accepts normal clips only. It initializes the Deep SVDD center from all normal training
embeddings, keeps that center fixed during optimization, and minimizes mean squared distance to
the center. Anomaly data is neither required nor accepted in the training split.

After training, the threshold is calibrated from held-out **normal validation** scores:

```text
threshold = quantile(validation_normal_scores, svdd_quantile)
```

The exact center and threshold are stored in both deployment and resumable checkpoints. Resume
restores the center rather than recomputing it. Benchmarking and inference use the stored threshold
and never tune it on test data. Higher scores mean more anomalous behavior.

## Supported 3D backbones

The authoritative image-classification registry supplies the compatible aliases and source
weights. Each family has a dedicated clip-native 3D class. Current aliases are:

| Model alias | 3D class | Feature dimension |
| --- | --- | ---: |
| `resnet18` | `ResNet3DBackbone` | 512 |
| `resnet50` | `ResNet3DBackbone` | 2048 |
| `densenet121` | `DenseNet1213DBackbone` | 1024 |
| `mobilenet_v3_large` | `MobileNetV3Large3DBackbone` | 960 |
| `efficientnet_b0` | `EfficientNetB03DBackbone` | 1280 |
| `convnext_tiny` | `ConvNeXt3DBackbone` | 768 |
| `convnext_small` | `ConvNeXt3DBackbone` | 768 |
| `convnext_base` | `ConvNeXt3DBackbone` | 1024 |
| `convnext_large` | `ConvNeXt3DBackbone` | 1536 |
| `draxnet` | `DraxNet3D` | 512 |
| `drax_mobilenet_v3_large` | `DraxMobileNetV3Large3D` | 960 |

Use `--action ls-models` for 3D class names, feature dimensions, parameter counts, and both
`average` and `sknet` configurations for each Drax family. Models are selected by the
classification registry's `standard` family. All `siamese-*` and other one-shot/few-shot models
are excluded automatically. Pretrained RGB backbones use the same ImageNet mean/std normalization
as image classification. Compatible 2D weights are inflated for 3D initialization. Standard
families record `inflated_full` provenance; Drax variants record `inflated_partial` because their
Drax-specific branches have no ImageNet source weights and remain freshly initialized.

List the live registry without downloading pretrained weights:

```bash
python -m mlx --mode video_anomaly_detection --action ls-models
```

The listing emits separate `average` and `sknet` configurations for each Drax family. Select one
during training with `--drax-fusion-mode average` or `--drax-fusion-mode sknet`:

```bash
python -m mlx --mode video_anomaly_detection --action train \
  --model draxnet --drax-fusion-mode sknet \
  --dataset ./prepared --output ./artifacts/draxnet-sknet
```

## Generic dataset layout

The first training/evaluation loader consumes extracted image sequences:

```text
dataset/
├── train/
│   └── normal/
│       └── clip001/001.tif ...
├── val/
│   └── normal/
│       └── clip001/001.tif ...
└── test/
    ├── normal/
    │   └── clip001/001.tif ...
    └── anomaly/
        └── clip002/001.tif ...
```

Frame names are sorted deterministically. Numeric stems are preserved as frame indices; other
names use their zero-based sequence positions. A sample is `[T,C,H,W]`. `--frame-stride` controls
sampling inside each clip and complete windows slide one source frame at a time. Sources too short
for `(T - 1) * stride + 1` frames are excluded; frames are not padded or repeated. Metadata retains
source, start/end frames, and sampled frame indices.

Therefore, the training frame-buffer length is exactly `T = --clip-length`. It contains sampled
frames, while the corresponding decoded/source span is `(T - 1) * --frame-stride + 1` frames.

Compressed video files are intentionally not a training-dataset input in this first version.
Extract them into source directories. Direct video decoding is supported by `infer-video`, and the
dataset interface remains separable for a future video-file adapter.

The prepared frame-sequence layout may be stored as a ZIP in S3 and staged for local training:

```bash
python -m mlx --mode video_anomaly_detection --action train \
  --model resnet18 --dataset-s3-uri s3://my-datasets/avenue-prepared.zip \
  --output ./artifacts/avenue --clip-length 16 --profile mlx-training
```

The ZIP must contain exactly one root (optionally below a wrapper directory) with
`train/normal` and `val/normal`; test data may be included. The normal-only validation and
training safeguards are applied after staging exactly as they are for local paths. Install
`.[aws]` for Boto3. MLX safely extracts and caches the object under
`~/.cache/mlx/datasets`, configurable with `--dataset-cache-dir`, and writes
`dataset_source.json` with the training artifacts. Do not also pass `--dataset`.
See the shared [S3 dataset training guide](../s3-dataset-training.md) for credentials, cache
identity, safe extraction, provenance, local-versus-SageMaker behavior, and troubleshooting.

## UCSD Ped2 preparation example

Ped2 training clips contain normal behavior only. Map each training sequence to
`train/normal/<sequence>/`, reserve normal-only sequences or windows for `val/normal/`, and place
test sequences under `test/normal/` or `test/anomaly/` according to their temporal labels. For
example:

```bash
mkdir -p UCSDped2-prepared/{train/normal,val/normal,test/normal,test/anomaly}
cp -R UCSDped2/Train/Train001 UCSDped2-prepared/train/normal/Train001
cp -R UCSDped2/Train/Train016 UCSDped2-prepared/val/normal/Train016
```

Ped2 test sequences can contain both normal and anomalous intervals. For correct frame-level
research metrics, split mixed sequences into label-consistent source directories (while retaining
the original frame numbers), or generate the generic layout with an external preparation script.
Ground-truth pixel masks are not required; spatial localization is outside this version.

## Training

```bash
python -m mlx \
  --mode video_anomaly_detection \
  --action train \
  --model resnet18 \
  --backbone-mode 3d \
  --backbone-temporal-kernel-size 3 \
  --dataset ./UCSDped2-prepared \
  --output ./artifacts/ped2-resnet18-3d-svdd \
  --clip-length 16 --frame-stride 1 \
  --height 224 --width 224 \
  --svdd-dim 128 --svdd-hidden-dim 256 --svdd-quantile 0.95 \
  --epochs 50 --batch-size 8 --lr 0.001 \
  --pretrained --device cuda:0 --seed 42
```

Artifacts are:

```text
<output>/
├── <backbone>-3d-svdd.pth
├── <backbone>-3d-svdd.last.pth
├── training.csv
├── training_history.png
└── run_metadata.json
```

The deployment checkpoint stores full reconstruction metadata, 3D class and mode, temporal kernel
and stride policy, global pooling, model state, feature dimensions, input/window configuration,
labels, score type, fixed center, calibrated threshold, quantile, exact pretrained provenance, and
MLX version. The `.last.pth` additionally stores optimizer, completed epoch, history, best
validation objective, and Python/NumPy/PyTorch RNG state. Drax artifact names also include their
fusion mode.

## Resume

Keep the same architecture/window configuration and set the original `.last.pth` as `--model-path`.
`--epochs` is the new total target, not an additional count:

```bash
python -m mlx --mode video_anomaly_detection --action train \
  --model resnet18 --dataset ./UCSDped2-prepared --output ./artifacts/ped2 \
  --model-path ./artifacts/ped2/resnet18-3d-svdd.last.pth --epochs 100
```

The checkpoint determines whether the reconstructed model is 3D or the legacy frame-2D+TCN
architecture. A version-1 checkpoint without `backbone_mode` automatically selects `frame-2d`
unless the caller explicitly requests an incompatible mode.

To start a new legacy-compatible run rather than resume one:

```bash
python -m mlx --mode video_anomaly_detection --action train \
  --model resnet18 --backbone-mode frame-2d --temporal-model tcn \
  --dataset ./prepared --output ./artifacts/legacy-tcn
```

## Benchmarking and artifacts

```bash
python -m mlx --mode video_anomaly_detection --action benchmark \
  --model-path ./artifacts/ped2/resnet18-3d-svdd.pth \
  --dataset ./UCSDped2-prepared/test \
  --output ./artifacts/ped2/benchmark --batch-size 8 --device cuda:0
```

Clip metrics include AUROC, average precision/AUPRC, specificity/normal acceptance, sensitivity/
anomaly recall, precision, F1, balanced accuracy, false-positive/false-negative rates, threshold,
confusion counts, and normal/anomaly score statistics. Overlapping window scores are also
aggregated per sampled frame with `--frame-aggregation mean` (default) or `max`; frame metrics are
reported separately.

```text
benchmark/
├── metrics.json
├── metrics.csv
├── run_metadata.json
├── predictions.csv
├── predictions.jsonl
├── frame_predictions.csv
├── roc_curve.csv
├── pr_curve.csv
├── frame_roc_curve.csv
├── frame_pr_curve.csv
├── roc_curve.png
├── pr_curve.png
├── score_distribution.png
├── confusion_matrix.png
└── benchmark_report.md
```

Provenance includes the checkpoint SHA-256, backbone mode and class, dataset, dimensions, window
configuration, device, batch size, threshold, score type, timestamp, and evaluator version.

## Video inference

Inference is headless and uses a sliding temporal window over the complete decoded frame:

```bash
python -m mlx --mode video_anomaly_detection --action infer-video \
  --model-path ./model.pth --file-path ./sample.mp4 --output ./inference
```

It writes `predictions.jsonl` and `predictions.csv` with start/end frame and time, sampled frame
indices, anomaly score, stored threshold, and anomaly decision. No object detector is used.

The core fields in each JSONL record follow this shape (records also retain sampled frame indices):

```json
{
  "start_frame": 120,
  "end_frame": 135,
  "start_time_seconds": 4.0,
  "end_time_seconds": 4.5,
  "anomaly_score": 1.82,
  "threshold": 0.73,
  "is_anomaly": true
}
```

## Interpretation and limitations

- Higher squared-distance score means more anomalous relative to learned normal clips.
- The threshold comes only from normal validation data.
- An anomaly decision does not identify the semantic incident type.
- The model is scene/clip-level rather than object-centric.
- Spatial pixel-level anomaly localization is not implemented.
- Training/benchmark datasets currently require extracted frame sequences; compressed files are
  accepted by video inference only.
- Pretraining is inflated from 2D ImageNet weights; native video/Kinetics weights are not supplied.
- Clip-native 3D backbones use more memory and compute than the legacy frame-2D path.
- Labels assigned by the generic directory layout are sequence/window labels. Mixed-label videos
  should be split or prepared with label-consistent temporal segments for valid frame metrics.
