# Image Classification

Mode: `image_classification`

Package: `mlx.modes.image_classification`

## Overview

This mode provides the image-classification workflow exposed by:

```bash
python -m mlx --mode image_classification
```

The source is organized by responsibility:

- `runner.py`: default config and action dispatch.
- `train.py`: training loops and smoke tests for one-shot and standard classifiers.
- `evaluation.py`: benchmark flows for both model families.
- `inference.py`: image inference for both model families.
- `ood/`: Deep SVDD center initialization, scoring, loss, and threshold calibration.
- `cam.py`: Grad-CAM, AblationCAM, and ScoreCAM rendering for trained checkpoints.
- `data.py`: dataset loading, dataset building, and shared image preprocessing.
- `models/`: Siamese and standard classification model builders.
- `utils.py`: model resolution and checkpoint metadata helpers.
- `presentation.py`: rich tables and one-shot match rendering.
- `data.py` exposes `ImageClassificationDataset`, which expects a dataset root containing `train/` and `val/`.

## Supported Models

This mode supports two training setups:

- One-shot similarity models: `siamese-le-net` plus `siamese-` variants of every standard model listed below, such as `siamese-resnet18` and `siamese-draxnet`
- Standard classification models: `resnet18`, `resnet50`, `densenet121`, `mobilenet_v3_large`, `efficientnet_b0`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `draxnet`, `drax_mobilenet_v3_large`

The selected `--model` determines which training, benchmarking, and inference path is used. Torchvision-backed standard models are loaded by name and their classifier heads are adapted to the dataset class count. Additional custom standard classifiers can be plugged in later through the model registry.

List every registered model and its total parameter count without downloading pretrained weights:

```bash
python -m mlx --mode image_classification --action ls-models
```

The counts use `--num-classes` for standard classifier heads and `--embedding-size` for one-shot models.

All standard classifiers share the same preprocessing family:

- Training: `Resize`, `ToTensor`, `Normalize`. Add `--apply-transformations` to include `RandomHorizontalFlip` and `RandomRotation(10)`.
- Validation / benchmark / inference: `Resize`, `ToTensor`, `Normalize`
- RGB normalization uses ImageNet mean/std: `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`
- Grayscale normalization uses mean/std: `(0.5,)` / `(0.5,)`

Experimental custom-block documentation:

- [Experimental Blocks](./experimental_blocks.md)

## Dataset Expectations

### Training

Training expects a dataset root with at least:

```text
<dataset-root>/
├── train/
│   └── <label>/
└── val/
    └── <label>/
```

For one-shot training and evaluation, each label directory must contain at least two images so positive pairs can be generated.

This same split layout is used by both model families:

- Standard classifiers use `train/` and `val/` as supervised class datasets.
- One-shot models use `train/` and `val/` to generate positive and negative image pairs.

The standard classification path uses `ImageClassificationDataset(dataset_path, split="train" | "val", transform=...)`, which returns `(x, y)` pairs where `x` is the transformed image tensor and `y` is the label index.

### Dataset Builder

`build-dataset` expects a flat label-organized source dataset:

```text
<source-dataset>/
└── <label>/
    ├── image-1.png
    └── image-2.png
```

It creates `train/`, `val/`, and `test/` splits in a new output directory. You can run it interactively or pass counts/ratios and an output path for a non-interactive build.

## Training

For image-classification training, `--output` is treated as an artifact directory. MLX creates the directory if needed and writes:

```text
<output>/
├── {model}.pth
├── {model}.last.pth
└── training.csv
```

`{model}.pth` is the best validation-loss model when `--use-best` is enabled.
`{model}.last.pth` is a resumable training-state checkpoint written after every
completed epoch. It includes the model, Adam optimizer, best validation loss,
epoch history, and random-number-generator state.

`training.csv` contains one row per completed epoch with this schema:

```text
epoch,train_loss,val_loss,accuracy,precision,recall,f1
```

Joint Deep SVDD runs extend this schema with separate train/validation classification and SVDD losses. Ordinary runs retain the original columns exactly.

```text
epoch,train_loss,train_classification_loss,train_svdd_loss,val_loss,val_classification_loss,val_svdd_loss,accuracy,precision,recall,f1
```

Resume an interrupted run by passing its last checkpoint and the original
total target epoch count:

```bash
python -m mlx \
    --mode image_classification \
    --action train \
    --model resnet18 \
    --dataset ~/datasets/animals \
    --output ./artifacts/resnet18 \
    --epochs 50 \
    --model-path ./artifacts/resnet18/resnet18.last.pth
```

Training continues at the next epoch and reconstructs `training.csv` from the
checkpoint history. The model, family, labels, input size, color mode, and Drax
fusion mode must match the checkpoint.

### Standard Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --model resnet18 \
    --action train \
    --dataset ~/datasets/animals \
    --output ./artifacts/resnet18 \
    --epochs 50 \
    --batch-size 16 \
    --device cuda:0 \
    --lr 0.001
```

Important arguments:

- `--model`: standard classifier such as `resnet18` or `resnet50`.
- `--dataset`: dataset root containing `train/` and `val/`.
- `--output`: output artifact directory. Training writes `{model}.pth`, `{model}.last.pth`, and `training.csv` inside it.
- `--model-path`: optional `{model}.last.pth` checkpoint used to resume at the next epoch.
- `--epochs`, `--batch-size`, `--device`, `--lr`: standard training controls.
- `--seed` / `--random-seed`: optional integer seed applied globally across Python, NumPy, and PyTorch for reproducible runs.
- `--pretrained`: enable pretrained initialization for supported torchvision backbones.
- `--use-best`: save only the best validation-loss checkpoint. By default, training saves every epoch to the final model path, so the last epoch is the final model.
- `--height`, `--width`: input dimensions used to build `input_size`.
- The terminal UI prints one completed epoch per line above the training progress bars, including training loss, validation loss, accuracy, precision, recall, and F1.

Supported standard models:

- `resnet18`
- `resnet50`
- `densenet121`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `convnext_tiny`
- `convnext_small`
- `convnext_base`
- `convnext_large`
- `draxnet`
- `drax_mobilenet_v3_large`

### Optional Joint Deep SVDD

Every standard classifier above supports optional joint Deep SVDD training. It is disabled by default, so existing models, commands, checkpoints, outputs, and classification behavior remain unchanged. One-shot Siamese models do not use this option.

The joint model computes one shared penultimate feature vector and sends it to two independent branches:

```text
shared image features
├── existing classification head -> class logits
└── independent projection head -> Deep SVDD embedding
```

The projection head prevents the global compactness objective from acting directly on the exact representation that must separate the classes. All labeled images in `train/` are in-distribution examples; no OOD or negative-image directory is required or implicitly added.

For a chest-X-ray dataset, valid images can contain classes `a`, `b`, and `c`. The classifier distinguishes those labels while the OOD gate can reject natural images, dog images, unrelated radiology modalities, and corrupted images.

Ordinary classification training remains:

```bash
python -m mlx \
  --mode image_classification \
  --action train \
  --model draxnet \
  --dataset ./dataset \
  --output ./artifacts/model
```

Enable joint training explicitly:

```bash
python -m mlx \
  --mode image_classification \
  --action train \
  --model draxnet \
  --dataset ./chest-xray-dataset \
  --output ./artifacts/draxnet-svdd \
  --ood-method deep-svdd \
  --svdd-weight 0.05 \
  --svdd-dim 128 \
  --svdd-hidden-dim 256 \
  --svdd-quantile 0.95 \
  --svdd-warmup-epochs 0 \
  --epochs 50 \
  --device cuda:0
```

Arguments and defaults:

- `--ood-method none`: use ordinary classification. Set `deep-svdd` to enable the joint model.
- `--svdd-weight 0.05`: contribution of mean squared distance to the fixed center.
- `--svdd-dim 128`: SVDD embedding dimension.
- `--svdd-hidden-dim 256`: projection-head hidden dimension.
- `--svdd-quantile 0.95`: quantile of valid validation scores used as the acceptance threshold; it must be strictly between zero and one.
- `--svdd-warmup-epochs 0`: number of initial classification-only epochs.

Before the first joint epoch, MLX initializes a fixed center from all training-set embeddings. Resumable checkpoints restore that exact center and never silently recompute it. Best-checkpoint selection uses joint validation loss (`classification loss + svdd weight * SVDD loss`) only when Deep SVDD is enabled; ordinary runs continue to use the existing classification validation loss.

Resume joint training with the same OOD method and projection dimensions:

```bash
python -m mlx \
  --mode image_classification \
  --action train \
  --model draxnet \
  --dataset ./chest-xray-dataset \
  --output ./artifacts/draxnet-svdd \
  --model-path ./artifacts/draxnet-svdd/draxnet.last.pth \
  --ood-method deep-svdd \
  --svdd-dim 128 \
  --svdd-hidden-dim 256 \
  --epochs 100
```

The checkpoint OOD method, embedding dimension, hidden dimension, class labels, input size, color mode, model family, and Drax fusion mode must match the resumed run. An ordinary checkpoint cannot be resumed as a Deep SVDD model; MLX reports this as a user-facing compatibility error.

After training, MLX reloads the selected deployment checkpoint and calibrates its threshold from valid `val/` images—not test or OOD images. The deployment checkpoint stores the center in both the model state and readable `ood` metadata, plus the calibrated threshold, quantile, dimensions, and squared-Euclidean score type. Intermediate `.last.pth` checkpoints may have no threshold and cannot be used for OOD-gated inference until calibration is completed.

The added checkpoint metadata has this shape (the surrounding checkpoint retains MLX's existing keys such as `state_dict`, `family`, and `model_config`):

```python
{
    "ood": {
        "method": "deep-svdd",
        "center": center_tensor,
        "threshold": 0.73,  # None before calibration
        "quantile": 0.95,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "score_type": "squared_euclidean",
    }
}
```

The center also appears as the registered `svdd_center` buffer in `state_dict`. Ordinary checkpoints omit the `ood` block, preserving the pre-SVDD checkpoint layout.

Run inference with the final checkpoint:

```bash
python -m mlx \
  --mode image_classification \
  --action infer-image \
  --model draxnet \
  --model-path ./artifacts/draxnet-svdd/draxnet.pth \
  --input-img ./sample.png \
  --device cuda:0
```

#### Determining whether an image is OOD

For a Deep SVDD checkpoint, MLX calculates the squared Euclidean distance between the image's SVDD embedding and the fixed training-distribution center:

```text
ood_score = sum((svdd_embedding - svdd_center) ** 2)
```

It then compares that score with the validation-calibrated threshold stored in the checkpoint:

```text
accepted = ood_score <= ood_threshold
```

- `accepted: true` means the score is at or below the threshold, so the image is treated as in-distribution and `predicted_label` can be used.
- `accepted: false` means the score exceeds the threshold, so the image is treated as out-of-distribution. In that case, `predicted_label` and `confidence` are `None`, and `rejection_reason` is `out_of_distribution`.
- Lower scores indicate greater proximity to the learned training-image center. Higher scores indicate greater dissimilarity.

When calling `infer_image_classification` from Python, inspect `accepted` as the primary decision field:

```python
from mlx.modes.image_classification.inference import infer_image_classification

result = infer_image_classification(
    {
        "model": "draxnet",
        "model_path": "./artifacts/draxnet-svdd/draxnet.pth",
        "input_img": "./sample.png",
        "device": "cpu",
    }
)

if result["accepted"]:
    print(
        f"In-distribution: {result['predicted_label']} "
        f"(score={result['ood_score']:.4f}, "
        f"threshold={result['ood_threshold']:.4f})"
    )
else:
    print(
        f"OOD: score={result['ood_score']:.4f} exceeds "
        f"threshold={result['ood_threshold']:.4f}"
    )
```

The CLI renders an explicit rejection message when `accepted` is false. For automation, rely on the returned `accepted` boolean instead of comparing rounded display values. The raw `ood_score` and `ood_threshold` fields are also returned for logging and monitoring.

An accepted result includes the trusted class prediction:

```python
{
    "accepted": True,
    "predicted_label": "a",
    "confidence": 0.91,
    "ood_score": 0.42,
    "ood_threshold": 0.73,
    "rejection_reason": None,
}
```

A rejected result does not expose a trusted class label:

```python
{
    "accepted": False,
    "predicted_label": None,
    "confidence": None,
    "ood_score": 4.81,
    "ood_threshold": 0.73,
    "rejection_reason": "out_of_distribution",
}
```

If a Deep SVDD checkpoint has not been calibrated, inference stops instead of choosing a threshold implicitly:

```text
The checkpoint contains a Deep SVDD model, but no calibrated rejection threshold. Run threshold calibration or use a final deployment checkpoint.
```

Deep SVDD does not mathematically guarantee rejection of every unseen input. Acceptance means only: “The image is sufficiently similar to the validated training-image distribution.” It does not prove medical validity, diagnostic correctness, or semantic suitability. Calibration quality depends on representative training and validation data, and distribution shifts close to the learned boundary can still be accepted. Joint Deep SVDD is currently limited to standard classifiers; Siamese training does not use the OOD branch, and CAM generation currently expects an ordinary classification checkpoint.

## OOD Training and Benchmarking

### Training an OOD-enabled classifier

Use the normal labelled `train/` and `val/` splits. Every class under these splits is considered valid and in-distribution for Deep SVDD:

```text
./chest-xray-dataset/
├── train/
│   ├── a/
│   ├── b/
│   └── c/
├── val/
│   ├── a/
│   ├── b/
│   └── c/
└── test/
    ├── a/
    ├── b/
    └── c/
```

Do not put known OOD examples in `train/` or `val/`. Joint training does not need negative examples, and `val/` is used to calibrate the rejection threshold after the selected checkpoint is loaded.

```bash
python -m mlx \
  --mode image_classification \
  --action train \
  --model resnet18 \
  --dataset ./chest-xray-dataset \
  --output ./artifacts/resnet18-svdd \
  --ood-method deep-svdd \
  --svdd-weight 0.05 \
  --svdd-dim 128 \
  --svdd-hidden-dim 256 \
  --svdd-quantile 0.95 \
  --epochs 50 \
  --device cuda:0
```

Use the final `resnet18.pth` for inference and OOD benchmarking. The resumable `resnet18.last.pth` may not contain a calibrated threshold.

### Preparing an OOD benchmark

A meaningful OOD benchmark needs two held-out collections that were not used for training or threshold calibration:

```text
./ood-benchmark/
├── id/
│   └── held-out valid chest X-rays
└── ood/
    ├── natural images
    ├── dog images
    ├── unrelated radiology modalities
    └── corrupted images
```

The ID collection measures how often valid inputs are accepted. The OOD collection measures how often unrelated inputs are rejected. Keep these sets separate from `train/` and `val/`; in particular, do not adjust `--svdd-quantile` after looking at final benchmark results.

The normal `benchmark` action continues to report classification metrics from class logits. It can be run on the held-out ID split to confirm that accepted-distribution classification remains useful:

```bash
python -m mlx \
  --mode image_classification \
  --action benchmark \
  --model resnet18 \
  --model-path ./artifacts/resnet18-svdd/resnet18.pth \
  --dataset ./chest-xray-dataset/test \
  --batch-size 16 \
  --device cuda:0
```

That command is a classification benchmark; it does not treat the labelled test classes as OOD. To measure OOD rejection, score both benchmark collections with the threshold already stored in the final checkpoint:

```python
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from mlx.modes.image_classification.data import iter_dataset_images, load_image_tensor
from mlx.modes.image_classification.models.joint_svdd import JointDeepSVDDClassifier
from mlx.modes.image_classification.utils import load_checkpoint_bundle

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, metadata = load_checkpoint_bundle(
    {
        "model": "resnet18",
        "model_path": "./artifacts/resnet18-svdd/resnet18.pth",
        "device": device,
    }
)

if not isinstance(model, JointDeepSVDDClassifier):
    raise RuntimeError("The checkpoint is not a Deep SVDD classifier.")
if not torch.isfinite(model.svdd_threshold):
    raise RuntimeError("The checkpoint has no calibrated OOD threshold.")

model = model.to(device).eval()


@torch.inference_mode()
def collect_scores(directory: str) -> list[float]:
    scores = []
    for image_path in iter_dataset_images(Path(directory)):
        image = load_image_tensor(
            image_path,
            input_size=metadata["input_size"],
            colored=metadata["colored"],
        ).unsqueeze(0).to(device)
        output = model(image)
        scores.append(float(model.compute_svdd_score(output.svdd_embedding)[0]))
    if not scores:
        raise RuntimeError(f"No benchmark images found under {directory}.")
    return scores


id_scores = collect_scores("./ood-benchmark/id")
ood_scores = collect_scores("./ood-benchmark/ood")
threshold = float(model.svdd_threshold)

id_acceptance_rate = sum(score <= threshold for score in id_scores) / len(id_scores)
ood_rejection_rate = sum(score > threshold for score in ood_scores) / len(ood_scores)
balanced_ood_accuracy = (id_acceptance_rate + ood_rejection_rate) / 2
auroc = roc_auc_score(
    [0] * len(id_scores) + [1] * len(ood_scores),
    id_scores + ood_scores,
)

print(f"threshold:            {threshold:.6f}")
print(f"ID acceptance rate:   {id_acceptance_rate:.2%}")
print(f"OOD rejection rate:   {ood_rejection_rate:.2%}")
print(f"balanced OOD accuracy:{balanced_ood_accuracy:.2%}")
print(f"OOD AUROC:             {auroc:.4f}")
```

Interpret the metrics as follows:

- **ID acceptance rate**: fraction of held-out valid images with `ood_score <= ood_threshold`.
- **OOD rejection rate**: fraction of known OOD images with `ood_score > ood_threshold`.
- **Balanced OOD accuracy**: average of ID acceptance and OOD rejection rates, so unequal set sizes do not dominate the result.
- **OOD AUROC**: threshold-independent ranking quality, treating higher scores as more OOD-like. A useful AUROC does not replace reporting performance at the deployed threshold.

Also report the number and source of ID and OOD images, score distributions, and per-source OOD rejection rates. A single pooled OOD rate can hide failures on a specific category such as another radiology modality. The validation quantile targets acceptance on validation data only; a `0.95` quantile does not guarantee exactly 95% acceptance on a shifted test set.

Parameter counts for the available standard classifiers, using the current implementations with a 1000-class classifier head:

| Model | Parameters | Special Properties |
| --- | ---: | --- |
| `efficientnet_b0` | 5,288,548 | <ul><li>Smallest standard backbone in this repo</li><li>Compound-scaled EfficientNet family</li><li>Good baseline for efficiency-focused runs</li></ul> |
| `mobilenet_v3_large` | 5,483,032 | <ul><li>Mobile-oriented architecture</li><li>Uses inverted residual blocks</li><li>Good low-parameter benchmark</li></ul> |
| `drax_mobilenet_v3_large` | 6,058,232 | <ul><li>MobileNetV3 Large backbone with a bottlenecked late-stage `DraxBlock` refiner</li><li>Preserves the pretrained MobileNet feature extractor and classifier path</li><li>Adds moderate capacity with much lower overhead than full-width Drax insertion</li></ul> |
| `densenet121` | 7,978,856 | <ul><li>Feature reuse through dense connections</li><li>Lower parameter count than ResNet-18</li><li>Strong classical CNN baseline</li></ul> |
| `resnet18` | 11,689,512 | <ul><li>Smallest ResNet variant available here</li><li>Clean apples-to-apples baseline for `draxnet`</li><li>Standard residual basic blocks</li></ul> |
| `draxnet` | 16,994,856 | <ul><li>Local `ResNet-18`-style implementation</li><li>Current default uses `DraxResidualBlock` in `layer4`</li><li>Designed for custom block experimentation</li></ul> |
| `resnet50` | 25,557,032 | <ul><li>Deeper ResNet with bottleneck blocks</li><li>Common strong baseline</li><li>Larger than `draxnet` and `resnet18`</li></ul> |
| `convnext_tiny` | 28,589,128 | <ul><li>Smallest ConvNeXt variant available here</li><li>Modern conv backbone</li><li>Larger than `resnet50` in parameter count</li></ul> |
| `convnext_small` | 50,223,688 | <ul><li>Mid-sized ConvNeXt variant</li><li>Substantially larger than `convnext_tiny`</li><li>Useful for capacity scaling comparisons</li></ul> |
| `convnext_base` | 88,591,464 | <ul><li>Large ConvNeXt backbone</li><li>High-capacity benchmark</li><li>Much heavier training/inference footprint</li></ul> |
| `convnext_large` | 197,767,336 | <ul><li>Largest backbone currently exposed</li><li>Very high parameter count</li><li>Best suited for heavyweight benchmarking</li></ul> |

### DraxNet Notes

`draxnet` is currently a local `ResNet-18`-style backbone with a configurable per-stage block layout.

`drax_mobilenet_v3_large` keeps the torchvision `mobilenet_v3_large` backbone and inserts a bottlenecked `DraxBlock` refiner after the final feature stage. This is intended for late-stage feature mixing experiments while preserving MobileNet's efficient scaffold and pretrained initialization path.

Current default:

```text
basic,basic,basic,drax
```

That means the first three stages use plain residual blocks and `layer4` uses `DraxResidualBlock`.

`Drax` stands for `Dynamic Residual Attention eXchange`.

Detailed architecture, naming, and benchmarking notes are documented in:

- [Experimental Blocks](./experimental_blocks.md)

### One-Shot Classification

One-shot classification trains a Siamese similarity model. Instead of learning a fixed softmax classifier over known classes, the model learns whether two images are from the same class. Training and validation data are still organized by label, but the loader builds positive pairs from two images in the same label directory and negative pairs from images in different label directories.

Recommended dataset layout:

```text
<dataset-root>/
├── train/
│   ├── alphabet_a/
│   │   ├── sample-1.png
│   │   └── sample-2.png
│   └── alphabet_b/
│       ├── sample-1.png
│       └── sample-2.png
├── val/
│   ├── alphabet_a/
│   └── alphabet_b/
└── test/
    ├── alphabet_a/
    └── alphabet_b/
```

`train/` and `val/` are required for training. `test/` is optional for training, but it is the usual reference split for `benchmark`, `infer-image`, and `cam`.

Every label used by a one-shot split needs at least two images. Labels with fewer than two images cannot produce positive pairs and are ignored by the one-shot pair loader.

Example:

```bash
python -m mlx \
    --mode image_classification \
    --model siamese-le-net \
    --action train \
    --dataset ~/datasets/omniglot \
    --output ./artifacts/siamese \
    --epochs 50 \
    --batch-size 8 \
    --device cuda:0
```

With the default `--use-best` behavior, the best validation-loss checkpoint is written to `./artifacts/siamese/siamese-le-net.pth`.

Important arguments:

- `--model`: one-shot model name.
- `--dataset`: dataset root containing `train/` and `val/`.
- `--output`: output artifact directory. Training writes `{model}.pth`, `{model}.last.pth`, and `training.csv` inside it.
- `--model-path`: optional `{model}.last.pth` checkpoint used to resume at the next epoch.
- `--embedding-size`: Siamese embedding width.
- `--epochs`, `--batch-size`, `--device`, `--lr`: training controls.
- `--seed` / `--random-seed`: optional integer seed applied globally across Python, NumPy, and PyTorch for reproducible runs.
- `--height`, `--width`: input dimensions used to build `input_size`.
- The terminal UI prints one completed epoch per line above the training progress bars, including training loss, validation loss, accuracy, precision, recall, and F1.

Supported one-shot models:

| Model | Preserved backbone property |
| --- | --- |
| `siamese-le-net` | Compact four-stage LeNet-style convolutional embedding |
| `siamese-resnet18`, `siamese-resnet50` | Residual connections |
| `siamese-densenet121` | Dense feature concatenation |
| `siamese-mobilenet_v3_large` | Inverted residuals, squeeze-excitation, and hard-swish |
| `siamese-efficientnet_b0` | EfficientNet MBConv architecture |
| `siamese-convnext_tiny`, `siamese-convnext_small`, `siamese-convnext_base`, `siamese-convnext_large` | ConvNeXt blocks at four capacity levels |
| `siamese-draxnet` | Configurable Drax attention and fusion stages |
| `siamese-drax_mobilenet_v3_large` | MobileNet V3 features refined by a Drax adapter |

The backbone is shared between both inputs. Each model projects to `--embedding-size`, applies a sigmoid embedding activation, and learns a same-class probability from the absolute embedding difference. `--pretrained` initializes supported backbones before the complete Siamese model is fine-tuned. DraxNet retains its existing restriction that pretrained weights require all-basic stages.

For `siamese-draxnet` and `siamese-drax_mobilenet_v3_large`, use `--drax-fusion-mode average` for the fixed equal-weight residual average or `--drax-fusion-mode sknet` for learned, input-dependent channel weights. The default remains `average` for compatibility.

### One-Shot Dataset Builder

Use `build-dataset` when your source images are grouped by label but do not yet have `train/`, `val/`, and `test/` folders. This is the common setup for one-shot datasets such as Omniglot-style character folders.

Expected source layout:

```text
./data/omniglot-raw/
├── character_001/
│   ├── image-01.png
│   ├── image-02.png
│   └── ...
├── character_002/
│   ├── image-01.png
│   ├── image-02.png
│   └── ...
└── character_003/
    ├── image-01.png
    ├── image-02.png
    └── ...
```

Build a one-shot-ready split with fixed image counts per label:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ./data/omniglot-raw \
    --output ./data/omniglot-one-shot \
    --train-count 12 \
    --val-count 4 \
    --test-count 4 \
    --overwrite \
    --seed 42
```

This creates:

```text
./data/omniglot-one-shot/
├── train/
│   ├── character_001/
│   ├── character_002/
│   └── character_003/
├── val/
│   ├── character_001/
│   ├── character_002/
│   └── character_003/
└── test/
    ├── character_001/
    ├── character_002/
    └── character_003/
```

Then train the one-shot model from the generated split:

```bash
python -m mlx \
    --mode image_classification \
    --model siamese-le-net \
    --action train \
    --dataset ./data/omniglot-one-shot \
    --output ./artifacts/siamese \
    --epochs 50 \
    --batch-size 8 \
    --device cuda:0 \
    --seed 42
```

For proportional splits, use ratio mode. Ratios are normalized and applied within each label independently:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ./data/omniglot-raw \
    --split-mode ratios \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --output ./data/omniglot-one-shot \
    --overwrite \
    --seed 42
```

The builder copies image files; it does not generate pairs on disk. Pair generation happens at training and benchmarking time through `OneShotPairDataset`.

## Available Actions

- `train`: train the selected model and write artifacts to `--output`, including `{model}.pth` and `training.csv`.
- `test`: run a random-tensor smoke test for the configured model.
- `benchmark`: evaluate a trained checkpoint against a dataset directory. Standard models classify labels directly; one-shot models evaluate pair similarity.
- `infer-image`: run inference for one input image. Standard models output class probabilities; one-shot models rank a reference dataset by learned same-class probability.
- `cam`: render class activation maps for test images from a trained checkpoint. Supported methods are `gradcam`, `ablationcam`, and `scorecam`.
- `build-dataset`: interactively create train/val/test splits from a label-organized source dataset.

## Build Dataset

`build-dataset` is used to convert a label-organized source dataset into a split dataset with `train/`, `val/`, and `test/` directories.

Required flag:

- `--dataset-path`: source dataset root. This must point to the unsplit label-organized dataset you want to process.

Behavior:

- The command scans each label directory under `--dataset-path`.
- It prints a label summary showing how many images were found per label.
- In count mode, it prompts for three values: images per label for `TRAIN`, `VAL`, and `TEST`.
- In ratio mode, it prompts for three values: ratios for `TRAIN`, `VAL`, and `TEST`, then splits each label independently using those ratios.
- It finally prompts for the output path where the split dataset should be created.
- If the output directory already exists, MLX asks for confirmation before overwriting it.
- If a label has fewer images than the requested total, MLX prints a warning before continuing.

Expected input layout:

```text
<source-dataset>/
├── cats/
│   ├── cat-1.jpg
│   ├── cat-2.jpg
│   └── ...
├── dogs/
│   ├── dog-1.jpg
│   ├── dog-2.jpg
│   └── ...
└── horses/
    ├── horse-1.jpg
    ├── horse-2.jpg
    └── ...
```

Example command:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw
```

To build the split dataset in one command without prompts, pass the split counts and output path:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw \
    --output ~/datasets/animals \
    --train-count 100 \
    --val-count 20 \
    --test-count 20 \
    --overwrite \
    --seed 42
```

To build a split from ratios, use `--split-mode ratios`. MLX applies the ratios within each label independently:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw \
    --split-mode ratios \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --output ~/datasets/animals \
    --overwrite \
    --seed 42
```

You can also pass only some ratio values on the command line and let MLX prompt for the rest:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ~/datasets/animals-raw \
    --split-mode ratios \
    --train-ratio 0.7 \
    --output ~/datasets/animals \
    --seed 42
```

Non-interactive build-dataset flags:

- `--train-count`: images per label copied into `train/`
- `--val-count`: images per label copied into `val/`
- `--test-count`: images per label copied into `test/`
- `--train-ratio`: train split ratio applied within each label
- `--val-ratio`: validation split ratio applied within each label
- `--test-ratio`: test split ratio applied within each label
- `--split-mode`: `counts` or `ratios`; ratio mode splits each label independently using the provided ratios
- `--output`: destination directory for the split dataset
- `--overwrite`: replace an existing output directory without prompting
- `--seed` / `--random-seed`: global seed value; dataset splitting uses it for deterministic shuffling

Behavior notes:

- If any of `--train-count`, `--val-count`, `--test-count`, or `--output` are omitted, MLX prompts only for the missing values.
- If `--split-mode ratios` is used and any of `--train-ratio`, `--val-ratio`, `--test-ratio`, or `--output` are omitted, MLX prompts only for the missing values.
- Count mode and ratio mode are mutually exclusive.
- If `--output` already exists in non-interactive mode, MLX raises an error unless `--overwrite` is set.

Example interactive flow:

```text
How many images per label for TRAIN? 20
How many images per label for VAL? 5
How many images per label for TEST? 5
Enter output path for split dataset ~/datasets/animals-split
```

Example interactive flow in ratio mode:

```text
Train ratio? 0.7
Validation ratio? 0.15
Test ratio? 0.15
Enter output path for split dataset ~/datasets/animals-split
```

This creates a dataset like:

```text
~/datasets/animals-split/
├── train/
│   ├── cats/
│   ├── dogs/
│   └── horses/
├── val/
│   ├── cats/
│   ├── dogs/
│   └── horses/
└── test/
    ├── cats/
    ├── dogs/
    └── horses/
```

Another example:

```bash
python -m mlx \
    --mode image_classification \
    --action build-dataset \
    --dataset ./data/omniglot-raw
```

Then provide prompts such as:

```text
How many images per label for TRAIN? 12
How many images per label for VAL? 4
How many images per label for TEST? 4
Enter output path for split dataset ./data/omniglot-split
```

Use the generated output directory as `--dataset` for `train`, and use its `test/` directory for `benchmark`, `infer-image`, or `cam` when needed.

## Benchmarking

### Standard Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action benchmark \
    --model-path ~/datasets/animals/checkpoints/best_epoch_12.pt \
    --dataset ~/datasets/animals/test \
    --output ./benchmark-results \
    --device cpu
```

For standard classifiers, `benchmark` loads class labels from the checkpoint metadata and evaluates accuracy, average precision, average recall, F1, and ROC AUC against the labelled images in `--dataset`. It also reports per-class AUC, sensitivity, and specificity. If `--dataset` points to the dataset root and a `test/` directory exists, MLX evaluates that `test/` directory automatically.

If `--output` is set for `benchmark`, MLX writes benchmark artifacts to that directory:

- `metrics.csv`: aggregate metrics including accuracy, average precision, average recall, F1, available ROC AUC values, plus per-class AUC, sensitivity, and specificity
- `confusion_matrix.csv`: raw confusion-matrix counts with multi-class support
- `confusion_matrix.png`: rendered confusion-matrix heatmap
- `roc_curve.png`: ROC curve plot. For multi-class classification, MLX renders one-vs-rest curves per class.

### One-Shot Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action benchmark \
    --dataset ~/datasets/omniglot/test \
    --model-path ~/datasets/omniglot/checkpoints/best_epoch_10.pt \
    --output ./benchmark-results \
    --num-pairs 100 \
    --seed 42 \
    --device cpu
```

For one-shot models, `benchmark` requires `--model-path` and evaluates deterministic positive and negative similarity pairs built from the provided dataset directory. `--num-pairs` controls how many pairs are sampled per label, split as evenly as possible between same-label and different-label pairs. Use `--seed` for reproducible pair sampling.

When `--output` is set, MLX writes the shared benchmark artifacts with binary labels:

- `metrics.csv`: aggregate pair-verification metrics including accuracy, macro precision, macro recall, F1, ROC AUC, average precision, equal error rate, best-F1 threshold, Youden-threshold values, and an N-way classification metric summary
- `confusion_matrix.csv`: binary confusion-matrix counts for `different` and `same`
- `confusion_matrix.png`: rendered binary confusion matrix
- `roc_curve.png`: binary ROC curve

It also writes one-shot research artifacts:

- `pair_predictions.csv`: one row per evaluated pair with both image paths, both labels, target, prediction, and same-class probability
- `threshold_metrics.csv`: threshold sweep with accuracy, precision, recall, specificity, F1, FPR, FNR, TP, FP, TN, and FN
- `precision_recall_curve.png`: precision-recall curve with average precision
- `score_distribution.png`: score histograms for same-label and different-label pairs

For label-level one-shot classification results, MLX also writes `n_way_classification/` using a leave-one-out nearest-reference evaluation over the benchmark directory:

- `n_way_classification/metrics.csv`: standard-style classification metrics for predicted labels
- `n_way_classification/confusion_matrix.csv`: multi-class confusion-matrix counts
- `n_way_classification/confusion_matrix.png`: rendered multi-class confusion matrix
- `n_way_classification/roc_curve.png`: one-vs-rest ROC curves derived from per-label similarity scores
- `n_way_classification/predictions.csv`: one row per query image with true label, predicted label, best reference image, best same-class probability, and correctness

## Class Activation Maps

The `cam` action explains trained image-classification checkpoints by rendering activation heatmaps over test images. It uses the `pytorch-grad-cam` package, installed through the `grad-cam` dependency.

Supported methods:

- `gradcam`: gradient-weighted class activation maps.
- `ablationcam`: activation ablation maps. This is usually slower than Grad-CAM.
- `scorecam`: score-weighted activation maps. This is usually slower than Grad-CAM.

The operation is implemented in `mlx.modes.image_classification.cam` so the same components can be reused from notebooks without opening OpenCV windows.

### Standard Classification CAM

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action cam \
    --model resnet18 \
    --model-path ./artifacts/resnet18/resnet18.pth \
    --dataset ./data/animals \
    --output ./cam-results/resnet18 \
    --cam-method gradcam \
    --max-samples 20 \
    --device cuda:0
```

For standard classifiers, `cam` loads class labels from checkpoint metadata and reads images from `--dataset`. If `--dataset` points to a dataset root and a `test/` directory exists, MLX uses that `test/` directory automatically. By default, each image is explained for the model's predicted class. Pass `--target-index` to explain a specific class index instead.

### One-Shot CAM

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action cam \
    --model siamese-le-net \
    --model-path ./artifacts/siamese/siamese-le-net.pth \
    --dataset ./data/omniglot-one-shot \
    --output ./cam-results/siamese \
    --cam-method gradcam \
    --num-pairs 25 \
    --max-samples 10 \
    --device cuda:0
```

For one-shot models, `cam` builds deterministic test pairs using the same pair-sampling helper as benchmarking. Each pair produces two CAM outputs: one for the first image while the second image is fixed, and one for the second image while the first image is fixed. The target is the Siamese similarity output, so the heatmap shows which regions affected the pair-similarity score.

`--num-pairs` controls how many candidate pairs are sampled per label before `--max-samples` is applied. Use `--seed` for reproducible pair sampling.

### CAM Options

- `--cam-method`: one of `gradcam`, `ablationcam`, or `scorecam`. Defaults to `gradcam`.
- `--model-path`: trained MLX checkpoint to explain.
- `--dataset`: dataset root or test directory. If a `test/` child exists, it is used automatically.
- `--output`: optional directory for rendered CAM images.
- `--display` / `--no-display`: show rendered images in OpenCV windows. CLI usage defaults to display enabled; notebook usage should usually set `display=False`.
- `--save-images` / `--no-save-images`: write rendered overlays under `--output`. Enabled by default when `--output` is provided.
- `--max-samples`: maximum number of test samples to process. For one-shot CAM this limits sampled pairs, and each pair emits two images.
- `--target-layer`: optional dotted module path for the CAM target layer, such as `layer4.-1`, `features.-1`, or `embedding.3`.
- `--target-index`: optional class index for standard classifiers. For one-shot models this targets the Siamese output index, normally `0`.
- `--aug-smooth`: enable test-time augmentation smoothing from `pytorch-grad-cam`.
- `--eigen-smooth`: enable Eigen smoothing from `pytorch-grad-cam`.
- `--window-delay`: OpenCV `waitKey` delay in milliseconds between displayed images. The default `0` waits for a key press.

Default target-layer selection is conservative:

- ResNet-style and `draxnet` models use `layer4.-1`.
- Models with a `features` stack use `features.-1`.
- `siamese-le-net` uses `embedding.3`.
- If no known structure matches, MLX uses the last convolutional layer it can find.

### Notebook Usage

Use `generate_image_classification_cams` directly to get Python objects instead of only files or OpenCV windows:

```python
from mlx.modes.image_classification.cam import generate_image_classification_cams

results = generate_image_classification_cams({
    "model": "siamese-le-net",
    "model_path": "./artifacts/siamese/siamese-le-net.pth",
    "dataset_path": "./data/omniglot-one-shot",
    "output_path": "./cam-results/siamese",
    "device": "cuda:0",
    "cam_method": "gradcam",
    "num_pairs": 25,
    "max_samples": 10,
    "display": False,
})

first = results[0]
first.visualization      # RGB uint8 overlay image
first.grayscale_cam      # 2D CAM array
first.output_path        # saved file path when save_images is enabled
```

## Image Inference

### Standard Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action infer-image \
    --model-path ~/datasets/animals/checkpoints/best_epoch_12.pt \
    --input-img ~/datasets/query/cat.jpg \
    --device cpu
```

For ordinary standard classifiers, `infer-image` predicts the top classes for `--input-img`. `--dataset` is not required because labels are loaded from checkpoint metadata. For a Deep SVDD deployment checkpoint, inference first compares the squared-Euclidean embedding score with the calibrated threshold. Accepted images return the class prediction, confidence, OOD score, and threshold. Rejected images return `predicted_label=None`, `confidence=None`, and `rejection_reason="out_of_distribution"` rather than presenting the highest logit as a trusted prediction.

### One-Shot Classification

Example:

```bash
python -m mlx \
    --mode image_classification \
    --action infer-image \
    --model-path ./artifacts/siamese/siamese-le-net.pth \
    --dataset ~/datasets/omniglot/test \
    --input-img ~/datasets/query/sample.png \
    --model siamese-le-net \
    --device cpu
```

For one-shot models, `infer-image` requires `--model-path`. It loads the query image, computes its embedding, compares it against images under `--dataset`, and renders the top matches.
