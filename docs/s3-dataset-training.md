# Training from an S3 Dataset ZIP

MLX can train every train-capable mode from a dataset ZIP stored in Amazon S3. This applies to:

- object detection;
- image classification, including standard and one-shot models;
- one-class image recognition;
- semantic segmentation;
- video anomaly detection.

The option is a portable dataset source, not a new dataset format. MLX stages the archive and
then passes a resolved local directory to the same loader used by `--dataset`. Benchmarking,
inference, dataset building, tracking, and NLP actions do not accept `--dataset-s3-uri`.

## Installation and AWS credentials

Install Boto3 through the AWS extra when MLX was installed as a package:

```bash
python -m pip install ".[aws]"
```

Development installations made with `requirements.txt` already include Boto3. MLX uses Boto3's
standard credential chain. You can select a named shared-credentials profile explicitly:

```bash
--profile mlx-training
```

If `--profile` is omitted, environment credentials, workload roles, `AWS_PROFILE`, and the
default shared profile work as they normally do with Boto3. MLX does not put credentials or the
profile name in dataset provenance.

The selected identity needs permission to read the ZIP. `HeadObject` uses the `s3:GetObject`
permission in an IAM policy, so a minimal object statement is:

```json
{
  "Effect": "Allow",
  "Action": "s3:GetObject",
  "Resource": "arn:aws:s3:::my-datasets/path/dataset.zip"
}
```

An object encrypted with a customer-managed KMS key may also require `kms:Decrypt` and a key
policy that permits the selected identity.

## Local training

Supply the S3 object instead of `--dataset`, and always provide a persistent output location:

```bash
python -m mlx \
  --mode image_classification \
  --action train \
  --model resnet18 \
  --dataset-s3-uri s3://my-datasets/image-classification.zip \
  --output ./artifacts/resnet18-s3 \
  --profile mlx-training
```

The relevant options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset-s3-uri` | none | S3 URI of the dataset ZIP. The key must end in `.zip`. |
| `--dataset-cache-dir` | `~/.cache/mlx/datasets` | Persistent cache used for local S3 staging. |
| `--profile` | Boto3 default chain | Optional named AWS profile. |
| `--output` | mode-specific | Required for S3 training so provenance and model artifacts have a persistent destination. |

`--dataset` and `--dataset-s3-uri` are mutually exclusive when both are explicitly supplied.
S3 URIs with query strings or fragments are rejected. Upload a ZIP object rather than a TAR,
directory prefix, or individual file collection.

### Mode examples

Object detection:

```bash
python -m mlx --mode object_detection --action train \
  --provider ultralytics --model yolo26 \
  --dataset-s3-uri s3://my-datasets/detection.zip \
  --output ./runs/detection-s3 --profile mlx-training
```

Image classification:

```bash
python -m mlx --mode image_classification --action train \
  --model resnet18 \
  --dataset-s3-uri s3://my-datasets/classification.zip \
  --output ./artifacts/resnet18-s3 --profile mlx-training
```

One-class image recognition:

```bash
python -m mlx --mode image_recognition_oc --action train \
  --model deep-svdd --backbone resnet18 \
  --dataset-s3-uri s3://my-datasets/one-class-images.zip \
  --output ./artifacts/one-class-s3 --profile mlx-training
```

Segmentation:

```bash
python -m mlx --mode segmentation --action train \
  --model unet-resnet18 \
  --dataset-s3-uri s3://my-datasets/segmentation.zip \
  --output ./artifacts/segmentation-s3 --profile mlx-training
```

Video anomaly detection:

```bash
python -m mlx --mode video_anomaly_detection --action train \
  --model resnet18 --backbone-mode 3d --clip-length 16 \
  --dataset-s3-uri s3://my-datasets/avenue-prepared.zip \
  --output ./artifacts/avenue-s3 --profile mlx-training
```

## ZIP dataset contracts

The mode-specific structure may start at the archive root or below an optional wrapper path.
MLX searches for exactly one valid dataset root. Keep only one dataset in each ZIP; ambiguous
archives are rejected.

### Object detection

The archive must contain exactly one `data.yaml`:

```text
detection-dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Paths inside `data.yaml` should remain valid relative to its containing dataset root. Provider
aliases such as `coco8` are not S3 objects and should continue to use `--dataset coco8`.

### Image classification

The archive must contain exactly one root with `train/` and `val/`:

```text
classification-dataset/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/                 # optional for training
```

The ordinary mode rules still apply. In particular, one-shot label directories need enough
images to construct positive and negative pairs.

### One-class image recognition

The archive must contain normal-only training and validation images:

```text
one-class-image-dataset/
├── train/
│   └── normal/
├── val/
│   └── normal/
└── test/                 # optional for training
    ├── normal/
    └── anomaly/
```

Anomaly images under `train/` or `val/` remain an error after staging.

### Segmentation

The archive must contain the paired training and validation directories:

```text
segmentation-dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/                 # optional for training
    ├── images/
    └── masks/
```

Image/mask naming and class-index requirements are unchanged from local training.

### Video anomaly detection

The archive must contain normal-only training and validation frame sequences:

```text
video-anomaly-dataset/
├── train/
│   └── normal/
│       └── clip001/001.tif ...
├── val/
│   └── normal/
│       └── clip001/001.tif ...
└── test/                 # optional for training
    ├── normal/
    └── anomaly/
```

Staging does not weaken the mode's one-class safeguards. Anomaly data under `train/` is still
rejected, validation threshold calibration still uses normal samples only, and the current
training loader still expects extracted image sequences rather than MP4 files.

## Creating and uploading the ZIP

Build the ZIP from the directory above the dataset root so relative paths are preserved. Python's
standard-library ZIP command works across supported platforms:

```bash
cd /path/to/prepared-datasets
python -m zipfile -c avenue-prepared.zip avenue-prepared
aws s3 cp avenue-prepared.zip s3://my-datasets/video-anomaly/avenue-prepared.zip \
  --profile mlx-training
```

Confirm the uploaded object's size, ETag, and optional VersionId before starting a long run:

```bash
aws s3api head-object \
  --bucket my-datasets \
  --key video-anomaly/avenue-prepared.zip \
  --profile mlx-training
```

Avoid adding unrelated datasets, prior ZIP files, symlinks, sockets, or named pipes. For object
detection, also avoid backup copies of `data.yaml`, because the resolver deliberately requires
exactly one.

## Cache lifecycle

Before downloading, MLX calls `HeadObject` and forms a cache identity from the bucket, key,
VersionId or ETag provenance, object size, and modification metadata. It then:

1. downloads to a temporary sibling directory while reporting byte progress;
2. verifies the downloaded byte count and computes SHA-256;
3. safely extracts the ZIP;
4. resolves the mode-specific dataset root;
5. writes a completion manifest;
6. atomically publishes the cache entry.

The downloaded ZIP is removed after successful extraction; the extracted dataset remains cached.
An unchanged object reuses its completed cache entry. A changed VersionId, ETag, size, or
modification identity produces a new entry. An interrupted or incomplete entry is not reused.

Override the cache for a larger or faster volume when needed:

```bash
--dataset-cache-dir /mnt/mlx-dataset-cache
```

MLX checks declared uncompressed size against available disk space. Cache eviction is currently
manual. Use `dataset_source.json` to identify the exact `cache_identity` before removing a single
entry; do not remove the cache while a training process is using it.

## Safe extraction

MLX validates every ZIP member before writing files. It rejects:

- absolute paths, parent traversal, Windows drive paths, and malformed empty path segments;
- symbolic links and non-regular special files;
- duplicate normalized paths, including case-only collisions;
- file/directory path conflicts;
- archives that exceed available space or a SageMaker extraction limit;
- invalid or corrupt ZIP files.

Files are streamed from the ZIP rather than loading the entire archive into memory.

## Dataset provenance

Local S3 training writes `dataset_source.json` in the training artifact directory. For a legacy
segmentation checkpoint-file output, it is written in the associated research-artifact directory.
A representative document is:

```json
{
  "bucket": "my-datasets",
  "cache_identity": "8ca9...",
  "content_length": 123456789,
  "dataset_root": "dataset/classification-dataset",
  "etag": "0123456789abcdef",
  "key": "classification.zip",
  "last_modified": "2026-08-29T10:00:00+00:00",
  "sha256": "abcdef...",
  "uri": "s3://my-datasets/classification.zip",
  "version_id": null
}
```

The document is written before training and refreshed after successful completion. It contains
no credentials, access tokens, or AWS profile name. Preserve it with the checkpoint when moving
artifacts between systems.

For reproducible resume behavior, prefer an immutable key or enable S3 versioning. Local staging
inspects the current object identity each time; replacing an object at the same URI intentionally
creates a new cache entry.

## SageMaker behavior

Object detection and image classification also support `--platform aws`. In this path SageMaker
uses its managed training input channel to fetch the configured ZIP; the local staging cache is
not used. The container uses the same safe extractor and applicable root resolver.

The YAML value remains the default:

```yaml
aws:
  dataset_s3_uri: s3://my-datasets/classification.zip
```

For a new submission, override it explicitly:

```bash
python -m mlx --mode image_classification --platform aws --action train \
  --config ./aws-classification.yaml \
  --dataset-s3-uri s3://portable-datasets/classification.zip
```

An AWS resume continues to validate the dataset URI against the original run specification. A
different override is rejected. `--dataset-cache-dir` has no effect on SageMaker managed input.
One-class image recognition, segmentation, and video anomaly detection currently support S3 ZIP
staging for local training, not SageMaker lifecycle execution.

## Troubleshooting

| Error | What to check |
| --- | --- |
| URI must point to `.zip` | Upload a ZIP object and pass its complete `s3://bucket/key.zip` URI. |
| Unable to inspect/download S3 dataset | Verify the profile, Region/endpoint environment, `s3:GetObject`, bucket policy, KMS policy, and object key. |
| Both local and S3 datasets supplied | Remove either `--dataset` or `--dataset-s3-uri`. |
| S3 training requires `--output` | Choose a persistent artifact directory or supported mode-specific output path. |
| No valid or multiple dataset roots | Inspect the extracted layout against the contracts above and keep one dataset per ZIP. |
| Unsafe path, link, special file, or duplicate | Rebuild the ZIP from regular files using relative paths. |
| No complete video anomaly windows | Ensure each normal source has at least `(clip_length - 1) * frame_stride + 1` frames. |
| Insufficient extraction space | Free space or point `--dataset-cache-dir` at a larger volume. |

Once staged, subsequent dataset/model validation errors come from the existing mode loader and
have the same meaning as local `--dataset` training.
