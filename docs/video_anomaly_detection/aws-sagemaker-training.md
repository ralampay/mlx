# AWS SageMaker training for every video-anomaly model

The AWS `train-all` action starts one SageMaker job and trains every configuration returned by the
live clip-native 3D registry in sequence. The dataset is extracted once. Standard backbones run
once; `draxnet` and `drax_mobilenet_v3_large` run with both `average` and `sknet` fusion. Frame-2D
compatibility and Siamese models are excluded.

## Prerequisites and submission

- Install `python -m pip install ".[aws]"`.
- Configure an AWS profile and verify it with
  `aws sts get-caller-identity --profile mlx-training`.
- Keep the dataset and result buckets in the configured SageMaker region.
- Request SageMaker quota for the selected instance type.
- Provide a compatible ECR `aws.image_uri` and SageMaker `aws.execution_role_arn`, as shown in the
  example; alternatively allow MLX to build/provision them locally with Docker and additional IAM
  permissions.

Copy [`aws-training.example.yaml`](./aws-training.example.yaml), replace the result bucket, then
submit:

```bash
python -m mlx --mode video-anomaly-detection --platform aws --action train-all \
  --config ./aws-video-anomaly.yaml --profile mlx-training
```

`train-all` rejects `--model`: the model inventory is generated from the live registry and frozen
in S3 before submission. Explicit common training options on the CLI override YAML. The result
contains the batch ID and SageMaker job name.

## Avenue dataset contract

The example uses
`s3://mlx-video-anomaly-datasets/avenue-video-anomaly-detection.zip`. The ZIP must contain one MLX
dataset root, directly or below one wrapper directory:

```text
avenue/
├── train/normal/<sequence>/<ordered-frame>.jpg
├── val/normal/<sequence>/<ordered-frame>.jpg
└── test/{normal,anomaly}/...       # optional for training
```

The entrypoint uses the shared safe ZIP extractor and mode-owned dataset-root validation. A raw
Avenue download without `train/normal` and `val/normal` is rejected; prepare the frame-sequence
layout described in the [video-anomaly guide](./README.md#generic-dataset-layout).

## Recommended parameters

The example starts with `ml.g6e.2xlarge`, batch size 1, 224×224 frames, 16-frame clips, temporal
kernel 3, 50 epochs, learning rate 0.001, four loader workers, and pretrained inflation. The
single-GPU G6e size provides more accelerator memory than G4dn/G5 starting sizes, which matters for
the largest inflated ConvNeXt model. Availability and quota vary by region.
See AWS's [supported SageMaker instance values](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceSpec.html)
and [G6e specifications](https://aws.amazon.com/ec2/instance-types/accelerated-computing/).

If a model runs out of GPU memory, reduce `height` and `width` first; batch size is already one. If
G6e is unavailable, select another single-GPU instance with sufficient accelerator memory.
Multi-GPU instances do not accelerate this sequential implementation. `pretrained: true` can
download torchvision weights, so keep network isolation disabled or bake the weights into the
image.

Managed Spot is enabled by default. Every epoch publishes a rotating full-state checkpoint, so a
Spot interruption or later resume does not restart the active model from epoch zero.
See [Managed Spot Training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)
and [SageMaker checkpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html).

## Status and recovery

```bash
python -m mlx --mode video-anomaly-detection --platform aws --action status \
  --config ./aws-video-anomaly.yaml --job-name JOB_NAME

python -m mlx --mode video-anomaly-detection --platform aws --action status \
  --config ./aws-video-anomaly.yaml --job-name JOB_NAME --watch

python -m mlx --mode video-anomaly-detection --platform aws --action resume \
  --config ./aws-video-anomaly.yaml --job-name FAILED_OR_STOPPED_JOB
```

Status merges SageMaker state with `batch-status.json`, including current model, epoch progress,
and completed/failed/pending variants. The batch is fail-fast. Resume requires the same dataset,
output root, training values, infrastructure settings, tags, role, and image. It verifies and skips
completed models, then restores the active model from the newest valid rotating checkpoint.

## S3 results

```text
<aws.output_s3_uri>/mlx-vad/batches/<batch-id>/
├── batch-spec.json
├── batch-status.json
├── models/<variant-id>/
│   ├── <variant>-3d-svdd.pth
│   ├── <variant>-3d-svdd.last.pth
│   ├── training.csv
│   ├── training_history.png
│   ├── run_metadata.json
│   └── artifact-manifest.json
├── recovery/<variant-id>/{current.json,resume-a.pth,resume-b.pth,...}
```

Attempt-level SageMaker model archives are written separately under
`<aws.output_s3_uri>/mlx-vad/attempts/<batch-id>/<sagemaker-job-name>/` so they are not downloaded
as checkpoint input during resume.

SageMaker also packages the completed `models/` tree as the attempt's model artifact. Direct
`models/<variant-id>/` prefixes are hash-verified before a resumed attempt skips them.

## IAM

The local caller submits and observes jobs. The SageMaker execution role reads the dataset, pulls
the image, and writes results. Replace all placeholders in these scoped examples:

- [`aws-caller-policy.example.json`](./aws-caller-policy.example.json)
- [`aws-execution-role-policy.example.json`](./aws-execution-role-policy.example.json)
- [`aws-execution-role-trust-policy.example.json`](./aws-execution-role-trust-policy.example.json)

If MLX creates ECR and IAM resources, the caller additionally needs ECR image-push and IAM
role/policy creation permissions. Prefer pre-provisioning `aws.image_uri` and
`aws.execution_role_arn` in controlled environments. KMS and VPC settings require the associated
key and network-interface permissions. The existing
[object-detection SageMaker guide](../object_detection/aws-sagemaker-training.md) documents the
same shared ECR, IAM, KMS, and VPC lifecycle in detail.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Dataset layout rejected | Confirm the extracted root has `train/normal` and `val/normal`. |
| S3 region mismatch | Use buckets in `aws.region`, or select the buckets' region. |
| Instance unavailable | Check SageMaker quota and regional G6e availability. |
| Pretrained download fails | Disable network isolation or bake weights into the image. |
| Resume configuration differs | Use the exact original YAML and image/role settings. |
| Completed artifact is corrupt | Restore the missing S3 object; inconsistent state is never silently skipped. |
