# AWS SageMaker Image-Classification Training

MLX acts as a local control-plane client: it validates S3, publishes an immutable ECR image when
needed, prepares a scoped execution role, and submits an asynchronous SageMaker training job.
Standard classifiers, joint Deep-SVDD classifiers, and one-shot Siamese models use the same local
training command inside the container.

## Prerequisites and data

Install the AWS integration and configure a Boto3-compatible profile:

```bash
python -m pip install ".[aws]"
aws configure sso --profile mlx-training
aws sso login --profile mlx-training
```

Upload one ZIP whose root, or single wrapper directory, has the normal classification layout:

```text
dataset/
├── train/<label>/*.{jpg,png,...}
├── val/<label>/*.{jpg,png,...}
└── test/<label>/*.{jpg,png,...}  # optional for training
```

The dataset and checkpoint buckets must already exist in the job Region. MLX does not create,
delete, empty, or configure lifecycle rules on them. Copy
[`aws-training.example.yaml`](./aws-training.example.yaml), then submit:

```bash
python -m mlx --mode image-classification --platform aws --action train \
  --config ./aws-classification.yaml --profile mlx-training
```

An explicit `--profile`, `--instance-type`, or training option overrides YAML only when present on
the command line. Credentials are never copied into the image, job hyperparameters, or manifests.
For a new submission, `--dataset-s3-uri s3://bucket/portable.zip` likewise overrides
`aws.dataset_s3_uri`; resume continues to require the dataset URI stored in the original run
specification.
The shared [S3 dataset training guide](../s3-dataset-training.md) documents the classification ZIP
contract, safe extractor, and differences between local caching and SageMaker managed input.

## Lifecycle and artifacts

```bash
python -m mlx --mode image-classification --platform aws --action status \
  --config ./aws-classification.yaml --job-name JOB_NAME

python -m mlx --mode image-classification --platform aws --action status \
  --config ./aws-classification.yaml --job-name JOB_NAME --watch

python -m mlx --mode image-classification --platform aws --action stop \
  --config ./aws-classification.yaml --job-name JOB_NAME

python -m mlx --mode image-classification --platform aws --action resume \
  --config ./aws-classification.yaml --job-name JOB_NAME
```

Managed Spot is enabled by default. Each completed epoch publishes a validated full-state
checkpoint into alternating recovery slots. SageMaker restores that directory after an
interruption; a corrupt newest slot falls back to the prior completed epoch. Manual resume keeps
the original dataset, model settings, serialized payload, execution role, and image. Only
capacity/runtime settings and a higher total `training.epochs` target may change.

SageMaker's `model.tar.gz` contains `{model}.pth`, `{model}.last.pth`, `training.csv`, and
`training-summary.json`. Checkpoint synchronization data lives under:

```text
s3://CHECKPOINT_LOCATION/mlx-ic/runs/<run-id>/
├── run-spec.json
├── recovery/{current.json,resume-a.pth,resume-b.pth,best-a.pth,best-b.pth,best.pth,...}
└── attempts/<job-name>/
```

`training.pretrained: true` can download torchvision weights. Allow the job outbound access or
provide `aws.image_uri` for an image that already contains the required weights. With network
isolation enabled, all code, dependencies, and weights must already be present in that image.

## IAM, KMS, and VPC

There are two identities: the local caller submits and monitors jobs; the SageMaker execution
role reads the dataset/ECR image and writes checkpoints, artifacts, logs, and metrics. Adapt the
included [caller policy](./aws-caller-policy.example.json), [execution policy](./aws-execution-role-policy.example.json),
and [trust policy](./aws-execution-role-trust-policy.example.json). Replace every placeholder and
keep `iam:PassRole` restricted to SageMaker.

Set both `aws.image_uri` and `aws.execution_role_arn` to use pre-provisioned infrastructure and
remove local ECR/IAM provisioning permissions. Optional `aws.kms_key_arn` adds KMS data access;
the key policy must also allow the execution role. A VPC configuration must include both subnet
and security-group IDs, with connectivity to S3, ECR, CloudWatch, and any pretrained-weight source.

All recoverable configuration, archive, dependency, AWS SDK, and checkpoint failures are reported
as actionable MLX/SageMaker job errors. Inspect `--action status`, the returned console URL, and
CloudWatch logs when a remote job fails.
