# AWS SageMaker Object-Detection Training

MLX runs on the local machine as a control-plane client. It validates S3, builds and pushes a
training image when necessary, submits an asynchronous SageMaker training job, and later queries
or stops that job. Model training runs inside SageMaker; the local MLX process does not need to
remain open after `train` returns.

```text
local MLX CLI + selected AWS profile
    -> S3 validation and run manifests
    -> ECR image build/push
    -> SageMaker CreateTrainingJob
        -> dataset ZIP from user-owned S3
        -> Managed Spot compute
        -> checkpoints and artifacts in user-owned S3
    -> local status / stop / resume commands
```

## Prerequisites

- Python with `python -m pip install ".[aws,object-detection]"`
- Docker running locally, unless `aws.image_uri` points to a pre-built ECR image
- AWS CLI v2 for configuring and testing local credentials
- an AWS identity with the caller permissions below
- a dataset ZIP already uploaded to S3
- a checkpoint bucket or prefix already created in the same Region as the dataset and job
- an available SageMaker quota for the selected instance type

MLX never creates, empties, deletes, or configures lifecycle rules on the dataset or checkpoint
buckets. It may create one ECR repository and one narrowly named SageMaker execution role unless
their ARNs are supplied in YAML.

## Configure a local AWS profile

AWS recommends temporary credentials through IAM Identity Center or role assumption for local
development. Configure an IAM Identity Center profile and log in:

```bash
aws configure sso --profile mlx-training
aws sso login --profile mlx-training
aws sts get-caller-identity --profile mlx-training
```

For an IAM user with access keys, configure a named profile instead:

```bash
aws configure --profile mlx-training
aws sts get-caller-identity --profile mlx-training
```

That creates entries equivalent to these. Do not commit either file or put access keys in MLX
YAML:

```ini
# ~/.aws/credentials
[mlx-training]
aws_access_key_id = REPLACE_ME
aws_secret_access_key = REPLACE_ME

# ~/.aws/config
[profile mlx-training]
region = ap-southeast-1
output = json
```

Select the profile in one of three ways, in this precedence order:

1. pass `--profile mlx-training` to the MLX command;
2. set `aws.profile: mlx-training` in the YAML file;
3. omit both to use Boto3's normal credential chain, including `AWS_PROFILE` or `default`.

An explicit CLI profile overrides the YAML value:

```bash
python -m mlx --mode object-detection --platform aws --action train \
    --config ./aws-training.yaml --profile mlx-training
```

The same profile option is accepted by `status`, `stop`, and `resume`. MLX passes the selected
name to `boto3.Session`; credentials remain in the standard AWS files and are never copied into a
training job, Docker image, run manifest, or checkpoint.

AWS profile file syntax and supported temporary-credential methods are documented in the
[AWS CLI shared configuration guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

## IAM identities and policies

There are two separate identities:

- **Local caller identity:** the IAM user or assumed role selected by the profile. It invokes AWS
  APIs, pushes the image, submits jobs, and monitors their lifecycle.
- **SageMaker execution role:** assumed by `sagemaker.amazonaws.com` after submission. It reads the
  dataset and image and writes checkpoints, output, logs, and metrics.

Do not attach the caller policy to the execution role or use the execution role's ARN as local
access keys. `iam:PassRole` should be limited to the approved execution role and to SageMaker, as
shown below. AWS explains this separation in its
[SageMaker execution-role guide](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)
and [PassRole guidance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_passrole.html).

### Local caller policy: automatic infrastructure mode

Use [aws-caller-policy.example.json](./aws-caller-policy.example.json) as the identity policy for
the profile's IAM user or role. Replace every `<PLACEHOLDER>` before creating the policy:

| Placeholder | Value |
| --- | --- |
| `<ACCOUNT_ID>` | 12-digit AWS account ID |
| `<REGION>` | job Region, for example `ap-southeast-1` |
| `<DATASET_BUCKET>` | dataset bucket name |
| `<DATASET_OBJECT_KEY>.zip` | complete key below the bucket, without a leading slash |
| `<CHECKPOINT_BUCKET>` | shared checkpoint bucket name |
| `<CHECKPOINT_PREFIX>` | configured prefix without leading/trailing slashes |
| `<RESOURCE_PREFIX>` | `aws.resource_prefix`, default `mlx-od` |
| `<ECR_REPOSITORY>` | `aws.ecr_repository`, default `<RESOURCE_PREFIX>-training` |
| `<EXECUTION_ROLE_NAME_OR_PREFIX_WILDCARD>` | exact configured role name, or `<RESOURCE_PREFIX>-sagemaker-*` |

If `checkpoint_s3_uri` is the bucket root, remove `<CHECKPOINT_PREFIX>/` from object ARNs and use
`*` as the execution role's checkpoint `s3:prefix` condition. The caller policy permits no bucket
creation or deletion, no ECR image deletion, no role deletion, and no SageMaker endpoint
deployment. ECR upload actions match AWS's
[private image push policy](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html).

Because automatic mode grants `iam:PutRolePolicy` and `iam:PassRole`, keep the role resource
narrow. An administrator-managed execution role is preferable in environments with strict
separation of duties.

After replacing the placeholders, an administrator can create and attach the customer-managed
policy. The administrator profile used here is separate from the restricted runtime profile:

```bash
aws iam create-policy \
    --policy-name MLXSageMakerTrainingCaller \
    --policy-document file://docs/object_detection/aws-caller-policy.example.json \
    --profile aws-administrator

aws iam attach-user-policy \
    --user-name mlx-training-user \
    --policy-arn arn:aws:iam::111122223333:policy/MLXSageMakerTrainingCaller \
    --profile aws-administrator
```

For role- or group-based access, use `attach-role-policy` or `attach-group-policy` instead. The
profile selected by MLX may represent an IAM user, an IAM Identity Center permission set, or an
assumed role; it does not have to contain long-lived IAM user keys.

### Reduced local caller policy: pre-provisioned mode

An administrator can create the ECR image and execution role first, then set both values:

```yaml
aws:
  image_uri: 111122223333.dkr.ecr.ap-southeast-1.amazonaws.com/mlx-od-training@sha256:...
  execution_role_arn: arn:aws:iam::111122223333:role/mlx-od-sagemaker-execution
```

With both configured, remove `CreateAndInspectTrainingRepository`, `PushTrainingImages`,
`AuthorizeEcrLogin`, and `ProvisionScopedSageMakerRole` from the example caller policy. Retain
`PassScopedRoleOnlyToSageMaker`, S3 validation/run-manifest permissions, SageMaker lifecycle
permissions, `sts:GetCallerIdentity`, and CloudWatch metric reads.

If only `execution_role_arn` is supplied, MLX still creates/pushes the ECR image. If only
`image_uri` is supplied, MLX still creates/updates the execution role and the image must be in
Amazon ECR so its repository can be scoped.

### Pre-provisioned SageMaker execution role

Use [aws-execution-role-trust-policy.example.json](./aws-execution-role-trust-policy.example.json)
as the role trust policy and
[aws-execution-role-policy.example.json](./aws-execution-role-policy.example.json) as its base
permissions policy. Replace the same S3, ECR, account, and Region placeholders. The local caller
must still have `iam:PassRole` for this exact role. SageMaker requires the caller to pass the role
used by `CreateTrainingJob`.

When MLX creates the role, it generates equivalent resource-scoped permissions itself. If a
bucket policy, ECR repository policy, organization service-control policy, permissions boundary,
or KMS key policy contains an explicit deny, the IAM identity policy alone cannot override it.

### Optional KMS permissions

When `aws.kms_key_arn` is configured, or when a dataset/checkpoint bucket uses a customer-managed
SSE-KMS key, the execution role needs the applicable actions on that key:

```json
{
  "Effect": "Allow",
  "Action": [
    "kms:Decrypt",
    "kms:DescribeKey",
    "kms:Encrypt",
    "kms:GenerateDataKey"
  ],
  "Resource": "<KMS_KEY_ARN>"
}
```

The KMS key policy must allow the execution role. MLX adds these IAM actions to roles it creates
for `aws.kms_key_arn`, but it does not inspect bucket default encryption or modify any KMS key
policy. Add separately scoped statements when input and checkpoint buckets use other keys.

### Optional VPC permissions

When `aws.vpc` is configured, the execution role also needs the following. MLX adds this statement
to roles it creates:

```json
{
  "Effect": "Allow",
  "Action": [
    "ec2:CreateNetworkInterface",
    "ec2:CreateNetworkInterfacePermission",
    "ec2:DeleteNetworkInterface",
    "ec2:DeleteNetworkInterfacePermission",
    "ec2:DescribeDhcpOptions",
    "ec2:DescribeNetworkInterfaces",
    "ec2:DescribeSecurityGroups",
    "ec2:DescribeSubnets",
    "ec2:DescribeVpcs"
  ],
  "Resource": "*"
}
```

Private subnets must provide the network paths needed by the job, such as S3 and ECR VPC
endpoints and CloudWatch connectivity, or suitable NAT access. See AWS's
[SageMaker training VPC guide](https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html).

## Dataset and checkpoint S3 preparation

The dataset object must be one ZIP containing exactly one `data.yaml` after extraction. Paths in
the ZIP may be nested below one top-level directory. The paths referenced by `data.yaml` must be
valid inside the extracted dataset.

```bash
aws s3 cp ./object-detection.zip \
    s3://my-dataset-bucket/datasets/object-detection.zip \
    --profile mlx-training

aws s3api head-object \
    --bucket my-dataset-bucket \
    --key datasets/object-detection.zip \
    --profile mlx-training

aws s3api head-bucket \
    --bucket my-training-bucket \
    --profile mlx-training
```

The checkpoint location can be a bucket root or prefix. Multiple runs safely share it because MLX
writes each logical run below:

```text
s3://CHECKPOINT_LOCATION/<resource-prefix>/runs/<run-id>/
├── run-spec.json
├── recovery/
│   ├── current.json
│   ├── resume-a.pt / resume-a.json / resume-a.state.pt
│   ├── resume-b.pt / resume-b.json / resume-b.state.pt
│   └── best.pt
└── attempts/<sagemaker-job-name>/
```

Dataset and checkpoint buckets must be in the configured job Region.

## Configuration reference

```yaml
version: 1

aws:
  region: ap-southeast-1
  profile: mlx-training
  dataset_s3_uri: s3://my-dataset-bucket/datasets/object-detection.zip
  checkpoint_s3_uri: s3://my-training-bucket/mlx-checkpoints
  instance_type: ml.g4dn.xlarge
  volume_size_gb: 100
  managed_spot: true
  max_runtime_seconds: 86400
  max_wait_seconds: 172800
  resource_prefix: mlx-od
  # Optional pre-provisioned infrastructure:
  # ecr_repository: mlx-od-training
  # execution_role_name: mlx-od-sagemaker-execution
  # execution_role_arn: arn:aws:iam::111122223333:role/mlx-od-sagemaker-execution
  # image_uri: 111122223333.dkr.ecr.ap-southeast-1.amazonaws.com/mlx-od-training@sha256:...
  # kms_key_arn: arn:aws:kms:ap-southeast-1:111122223333:key/...
  # network_isolation: false
  # vpc:
  #   subnet_ids: [subnet-0123456789abcdef0]
  #   security_group_ids: [sg-0123456789abcdef0]
  tags:
    project: computer-vision
    owner: ml-team

training:
  provider: ultralytics
  model: yolo26
  epochs: 100
  batch_size: 16
```

`managed_spot` defaults to `true`. If `max_wait_seconds` is omitted, MLX uses twice
`max_runtime_seconds`, capped at 30 days. `max_wait_seconds` must not be lower than runtime.
`--instance-type`, `--profile`, and explicitly supplied training CLI options override YAML; parser
defaults do not silently replace YAML values.

## Submit, monitor, stop, and continue

Submit returns immediately with a SageMaker job name, ARN, logical run ID, console URL, immutable
image digest, checkpoint URI, and output URI:

```bash
python -m mlx --mode object-detection --platform aws --action train \
    --config ./aws-training.yaml --profile mlx-training
```

Check once, continuously watch, or emit machine-readable JSON Lines:

```bash
python -m mlx --mode object-detection --platform aws --action status \
    --config ./aws-training.yaml --profile mlx-training --job-name JOB_NAME

python -m mlx --mode object-detection --platform aws --action status \
    --config ./aws-training.yaml --profile mlx-training --job-name JOB_NAME --watch

python -m mlx --mode object-detection --platform aws --action status \
    --config ./aws-training.yaml --profile mlx-training --job-name JOB_NAME \
    --watch --format json
```

Status includes the SageMaker primary/secondary state, last recoverable epoch, progress, ETA,
expected finish time, interruption count, training and billable seconds, approximate Spot savings,
failure reason, and artifact locations. ETA appears after recoverable epoch metrics exist.

Stop and resume later using the original job name:

```bash
python -m mlx --mode object-detection --platform aws --action stop \
    --config ./aws-training.yaml --profile mlx-training --job-name JOB_NAME

python -m mlx --mode object-detection --platform aws --action resume \
    --config ./aws-training.yaml --profile mlx-training --job-name JOB_NAME
```

Wait until the stopped job reaches `Stopped` before resuming. Resume creates a new SageMaker job
attempt for the same logical run, image, and recovery prefix. It can use a different instance type,
runtime/wait limit, or a higher total epoch target. Dataset URI, checkpoint base, provider, and
model cannot change.

Managed Spot interruption inside an active job is automatic: SageMaker restores
`/opt/ml/checkpoints`, and MLX selects the newest valid full-state snapshot. Recovery guarantees
completed epochs; an interrupted incomplete epoch may be repeated.

## Cost and operational guidance

- Keep Managed Spot enabled unless deterministic capacity is more important than price.
- Set realistic runtime and wait limits so abandoned jobs cannot wait indefinitely.
- Choose the smallest GPU instance that fits the model and batch size, then increase only when
  utilization or memory measurements justify it.
- Use the shared checkpoint prefix; do not create one bucket per run.
- Configure an S3 lifecycle rule administratively if old attempts should transition or expire.
  MLX intentionally does not modify bucket lifecycle policies.
- Use `stop`, not local process termination, to request an orderly manual stop. The returned job
  identity is the resume handle.
- Watch SageMaker and ECR service quotas and retain immutable image digests needed by resumable
  runs.

## Common failures

| Error | Check |
| --- | --- |
| profile not found or expired | `aws sts get-caller-identity --profile PROFILE`; renew SSO login |
| S3 validation denied | caller `GetObject`, `GetBucketLocation`, and checkpoint `ListBucket`; bucket policy denies |
| cannot pass role | exact caller `iam:PassRole` resource and `iam:PassedToService` condition |
| ECR push denied | repository ARN, ECR upload actions, local Docker daemon, and Region/account |
| SageMaker job fails before training | execution-role trust, S3/ECR permissions, quota, instance availability |
| no progress or ETA yet | wait for the first validated completed-epoch checkpoint and CloudWatch publication |
| resume finds no checkpoint | the prior attempt must have completed at least one epoch and synchronized recovery files |
| VPC job cannot start or download data | execution-role EC2 actions, subnet capacity, endpoints/NAT, routes, and security groups |
| KMS access denied | execution-role IAM policy and KMS key policy must both allow the operation |
