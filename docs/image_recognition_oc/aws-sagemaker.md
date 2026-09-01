# AWS SageMaker one-class image recognition

The AWS platform runs MLX's existing Deep SVDD trainer and benchmark command in SageMaker. It
supports one backbone, every compatible backbone in sequence, optional evaluation after training,
and standalone evaluation of an existing checkpoint.

## Configure and submit

Install the AWS dependencies and copy the example:

```bash
python -m pip install ".[aws]"
cp docs/image_recognition_oc/aws-training.example.yaml ./aws-svdd.yaml
```

Set `aws.dataset_s3_uri` to one MLX one-class ZIP. Set `aws.output_s3_uri` to the owned result
bucket or prefix and `aws.resource_prefix` to the dataset/experiment grouping. Both buckets must
be in `aws.region`.

Train the YAML's one backbone:

```bash
python -m mlx --mode image_recognition_oc --platform aws --action train \
  --config ./aws-svdd.yaml
```

For every Deep SVDD backbone, remove `training.backbone` and submit:

```bash
python -m mlx --mode image_recognition_oc --platform aws --action train-all \
  --config ./aws-svdd.yaml
```

The inventory comes from the live standard classifier registry and excludes Siamese models. The
two Drax backbones run with both `average` and `sknet` fusion. One SageMaker job extracts the ZIP
once and runs the frozen inventory sequentially.

## Benchmarking

Set `benchmark.enabled: true` to benchmark each deployment checkpoint after training. The test
split must contain both `test/normal` and `test/anomaly`; nested anomaly type directories are
indexed recursively. A failed benchmark preserves completed training and resume retries only the
evaluation.

For standalone evaluation, set `benchmark.model_s3_uri` to an exact `.pth` object and run:

```bash
python -m mlx --mode image_recognition_oc --platform aws --action benchmark \
  --config ./aws-svdd.yaml
```

The job uses the checkpoint's stored preprocessing, SVDD center, and calibrated threshold. It
never recalibrates on test images. Standalone benchmark jobs are restarted as new jobs rather
than resumed.

## Status, stop, and resume

```bash
python -m mlx --mode image_recognition_oc --platform aws --action status \
  --config ./aws-svdd.yaml --job-name JOB_NAME

python -m mlx --mode image_recognition_oc --platform aws --action status \
  --config ./aws-svdd.yaml --job-name JOB_NAME --watch

python -m mlx --mode image_recognition_oc --platform aws --action stop \
  --config ./aws-svdd.yaml --job-name JOB_NAME

python -m mlx --mode image_recognition_oc --platform aws --action resume \
  --config ./aws-svdd.yaml --job-name FAILED_OR_STOPPED_JOB
```

Every completed epoch publishes an alternating, checksum-validated full-state checkpoint.
Resume requires the original dataset, result root, resource prefix, image, role, infrastructure,
training values, benchmark settings, and frozen variant inventory. Completed variants are skipped
only after their artifact hashes are verified.

## S3 layout

```text
<output_s3_uri>/<resource_prefix>/
├── runs/<run-id>/
│   ├── run-spec.json
│   ├── run-status.json
│   ├── models/<variant-id>/
│   │   ├── <backbone>-deep-svdd.pth
│   │   ├── <backbone>-deep-svdd.last.pth
│   │   ├── training.csv
│   │   ├── training_history.png
│   │   ├── run_metadata.json
│   │   └── benchmark/...
│   ├── recovery/<variant-id>/...
│   └── attempts/<job-name>/...
└── benchmarks/<benchmark-id>/
    ├── benchmark-spec.json
    ├── artifacts/...
    └── attempts/<job-name>/...
```

SageMaker also packages completed artifacts into the attempt's `model.tar.gz`. MLX never creates
or deletes the configured S3 buckets.
