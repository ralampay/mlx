# Local feature distillation

MLX can train a LibreYOLO detector with a frozen detector teacher, either from
random initialization or by fine-tuning a student checkpoint. The teacher runs
only during training. The student remains independently usable for inference.
Package ownership and data flow are documented in [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Train locally with your own model and data

Install MLX's `object-detection-libreyolo` extra in the active Python environment.
For S3 data, also install the `aws` extra and configure credentials with read access
to the dataset object. Local datasets need no AWS credentials.
The script uses MLX's existing S3 download cache; it does not launch SageMaker.

```bash
bash ./train-foundational-drax-distillation-local.sh scratch
```

Use `--help` to see all script options. The original experiment remains the default,
but any compatible LibreYOLO student and teacher can be selected. Supported families
must expose compatible distillation feature taps; arbitrary architectures are not
guaranteed to match.

```bash
# Local YOLO dataset directory or data.yaml; relative paths use your current directory.
bash /path/to/mlx/train-foundational-drax-distillation-local.sh scratch \
  --model yolox-s \
  --teacher ./weights/teacher.pt \
  --dataset ./my-dataset/data.yaml \
  --output ./runs/my-student \
  --run-name experiment-1 \
  --epochs 100 --batch-size 8 --device 0

# A different S3 dataset and MGD feature distillation.
bash ./train-foundational-drax-distillation-local.sh scratch \
  --model yolox-m --teacher ./teacher.pt \
  --dataset-s3-uri s3://my-bucket/detection.zip \
  --output ./runs/my-mgd --distill-loss mgd --distill-mask-ratio 0.65

# Fine-tune an existing student with the same interface.
bash ./train-foundational-drax-distillation-local.sh fine-tune \
  --teacher ./teacher.pt --student-checkpoint ./student.pt \
  --dataset ./my-dataset --output ./runs/my-finetune
```

CLI options override environment values. Choose only one dataset source: `--dataset`
for local data or `--dataset-s3-uri` for S3. A CLI dataset choice replaces the default
or environment dataset. Existing environment overrides remain supported. Use a
separate output for each experiment. For fine-tuning, the loaded checkpoint determines
the actual student architecture.

Defaults: random student `yolox-drax-mobilenet-v3-large-l`, local teacher
`$HOME/Desktop/object-detection-models/foundational-yolox-l.pt`, dataset
`s3://mlx-object-detection-datasets/foundational-urban.zip`, 50 epochs, 640×640,
batch 4, nominal batch 64 (gradient accumulation), SGD, lr0 0.01, AMP, seed 42,
GPU 0, CWD weight 1.0 and temperature 1.0. `--no-pretrained` disables ImageNet
initialization. The script refuses scratch/fine-tune runs in an output containing
checkpoints so automatic recovery cannot silently replace scratch initialization.

Default scratch/resume output is `runs/foundational-urban-drax-cwd-scratch`;
fine-tune defaults to `runs/foundational-urban-drax-cwd-finetune`. The run subdirectory
is `foundational-urban-drax-cwd`. Console output appends to `training.log` in the
output directory. Fifty epochs is an experiment budget, not a convergence guarantee.
The GPU must fit both teacher and student. Reduce `BATCH_SIZE` if necessary;
nominal batch stays 64 but microbatch changes can affect BatchNorm and throughput.

```bash
EPOCHS=100 BATCH_SIZE=2 DEVICE=0 OUTPUT_DIR=./runs/drax-cwd-100 \
  bash ./train-foundational-drax-distillation-local.sh scratch

STUDENT_CHECKPOINT="$HOME/Desktop/object-detection-models/foundational-yolox-drax-mobilenet-v3-l.pt" \
OUTPUT_DIR=./runs/drax-cwd-finetune \
  bash ./train-foundational-drax-distillation-local.sh fine-tune

# Use the same output, teacher, loss settings, seed, and training configuration.
bash ./train-foundational-drax-distillation-local.sh resume

# Start from the repository directory; detach with Ctrl-b then d.
tmux new-session -s drax-training 'bash ./train-foundational-drax-distillation-local.sh scratch'
# Later:
tmux attach-session -t drax-training
```

Environment overrides: `STUDENT_MODEL`, `TEACHER_CHECKPOINT`, `STUDENT_CHECKPOINT`,
`DATASET_PATH` or `DATASET_S3_URI` (not both), `OUTPUT_DIR`, `RUN_NAME`, `EPOCHS`,
`BATCH_SIZE`, `NBS`, `HEIGHT`, `WIDTH`, `DEVICE`, `SEED`, `OPTIMIZER`, `LR0`,
`DISTILL_LOSS`, `DISTILL_WEIGHT`, `DISTILL_TEMPERATURE`, `DISTILL_MASK_RATIO`,
`VALIDATION_SPLIT`, `DATASET_CACHE_DIR`, `AWS_PROFILE`, and `PYTHON`.
Use `--no-amp` for FP32 and `--no-validate-after-training` to omit the extra final
benchmark (provider training-time validation still runs).
Keep the machine awake. Terminal detachment
does not protect against shutdown or GPU failure; resume recovers the latest
complete checkpoint. There are no SageMaker compute charges. S3 download/request
charges and local electricity still apply. The compressed dataset is about 22.5 GB;
allow space for both archive and extraction plus model artifacts.

## CLI and Python interface

`--distiller` is the teacher path, while `--model-path` initializes the student.
Distillation supports local object-detection `train` and `fine-tune` with
`--provider libreyolo`. Other providers, AWS runs, and inference actions reject
these options. Without a teacher, normal training behavior is unchanged.

| Option | Default with teacher | Meaning |
|---|---|---|
| `--distiller PATH` | Required to enable | Existing local teacher `.pt` |
| `--distill-loss cwd\|mgd` | `cwd` | Feature-distillation objective |
| `--distill-weight FLOAT` | CWD: 1.0; MGD: 0.00002 | Positive loss weight |
| `--distill-temperature FLOAT` | 1.0 | Positive CWD temperature, CWD only |
| `--distill-mask-ratio FLOAT` | 0.65 | MGD mask fraction in `[0,1)`, MGD only |

The script defaults to CWD and accepts `--distill-loss mgd` for MGD. If weight is
omitted, the selected method's default applies; CWD's weight is not reused for MGD. CLI fields are also
available on `TrainObjectDetectionRequest` and `FineTuneObjectDetectionRequest`
for direct Python calls through the existing training commands.

CWD compares channel-wise spatial distributions at the three YOLOX neck outputs
(P3/P4/P5). Both L variants expose matching 256/512/1024 channels. This is feature
distillation, not matching final class scores or predicted boxes. The objective
is detection loss plus weighted feature loss. MGD instead reconstructs teacher
features using training-only generator modules. Teacher features are detached;
only student and applicable distillation-module parameters are optimized.

`distillation.json` records the teacher SHA-256 and effective loss settings.
Resume requires matching contents/settings and this provenance file. Relocating
an identical teacher is allowed. To add distillation to an ordinary checkpoint,
use explicit fine-tuning in a new output, not resume. Preserve the original run
configuration when resuming; changing epochs can alter scheduling semantics.

The deployment model retains approximately 31.68M parameters for the six-class
Drax-L student. Loading the student checkpoint for prediction does not load the
teacher. Full training checkpoints can additionally contain optimizer and
`distiller` state for recovery; these are not inference parameters.

Compare against a no-distillation student with the same initialization policy,
training budget and seeds. Keep external evaluation datasets out of training and
model selection. Accuracy improvement or equivalence to YOLOX-L is not guaranteed.

## Verification

Focused MLX tests cover CLI/request mapping, invalid options, resume provenance,
provider forwarding, script modes, and propagation of failed training exit codes.
An opt-in offline synthetic test exercises real CWD/MGD scratch training,
fine-tuning, recovery, frozen teacher parameters, and student-only reload:

```bash
MLX_DISTILLATION_SMOKE=1 python -m pytest -q tests/test_distillation_integration.py
```

This uses tiny CPU models and verifies integration, not full-size GPU throughput,
convergence, or accuracy. LibreYOLO is reused without source modifications.
