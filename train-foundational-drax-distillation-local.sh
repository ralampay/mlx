#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
  cat <<'HELP'
Usage: train-foundational-drax-distillation-local.sh [scratch|fine-tune|resume] [options]

Local LibreYOLO feature distillation; defaults reproduce the foundational Drax run.
CLI options override environment variables. Relative paths use your current directory.

  --model NAME                 Student architecture (default: yolox-drax-mobilenet-v3-large-l)
  --teacher PATH               Frozen teacher .pt (--distiller is an alias)
  --student-checkpoint PATH    Student .pt for fine-tune (--model-path is an alias)
  --dataset PATH               Local YOLO dataset directory or YAML
  --dataset-s3-uri URI         S3 dataset ZIP (default: foundational-urban.zip)
  --output DIR                 Dedicated output directory
  --run-name NAME              Run subdirectory name
  --epochs N                   Default: 50
  --batch-size N               Default: 4
  --nbs N                      Nominal batch, default: 64
  --height N --width N         Default: 640 each
  --device DEVICE              Default: 0; cpu is also supported
  --seed N                     Default: 42
  --optimizer NAME --lr0 FLOAT Default: sgd, 0.01
  --distill-loss cwd|mgd       Default: cwd
  --distill-weight FLOAT       Default: selected method's default
  --distill-temperature FLOAT  CWD only, default: 1.0
  --distill-mask-ratio FLOAT   MGD only, default: 0.65
  --amp | --no-amp             Default: --amp
  --validation-split SPLIT     Default: val
  --validate-after-training | --no-validate-after-training
  --profile NAME --dataset-cache-dir DIR --python EXECUTABLE
  -h, --help                  Show help without loading models or downloading data

scratch always uses --no-pretrained; fine-tune requires a student checkpoint.
No full training is launched by --help. See docs/object_detection/distillation.md.
HELP
}
fail() { echo "$*" >&2; exit 2; }
mode=scratch
if [[ $# -gt 0 && "$1" != -* ]]; then mode=$1; shift; fi
case "$mode" in scratch|fine-tune|resume) ;; *) fail "Unknown mode: $mode" ;; esac
model=${STUDENT_MODEL:-yolox-drax-mobilenet-v3-large-l}
teacher=${TEACHER_CHECKPOINT:-"$HOME/Desktop/object-detection-models/foundational-yolox-l.pt"}
student=${STUDENT_CHECKPOINT:-}
dataset=${DATASET_PATH:-${DATASET_S3_URI:-s3://mlx-object-detection-datasets/foundational-urban.zip}}
if [[ -n "${DATASET_PATH:-}" && -n "${DATASET_S3_URI:-}" ]]; then
  fail 'Set only DATASET_PATH or DATASET_S3_URI, not both.'
fi
default_output="$script_dir/runs/foundational-urban-drax-cwd-scratch"
if [[ "$mode" == fine-tune ]]; then default_output="$script_dir/runs/foundational-urban-drax-cwd-finetune"; fi
output=${OUTPUT_DIR:-"$default_output"}
run_name=${RUN_NAME:-foundational-urban-drax-cwd}
python_bin=${PYTHON:-python}
epochs=${EPOCHS:-50}; batch=${BATCH_SIZE:-4}; nbs=${NBS:-64}
height=${HEIGHT:-640}; width=${WIDTH:-640}; device=${DEVICE:-0}; seed=${SEED:-42}
optimizer=${OPTIMIZER:-sgd}; lr0=${LR0:-0.01}
loss=${DISTILL_LOSS:-cwd}; weight=${DISTILL_WEIGHT:-}
temperature=${DISTILL_TEMPERATURE:-}; mask=${DISTILL_MASK_RATIO:-}
profile=${AWS_PROFILE:-}; cache=${DATASET_CACHE_DIR:-}
amp=--amp; validation=--validate-after-training; split=${VALIDATION_SPLIT:-val}
dataset_option=
while [[ $# -gt 0 ]]; do
  option=$1
  case "$option" in
    -h|--help) usage; exit 0 ;;
    --amp|--no-amp) amp=$option; shift; continue ;;
    --validate-after-training|--no-validate-after-training) validation=$option; shift; continue ;;
    --model|--teacher|--distiller|--student-checkpoint|--model-path|--dataset|--dataset-s3-uri|--output|--run-name|--epochs|--batch-size|--nbs|--height|--width|--device|--seed|--optimizer|--lr0|--distill-loss|--distill-weight|--distill-temperature|--distill-mask-ratio|--profile|--dataset-cache-dir|--python|--validation-split) ;;
    *) fail "Unknown option: $option (use --help)" ;;
  esac
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $option"
  value=$2
  case "$option" in
    --model) model=$value ;;
    --teacher|--distiller) teacher=$value ;;
    --student-checkpoint|--model-path) student=$value ;;
    --dataset|--dataset-s3-uri)
      [[ -z "$dataset_option" || "$dataset_option" == "$option" ]] || fail 'Choose --dataset or --dataset-s3-uri, not both.'
      dataset_option=$option; dataset=$value ;;
    --output) output=$value ;; --run-name) run_name=$value ;;
    --epochs) epochs=$value ;; --batch-size) batch=$value ;; --nbs) nbs=$value ;;
    --height) height=$value ;; --width) width=$value ;; --device) device=$value ;;
    --seed) seed=$value ;; --optimizer) optimizer=$value ;; --lr0) lr0=$value ;;
    --distill-loss) loss=$value ;; --distill-weight) weight=$value ;;
    --distill-temperature) temperature=$value ;; --distill-mask-ratio) mask=$value ;;
    --profile) profile=$value ;; --dataset-cache-dir) cache=$value ;;
    --python) python_bin=$value ;; --validation-split) split=$value ;;
  esac
  shift 2
done
[[ -f "$teacher" ]] || fail "Teacher checkpoint not found: $teacher"
[[ "$run_name" != */* && "$run_name" != . && "$run_name" != .. ]] || fail 'Run name must be a single directory name.'
if [[ "$mode" == fine-tune ]]; then
  [[ -n "$student" && -f "$student" ]] || fail 'fine-tune requires STUDENT_CHECKPOINT or --student-checkpoint pointing to an existing .pt file.'
elif [[ -n "$student" ]]; then
  fail 'A student checkpoint is only accepted in fine-tune mode; resume uses the output checkpoint.'
fi
if [[ "$mode" == resume ]]; then
  [[ -f "$output/$run_name/weights/last.pt" ]] || fail "Resume checkpoint not found: $output/$run_name/weights/last.pt"
elif [[ -d "$output" ]] && [[ -n "$(find "$output" -type f -name '*.pt' -print -quit)" ]]; then
  fail 'Output already contains checkpoints. Use resume or set a new --output directory.'
fi
args=(--mode object_detection --platform local --provider libreyolo
  --action train --model "$model" --output "$output" --run-name "$run_name" --distiller "$teacher"
  --distill-loss "$loss" --epochs "$epochs" --batch-size "$batch" --nbs "$nbs"
  --height "$height" --width "$width" --device "$device" --random-seed "$seed"
  --optimizer "$optimizer" --lr0 "$lr0" "$amp" --no-pretrained --use-best
  "$validation" --validation-split "$split")
if [[ "$dataset" == s3://* ]]; then
  [[ "$dataset_option" != --dataset ]] || fail 'Use --dataset-s3-uri for S3 data.'
  args+=(--dataset-s3-uri "$dataset")
else
  [[ "$dataset_option" != --dataset-s3-uri ]] || fail '--dataset-s3-uri requires an s3:// URI.'
  [[ -e "$dataset" ]] || fail "Local dataset not found: $dataset"
  args+=(--dataset "$dataset")
fi
case "$loss" in
  cwd) [[ -z "$mask" ]] || fail '--distill-mask-ratio applies only to MGD.'
       args+=(--distill-temperature "${temperature:-1.0}") ;;
  mgd) [[ -z "$temperature" ]] || fail '--distill-temperature applies only to CWD.'
       args+=(--distill-mask-ratio "${mask:-0.65}") ;;
  *) fail '--distill-loss must be cwd or mgd.' ;;
esac
if [[ -n "$weight" ]]; then args+=(--distill-weight "$weight"); fi
if [[ "$mode" == fine-tune ]]; then args+=(--action fine-tune --model-path "$student"); fi
if [[ -n "$profile" ]]; then args+=(--profile "$profile"); fi
if [[ -n "$cache" ]]; then args+=(--dataset-cache-dir "$cache"); fi
mkdir -p "$output"
PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m mlx "${args[@]}" 2>&1 | tee -a "$output/training.log"
