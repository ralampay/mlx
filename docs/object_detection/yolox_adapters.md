# YOLOX-L adapter study

This local-CUDA study always starts from
`~/Desktop/object-detection-models/foundational-yolox-l.pt`. The standard
LibreYOLO YOLOX-L model is instantiated first, the checkpoint is loaded
strictly, and only then are foundation parameters frozen and adapters injected.
The six classes are, in ID order: person, bicycle, motorcycle, car, bus, truck.
No detector classes or checkpoint tensor shapes are changed.

LibreYOLO owns the generic PyTorch adapter modules, registry, injection, YOLOX
placement policy, and strict checkpoint compatibility. MLX owns DAWN conversion,
local CUDA execution, training, evaluation, timing/memory measurements, seeds,
serialization, and aggregation. See `ARCHITECTURE.md` and LibreYOLO's
`docs/yolox_feature_adapters.md` for implementation details and literature.

## Dataset

For evaluation beyond the DAWN adaptation domain, see the
[ACDC and MRTMD zero-shot dataset guide](./zero_shot_datasets.md), including
verified composition tables, BibTeX references, and a LaTeX methods fragment.

The selected target is DAWN v3, downloaded under
`~/Desktop/datasets/object-detection/dawn/original` and converted without
changing the source files into `processed`. DAWN has the exact six foundation
classes and a meaningful rain/fog/snow/sand domain shift. The seed-42 split is
deterministic, exact-size, class/weather balanced, and recorded in
`processed/manifest.json`:

| Split | Images | Boxes | person | bicycle | motorcycle | car | bus | truck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 719 | 5,566 | 329 | 18 | 56 | 4,567 | 113 | 483 |
| validation | 154 | 1,131 | 79 | 4 | 13 | 940 | 19 | 76 |
| test | 154 | 1,148 | 69 | 4 | 12 | 947 | 29 | 87 |

DAWN does not publish capture-sequence identifiers, so near-frame leakage
cannot be excluded. Bicycle and motorcycle are rare, making their per-class
estimates high variance. The foundation corpus included COCO, BDD100K,
VisDrone, MOT17/MOT20, and PCB; those were not reused as the target. Other
candidates considered were ACDC (excellent fit but a 15.6 GB RGB download),
ExDark (no truck), Cityscapes (login and instance conversion), and KITTI
(cyclist semantics do not preserve the six-class mapping).

## Experimental defaults and measured smoke result

The default placement is the three YOLOX neck outputs at strides 8, 16, and 32.
This exposes every detection scale while keeping the backbone and head frozen.
The measured RTX 3060 calibration probed batches 1, 2, 4, and 8 for both Drax
and full fine-tuning with AMP. Both selected the intentionally capped batch 8;
the worst probe peak was 4,711.8 MiB. Real one-epoch Drax training peaked at
1,458.1 MiB allocated. The study uses physical/effective batch 8, AdamW,
640x640 inputs, no mosaic or mixup, and seed 42.

The one-epoch smoke run took 29.47 seconds for 719 images (24.39 images/s),
including validation. It had exact initialization identity, finite loss 3.3070,
unchanged frozen parameters, changed adapter parameters, and wrote a 1.35 MiB
adapter-only checkpoint. Its test metrics were mAP50 0.6527 and mAP50-95 0.4079;
one warmup epoch is an integration check, not evidence of adaptation benefit.
The frozen test metrics were 0.6528 and 0.4080. Fixed score-0.25/IoU-0.50
precision and recall were 0.7354 and 0.7796 for both at this short horizon.

Twenty epochs are recommended for the first exploratory runs. At measured Drax
throughput that is about 9.8 minutes per run; allow roughly 1.3 hours for all
eight trainable conditions plus the frozen baseline and about 6.6 hours for
five seeds. These are linear estimates; other methods can differ in throughput.

## Commands

Run from the MLX checkout after activating its environment. Every research
command explicitly requests CUDA; a missing CUDA runtime aborts before training.

```bash
cd /home/ralampay/workspace/mlx
source env/bin/activate
```

Environment and checkpoint verification:

```bash
nvidia-smi
python - <<'PY'
import platform, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, torch.cuda.get_device_name(i), f"{p.total_memory / 1024**3:.2f} GB")
PY
python -m mlx --mode object-detection --action adapter-verify --model yolox-l \
  --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --device cuda --format json
```

Dataset preparation and calibration (already completed; use a new calibration
output before intentionally repeating it because artifacts are not overwritten):

```bash
python -m mlx --mode object-detection --action adapter-prepare \
  --dataset ~/Desktop/datasets/object-detection/dawn/original \
  --output ~/Desktop/datasets/object-detection/dawn/processed
python -m mlx --mode object-detection --action adapter-calibrate --model yolox-l \
  --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters/calibration-repeat \
  --device cuda --height 640 --width 640 --amp
```

The completed frozen baseline and smoke configuration were:

```bash
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --methods frozen --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters --device cuda \
  --height 640 --width 640 --batch-size 8 --gradient-accumulation 1 --seed 42 --amp
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --adapter drax --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters --device cuda \
  --height 640 --width 640 --batch-size 8 --gradient-accumulation 1 --epochs 1 \
  --seed 42 --adapter-target neck --adapter-reduction 8 --adapter-alpha 1.0 --amp
```

Recommended Drax exploratory run:

```bash
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --adapter drax --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters/exploratory-20ep --device cuda \
  --height 640 --width 640 --batch-size 8 --gradient-accumulation 1 --epochs 20 \
  --seed 42 --adapter-target neck --adapter-reduction 8 --adapter-alpha 1.0 --amp
```

For any individual adapter, replace `METHOD` below with `bottleneck`, `ssf`,
`lora`, `convpass`, `conv-adapter`, or `drax`:

```bash
METHOD=drax
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --adapter "$METHOD" --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters/individual-20ep/$METHOD --device cuda \
  --height 640 --width 640 --batch-size 8 --gradient-accumulation 1 --epochs 20 \
  --seed 42 --adapter-target neck --adapter-reduction 8 --adapter-rank 8 --amp
```

Full single-seed comparison (provided only; not automatically run):

```bash
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --methods frozen,head-only,full-finetune,bottleneck,ssf,lora,convpass,conv-adapter,drax \
  --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters/comparison-20ep --device cuda \
  --height 640 --width 640 --batch-size 8 --gradient-accumulation 1 --epochs 20 \
  --seed 42 --adapter-target neck --adapter-reduction 8 --adapter-rank 8 --amp
```

Five paired seeds (provided only; not automatically run):

```bash
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --methods frozen,head-only,full-finetune,bottleneck,ssf,lora,convpass,conv-adapter,drax \
  --experiment-seeds 1,2,3,4,5 \
  --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters/five-seed-20ep --device cuda \
  --height 640 --width 640 --batch-size 8 --gradient-accumulation 1 --epochs 20 \
  --adapter-target neck --adapter-reduction 8 --adapter-rank 8 --amp
```

Generate aggregate CSV, JSON, and Markdown for any study root:

```bash
python -m mlx --mode object-detection --action adapter-report \
  --output ~/Desktop/experiments/yolox-l-adapters
```

After every declared method and seed has completed, cache per-image predictions for targeted
analysis. This performs evaluation only, requires CUDA, and verifies that each unsliced result
reproduces the original run before accepting its cache:

```bash
python -m mlx --mode object-detection --action adapter-slice-predict \
  --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters/five-seed-20ep \
  --device cuda
```

Generate weather, pooled-weather, object-size, class, class-weather, and frozen-baseline-difficulty
reports from the cache without another GPU pass:

```bash
python -m mlx --mode object-detection --action adapter-slice-report \
  --output ~/Desktop/experiments/yolox-l-adapters/five-seed-20ep \
  --seed 42 --bootstrap-samples 2000
```

Artifacts are stored in `sliced-analysis/` below the study root. Difficulty means tertiles of a
continuous frozen-model detection-quality score using cutoffs fixed on validation data; it is not
claimed to be a causal measure of domain-shift severity. Individual weather slices with fewer than
ten images are marked low support.

Completed runs are never silently overwritten. Use a new output root for a
different epoch count, seed set, precision mode, batch, or adapter configuration.

## Drax hybrid follow-up

`drax-hybrid` adds exact rank-8 LoRA weight updates and compressed local/dilated
spatial bypasses at the same 26 neck convolutions as LoRA. The spatial path uses
depthwise convolutions plus a compressed channel mixer, inspired by Convpass
and the existing Drax adapter. At reduction 8 it trains 2,234,900 parameters
(3.964%). This replaces the earlier three-projection default; saved injection
paths preserve reconstruction of earlier checkpoints. Use a new output directory
for the wider-placement experiment. Memory and runtime must be recalibrated.
Initialization is explicitly seeded before hybrid injection; historical runs
seeded training but did not record a separate adapter-initialization seed.
Reused baselines therefore support an exploratory comparison, not a controlled
architectural ablation. Prior test-set inspection also makes this follow-up
exploratory. Neither better accuracy nor a memory reduction is guaranteed.

Use the completed baseline study read-only; only the hybrid is trained:

```bash
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --adapter drax-hybrid --experiment-seeds 1,2,3,4,5 --epochs 20 \
  --checkpoint ~/Desktop/object-detection-models/foundational-yolox-l.pt \
  --dataset ~/Desktop/datasets/object-detection/dawn/processed \
  --output ~/Desktop/experiments/yolox-l-adapters/drax-hybrid-lora-matched-five-seed-20ep \
  --baseline-study ~/Desktop/experiments/yolox-l-adapters/five-seed-20ep \
  --device cuda --height 640 --width 640 --batch-size 8 --gradient-accumulation 1 \
  --adapter-target neck --adapter-reduction 8 --adapter-rank 8 --adapter-alpha 1 --amp
python -m mlx --mode object-detection --action adapter-report \
  --output ~/Desktop/experiments/yolox-l-adapters/drax-hybrid-lora-matched-five-seed-20ep \
  --comparison-method drax-hybrid
python -m mlx --mode object-detection --action adapter-slice-predict \
  --output ~/Desktop/experiments/yolox-l-adapters/drax-hybrid-lora-matched-five-seed-20ep \
  --comparison-method drax-hybrid --device cuda
python -m mlx --mode object-detection --action adapter-slice-report \
  --output ~/Desktop/experiments/yolox-l-adapters/drax-hybrid-lora-matched-five-seed-20ep \
  --comparison-method drax-hybrid --seed 42 --bootstrap-samples 2000
```

The saved baseline reference is used automatically by reporting and prediction
caching. Baseline metrics/configuration checksums and protocol compatibility are
checked on reuse. Prediction caching additionally verifies checkpoint hashes,
ground truth, evaluator settings, and prediction checksums. Only small cached
predictions are copied; training checkpoints and datasets remain at their source.
