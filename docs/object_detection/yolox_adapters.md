# YOLOX-L adapter experiment

The experiment always starts each condition from the strict-loaded standard
YOLOX-L checkpoint `~/Desktop/object-detection-models/foundational-yolox-l.pt`.
The checkpoint has six classes in this order: person, bicycle, motorcycle,
car, bus, truck. Its names match the prepared ACDC dataset without remapping.
No new foundation model or detection head is constructed.

The selected source is `~/Desktop/datasets/object-detection/eval2` (ACDC,
349 images, 2,146 boxes, about 639 MiB). Its source folder has one prepared
test split. `adapter-prepare` makes a seed-42, sequence-disjoint research
train/validation/test split with symlinks, preserving source files. The split
is recorded in `output/dataset/manifest.json`. The sequence-level split
reduces near-neighbor frame leakage. Exact independence from the foundation
training corpus remains unverified. Rare bicycle and motorcycle classes have
few validation examples; per-class conclusions should be cautious.

The prepared ACDC annotations use the foundation class order directly.
The source ACDC category IDs had already been mapped as 24/25→person,
33→bicycle, 32→motorcycle, 26→car, 28→bus, and 27→truck. No detection
head resizing or further class remapping occurs in the adapter experiment.

The default adapter placement is the three YOLOX neck outputs. Default
training uses AdamW, batch size 2, 640x640 images, a fixed seed, and no
mosaic or mixup. Use 1 epoch to smoke test and 12 epochs for an exploratory
run. The frozen method evaluates immediately and runs automatically before any
adapted method unless a matching frozen run already exists. All methods use the same
prepared split and validation path. This is an exploratory comparison, not a
publication-grade experiment; later multi-seed runs can use
`--experiment-seeds 1,2,3,4,5`. Reports show per-method performance and
paired Drax differences, with descriptive 95% intervals when more than one
paired seed is available. A future non-inferiority analysis needs a
prespecified margin and a powered design.

COCO mAP50 and mAP50-95 come from LibreYOLO validation. The experiment
separately computes class-aware micro precision and recall at score 0.25 and
IoU 0.50 because LibreYOLO's legacy `precision` and `recall` result keys
alias AP and AR rather than a fixed operating point. Recorded forward latency
is batch-1 raw-model time after warmup, excluding decode and NMS.

See `python -m mlx --help` for options and `libreyolo/docs/yolox_feature_adapters.md`
for architecture, parameter formulas and literature references.

## Commands

Run from the MLX checkout with LibreYOLO importable. `--checkpoint` aliases
the existing `--model-path` option; for YOLOX-L it defaults to the foundation
checkpoint shown below.
Other YOLOX sizes can be selected with an explicit, size-matched checkpoint;
the initial study and default checkpoint remain YOLOX-L.

```bash
python -m mlx --mode object-detection --action adapter-verify --model yolox-l --format json
python -m mlx --mode object-detection --action adapter-prepare --model yolox-l \
  --dataset ~/Desktop/datasets/object-detection/eval2 --output ./results/yolox-l-adapters
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --methods frozen --dataset ~/Desktop/datasets/object-detection/eval2 \
  --output ./results/yolox-l-adapters --device cuda:0 --batch-size 2 --seed 42
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --adapter drax --dataset ~/Desktop/datasets/object-detection/eval2 \
  --output ./results/yolox-l-drax-smoke --device cuda:0 --batch-size 2 --epochs 1 --seed 42
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --adapter drax --dataset ~/Desktop/datasets/object-detection/eval2 \
  --output ./results/yolox-l-drax --device cuda:0 --batch-size 2 --epochs 12 --seed 42
python -m mlx --mode object-detection --action adapter-experiment --model yolox-l \
  --methods frozen,head-only,full-finetune,bottleneck,ssf,lora,convpass,conv-adapter,drax \
  --dataset ~/Desktop/datasets/object-detection/eval2 \
  --output ./results/yolox-l-comparison --device cuda:0 --batch-size 2 --epochs 12 --seed 42
python -m mlx --mode object-detection --action adapter-report --output ./results/yolox-l-comparison
```

For five paired seeds, append `--experiment-seeds 1,2,3,4,5` to the all-method
command. This is a future run; it has not been launched. Every seed gets its
own frozen baseline. If an output already contains a method/seed run, choose
a new output root; completed runs are not overwritten.
