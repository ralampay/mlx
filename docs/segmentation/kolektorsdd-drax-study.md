# KolektorSDD small DRAX study

This study compares scratch-trained semantic segmentation models on the
[KolektorSDD-SEG archive](https://zenodo.org/records/12704122). The source
contains 52 paired 256 × 256 images with sparse binary defects. The study
targets fewer than 10 million parameters and practical batch-one CPU inference.

Run the reproducible commands from the repository root:

```bash
python -m scripts.experiments.kolektorsdd_drax prepare
python -m scripts.experiments.kolektorsdd_drax run --epochs 30 --batch-size 4
python -m scripts.experiments.kolektorsdd_drax analyze
```

The preparation command verifies the published archive checksum and writes
three area-stratified outer folds. Every model receives the same fold-specific
training, inner validation, and held-out test images. Checkpoints are selected
on inner-validation foreground Dice. The loss is cross entropy plus soft Dice
with a foreground class weight of 20; no pretrained weights are used.

The experiment directory is
`~/Desktop/experiments/semantic-segmentation/kolektorsdd-drax-small/`.
Each completed run writes per-image results and CPU latency. `summary.csv` and
`contrasts.csv` are created by `analyze`. The four prespecified contrasts test
skip placement, attention contribution, branch scaling, and overall gain over
plain MobileNet. Two-sided paired sign-flip tests use 100,000 sampled sign
assignments; 20,000 paired bootstrap samples provide effect intervals. Holm
correction covers the four image-level p-values. Exact sign-flip tests across
the three fold-level differences are reported as a sensitivity check. The
image-level tests are exploratory and can overstate evidence because images
within each fold share a fitted model. Predicted foreground coverage is reported
beside Dice to identify constant-mask failures. CPU latency is remeasured after
training in a single process at four threads.
