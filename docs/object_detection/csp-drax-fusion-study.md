# CSP-Drax fusion low-data study

This study compares `yolox-drax-csp-fusion-m` with `yolox-m` on the existing
deterministic low-data PASCAL VOC 2007 split:

- 1,000 training images;
- 500 validation images used for checkpoint selection;
- all 4,952 test images used once per completed run;
- eight paired seeds: 17, 29, 43, 59, 71, 89, 101, and 127.

The corrected protocol uses 100 epochs, SGD with learning rate 0.001, physical
and effective batch size 8, no gradient accumulation, no gradient clipping,
and the same YOLOX-M pretrained backbone/PAN tensors in both models. Both heads
are reset from the same recorded seed. The order within each pair alternates to
reduce time-order and thermal confounding.

The primary outcome is test AP50-95. The report includes an exact paired
two-sided sign-flip test, paired bootstrap 95% interval, and paired TOST with a
prespecified ±0.02 AP equivalence margin. A non-significant superiority test is
not interpreted as equivalence.

## Run

Download the official YOLOX-M checkpoint from the
[YOLOX 0.1.1 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth),
then run:

```bash
export PYTHONPATH=/path/to/libreyolo-csp-fusion:/home/ralampay/workspace/mlx
cd /home/ralampay/workspace/mlx
env/bin/python scripts/experiments/csp_drax_fusion_voc07.py prepare \
  --source-checkpoint ~/Desktop/object-detection-models/yolox_m.pth
env/bin/python scripts/experiments/csp_drax_fusion_voc07.py smoke
env/bin/python scripts/experiments/csp_drax_fusion_voc07.py run
env/bin/python scripts/experiments/csp_drax_fusion_voc07.py analyze
```

The default output is
`~/Desktop/experiments/csp-drax-fusion-vs-yolox-m-voc07`. Preparation records
the source URL and SHA-256 digest. Completed conditions are resumable and are
not silently reused if their initialization checkpoint changes.

## Pilot invalidation

The prior scratch-trained pilot cannot support a model-effect conclusion. Its
CSP-Drax arm alone used norm-1 gradient clipping, while `nbs=64` converted the
physical batch of 8 into eight-step accumulation for both arms. It also tested
the legacy pre-PAN refinement graph rather than cross-scale feature fusion.
Those results remain useful as diagnostics but are excluded from the corrected
confirmatory analysis.

## Design basis

The implementation is original project code. Its design was informed by:

- [YOLOX](https://arxiv.org/abs/2107.08430) for the control architecture;
- [CSPNet](https://arxiv.org/abs/1911.11929) for partial feature pathways;
- [Selective Kernel Networks](https://arxiv.org/abs/1903.06586) for adaptive branch selection;
- [EfficientDet](https://arxiv.org/abs/1911.09070) for efficient weighted multi-scale fusion;
- [ASFF](https://arxiv.org/abs/1911.09516) for adaptive spatial feature fusion;
- [Dynamic Head](https://arxiv.org/abs/2106.08322) and
  [Gold-YOLO](https://arxiv.org/abs/2309.11331) for attention across detector feature levels;
- [DAMO-YOLO](https://arxiv.org/abs/2211.15444) for allocating capacity to the neck while keeping the head small;
- [ECA-Net](https://arxiv.org/abs/1910.03151) for lightweight channel attention.
