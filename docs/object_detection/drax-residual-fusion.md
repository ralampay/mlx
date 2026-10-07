# DRAX residual fusion adapter

`drax-residual-fusion` is implemented in MLX's `object_detection/feature_adapters/residual_fusion.py`
and exposed through MLX's existing adapter experiment flow. See
[ARCHITECTURE.md](../../ARCHITECTURE.md) for ownership and data flow.

The adapter uses a down-projection, channel LayerNorm at each pixel, SiLU,
local and dilated depthwise 3x3 filters, and a two-channel spatial softmax gate.
It adds the fused spatial update to the compressed features before SiLU and the
up-projection. There is no intermediate dense mixer or LoRA branch. The frozen
base convolution remains the outer bypass. Gates start equally weighted;
the zero-initialized up-projection preserves foundation outputs exactly.
A singleton bottleneck uses LayerNorm's affine parameters without normalization
because centering a single channel would erase its input.

## Training handoff

Use the updated local LibreYOLO checkout; older installed releases do not
register this method. Add these options to the existing experiment invocation:

```text
--adapter drax-residual-fusion --adapter-reduction 16 --adapter-alpha 1 --adapter-target neck
```

For Python callers, use `inject_adapters(model, "drax-residual-fusion", targets,
reduction=16, alpha=1.0, train_head=True)` with targets obtained through
`yolox_targets(model, "neck", "drax-residual-fusion")`. Set `train_head` to match
the comparison protocol. The direct class defaults to reduction 16; shared CLI
and injection defaults remain 8, so specify 16 explicitly. Rank is unused.
Keep checkpoint, dataset, head policy, batch size, seeds and training schedule
matched to the baselines. Use a fresh output directory. Existing study queues
are not modified automatically. No accuracy improvement has yet been measured.

## Parameter comparison

Counts use YOLOX-L's same 26 neck convolution targets. Adapter counts include
all trainable weights and biases. The six-class head contributes 7,552,801
parameters when fully trained; frozen foundation parameters are excluded.

| Method | Adapter parameters | Adapter + six-class head | New adapter savings versus method |
| --- | ---: | ---: | --- |
| LoRA-8 | 178,176 | 7,730,977 | 783,548 more (5.40x adapter size) |
| LoRA-100 | 2,227,200 | 9,780,001 | 1,265,476 fewer (56.8%) |
| DRAX spatial, reduction 8 | 2,056,724 | 9,609,525 | 1,095,000 fewer (53.2%) |
| DRAX residual fusion, reduction 16 | 961,724 | 8,514,525 | Reference |

Per adapted convolution, with input/output channels I/O and hidden width
h = max(1, floor(I / reduction)), the new adapter has
`h * (I + O) + 25 * h + O + 2` trainable parameters.
Counts are regression-tested in LibreYOLO's
`tests/unit/test_drax_residual_fusion_adapter.py`.
