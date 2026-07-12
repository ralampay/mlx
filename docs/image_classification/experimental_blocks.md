# Experimental Blocks

Mode: `image_classification`

This document tracks the project-specific experimental building blocks used by custom image-classification backbones.

## DraxNet

`draxnet` is the current experimental standard classifier in this repository.

At the backbone level, it is a local `ResNet-18` implementation with a configurable per-stage block layout. The default stage pattern is:

```text
("basic", "basic", "basic", "drax")
```

That means:

- `layer1`: `BasicResidualBlock x 2`
- `layer2`: `BasicResidualBlock x 2`
- `layer3`: `BasicResidualBlock x 2`
- `layer4`: `DraxResidualBlock x 2`

So the current default model keeps the `ResNet-18` scaffold and only substitutes the final stage with the project-specific residual block.

`Drax` stands for `Dynamic Residual Attention eXchange`.

## DraxNet Structure

Conceptually, the current default `draxnet` looks like this:

```text
Input
  ↓
Stem
  7x7 conv, stride 2
  BN + ReLU
  3x3 maxpool, stride 2
  ↓
Layer1
  BasicResidualBlock x 2
  ↓
Layer2
  BasicResidualBlock x 2
  ↓
Layer3
  BasicResidualBlock x 2
  ↓
Layer4
  DraxResidualBlock x 2
  ↓
Global Avg Pool
  ↓
FC
```

Compared to plain `resnet18`, the network-level difference is:

```text
ResNet-18:  [basic, basic, basic, basic]
DraxNet:    [basic, basic, basic, drax]
```

## DraxResidualBlock

`DraxResidualBlock` is the outer custom residual block used by `DraxNet`.

Its current structure is:

```text
x
 ├─ shortcut ──────────────────────────────────────────────┐
 └─ 3x3 conv → BN → ReLU → DraxBlock → 3x3 conv → BN ────┤
                                                           +
                                                           ↓
                                                         ReLU
```

Key points:

- it preserves the standard residual-block contract
- it can downsample through the first `3x3` convolution and shortcut projection
- it inserts the project-specific `DraxBlock` as the internal mixer
- it is heavier than a plain `ResNet-18` basic block, which is why `draxnet` has more parameters than `resnet18`

## DraxBlock

`DraxBlock` is the inner project-specific mixer used inside `DraxResidualBlock`.

Its current implementation combines:

- a `ConvNeXtBlock` local branch
- an optional `SelfAttention2D` branch
- optional reduced-dimensional attention projections for the efficient mode
- residual fusion through `DropPath`

Conceptually:

```text
x
 ├─ ConvNeXt-style branch
 │    ConvNeXtBlock(x) - x
 │
 ├─ Attention branch
 │    SelfAttention2D(x or reduced x) - x
 │
 └─ Fuse
      average: 0.5 * (conv_delta + attention_delta)
      sknet: input-dependent channel-wise weighted sum
      → DropPath
      → residual add
```

Key points:

- it is a project-specific hybrid mixer, not a pure ConvNeXt block
- it is currently closer in spirit to conv-attention hybrid literature than to pure ResNet blocks
- it is designed to be replaceable as experiments continue

The `fusion_mode` constructor option selects the fusion strategy:

- `average` is the default and preserves the fixed `0.5 / 0.5` coefficients
- `sknet` globally pools the summed branch deltas, predicts two logits per channel through a bottleneck MLP, and applies a branch-wise softmax

The SKNet-style mode therefore allows each input and feature channel to choose a different balance between local convolution and global attention. It follows the adaptive selection mechanism from [Selective Kernel Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Selective_Kernel_Networks_CVPR_2019_paper.pdf), without reproducing SKNet's convolution branches or BatchNorm-equipped gate.

## Related Naming

The current naming split is:

- `BasicResidualBlock`: local plain `ResNet-18`-style block
- `DraxBlock`: internal custom mixer
- `DraxResidualBlock`: outer custom residual block
- `DraxNet`: full backbone

This keeps generic baseline components separate from project-specific experimental components.

## Configuration Notes

The main configuration hook for block substitution is:

- `draxnet_stage_blocks`
- `draxnet_fusion_mode`, either `average` (default) or `sknet`

Current default:

```text
basic,basic,basic,drax
```

Backward compatibility:

- the old token `cax` is still accepted as an alias for `drax`

Pretrained weights:

- pretrained loading currently works only when all stages use `basic` blocks
- that restriction exists because the available pretrained weights come from `torchvision` `resnet18`
- once `DraxResidualBlock` is introduced, the block structure no longer matches `torchvision` `resnet18`

## Benchmarking Guidance

For fair comparison, the most useful benchmark ladder is:

1. `resnet18`
2. `draxnet` with all `basic` stages
3. `draxnet` with the current `drax` stage in `layer4`
4. `convnext_tiny`

This separates:

- the baseline `ResNet-18` scaffold
- the effect of the project-specific block substitution
- the comparison against a stronger modern conv backbone

## Drax MobileNet V3 Large

`drax_mobilenet_v3_large` is a hybrid that keeps the standard `mobilenet_v3_large` feature extractor and adds a bottlenecked `DraxBlock` refiner after the final `960`-channel feature stage.

Conceptually:

```text
MobileNetV3 features
  ↓
1x1 down-project
  ↓
DraxBlock x N
  ↓
1x1 up-project
  ↓
Residual add back to MobileNet features
  ↓
Global Avg Pool
  ↓
Classifier
```

This design is intentional:

- it preserves the pretrained MobileNet backbone
- it applies Drax where the spatial map is already compact
- it avoids replacing internal inverted residual blocks, which would break more pretrained structure
- it keeps parameter growth controlled through the adapter bottleneck instead of running Drax at full `960` channels

Its fusion strategy is configured independently with `drax_mobilenet_fusion_mode`, using either `average` (default) or `sknet`.
