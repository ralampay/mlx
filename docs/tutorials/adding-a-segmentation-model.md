# Adding a segmentation or saliency model

## Contracts and smallest implementation

[Pixel models](../../examples/extensions/pixel_model.py) use a single 1x1 convolution.
A segmentation builder accepts `(name, config, *, num_classes)` and returns spatial logits
`[batch, classes, height, width]`. A saliency builder accepts `(name, config)` and returns
one-channel logits. Do not apply sigmoid in the saliency model: the workflow owns that step.

These are separate mode-owned registries. Shared default U-Nets cross the saliency compatibility
gateway; a new saliency architecture does not require modifying segmentation.

## Register, discover, run, and test

```python
import torch
from examples.extensions.pixel_model import build_segmenter, build_saliency
from mlx.modes.segmentation.models.registry import SegmentationModelRegistry
from mlx.modes.segmentation.models import build_segmentation_model, supported_model_names
from mlx.modes.saliency_mapping.models import SaliencyModelRegistry, build_saliency_model

segmentation = SegmentationModelRegistry({}).register("pixel", build_segmenter)
saliency = SaliencyModelRegistry({}).register("pixel", build_saliency, small=True)
assert supported_model_names(segmentation) == ["pixel"]
x = torch.zeros(2, 3, 8, 8)
assert build_segmentation_model("pixel", {}, num_classes=2, registry=segmentation)(x).shape == (2, 2, 8, 8)
assert build_saliency_model("pixel", {}, registry=saliency)(x).shape == (2, 1, 8, 8)
```

Pass the selected registry as `model_registry` to the owning mode's training, inference,
benchmark, listing, smoke-test, sample, or grouped command. Groups freeze the injected inventory.
`all-small` retains existing built-ins; a custom saliency registration can explicitly opt in.
CLI import references work when the module is installed; local Python aliases are not global.

```bash
python -m mlx --mode segmentation --action test --model examples.extensions.pixel_model:build_segmenter --height 8 --width 8
python -m mlx --mode saliency-mapping --action test --model examples.extensions.pixel_model:build_saliency --height 8 --width 8
```

## Files and expansion

Change the builder, registration, tests, and documentation. The executable train/reload/infer
round trips in `tests/test_extension_roundtrips.py` protect the complete alias-based workflow.
Add encoders or decoder blocks later; keep target interpolation and loss semantics mode-specific.
