# Adding a classifier

## Contract and implementation

The [mean classifier](../../examples/extensions/mean_classifier.py) pools each image to its
channel means and applies one linear layer. Its factory accepts `num_classes`, `colored`,
`pretrained`, and optional `config`. Output is class logits `[batch, classes]`.

Place custom builders in your package; built-ins live under the classification model package.

## Register, discover, run, and test

```python
import torch
from examples.extensions.mean_classifier import build_classifier
from mlx.modes.image_classification.models.standard import StandardModelRegistry
from mlx.modes.image_classification.models import build_image_classification_model, supported_model_names

registry = StandardModelRegistry({}).register("mean", build_classifier)
assert "mean" in supported_model_names(registry)
model = build_image_classification_model("mean", {}, num_classes=2, registry=registry)
assert model(torch.zeros(3, 3, 8, 8)).shape == (3, 2)
```

Pass `model_registry=registry` to classification training, inference, benchmarking, listing,
smoke testing, or CAM commands. Keep the registry available when reloading alias-based checkpoints.

The installed example can also be selected without registering an alias:

```bash
python -m mlx --mode image-classification --action test --model examples.extensions.mean_classifier:build_classifier --height 8 --width 8
```

Detailed `ls-models` counts parameters; `--names-only` avoids construction.
Deep-SVDD/feature reuse additionally requires the feature contract or a registered feature
adapter; a plain classifier is not automatically a valid feature backbone.

## Files and expansion

Change the builder, optional local/catalog registration, and tests—not dispatch.
Replace pooling with a larger architecture later without changing the factory contract.
