# Adding a loss

## Contract and smallest implementation

[AbsoluteError](../../examples/extensions/absolute_error.py) subclasses `nn.Module` and returns
`(prediction - target).abs().mean()`. That is the complete mathematical implementation.
Its definition provides metadata and `build(config)` for autoencoder selection.

A training loss must return one finite scalar tensor connected to the prediction graph.
Target shapes and semantics remain the owning mode's responsibility; a reconstruction loss
is not automatically appropriate for classification labels.

## Register, configure, discover, run, and test

```python
import torch
from mlx.modes.autoencoder.losses import ReconstructionLossRegistry
from mlx.modes.autoencoder.commands import ListAutoencoderLosses
from mlx.core.losses import validate_scalar_loss

registry = ReconstructionLossRegistry({}).register(
    "absolute", "examples.extensions.absolute_error:AbsoluteErrorDefinition",
)
assert ListAutoencoderLosses(registry).execute()[0]["name"] == "absolute"
definition, _ = registry.resolve("absolute")
prediction = torch.tensor([1., 3.], requires_grad=True)
loss = definition.build({})(prediction, torch.zeros(2))
assert loss.item() == 2.
validate_scalar_loss(loss, training=True)
loss.backward()
assert prediction.grad is not None
```

Pass `loss_registry=registry` to `TrainAutoencoder`, with request `loss="absolute"`.
Or select the definition path using `--loss`. The example accepts no options.
Native classification/segmentation accept a loss-module import path or a definition with
`build(config)`; inject a bound `build_loss(..., entries=...)` factory for local aliases.
Use `ls-losses` in the relevant mode to discover built-ins. Definitions stay mode-owned;
do not register algorithm-intrinsic objectives globally.

## Files and expansion

Change only the loss module, local/catalog registration, and tests. The complete one-epoch
custom-loss exercise is in `tests/test_extension_roundtrips.py`.
Add validated options to the definition later; training commands still obtain the loss through
the same factory.
