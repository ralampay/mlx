# Adding an autoencoder

## Contract and implementation

Autoencoders consume numeric matrices `[batch, input_dimensions]`.
`encode(x)` returns the bottleneck `[batch, bottleneck_dimensions]`;
`decode(z)` reconstructs the original shape; `forward(x)` composes the two.

The deliberately minimal reference is
[`TinyAutoencoder`](../../mlx/modes/autoencoder/architectures/tiny.py): two linear layers,
with no unnecessary base class beyond PyTorch's `nn.Module`. Its definition supplies
`name`, `description`, and `build(config)`.
Copy that file to your own installable package to experiment, or put a new built-in architecture
next to it. Preserve the dimension attributes and use ordinary PyTorch parameter registration.

## Register, discover, configure, run, and test

This runnable example gives the reference implementation a local experiment name:

```python
import torch
from mlx.modes.autoencoder.model_registry import AutoencoderRegistry
from mlx.modes.autoencoder.commands import ListAutoencoderModels

reference = "mlx.modes.autoencoder.architectures.tiny:TinyAutoencoderDefinition"
registry = AutoencoderRegistry({}).register("my-tiny", reference, description="Two linear layers")
assert ListAutoencoderModels(registry).execute()[0]["name"] == "my-tiny"
definition, path = registry.resolve("my-tiny")
model = definition.build({"input_dimensions": 4, "bottleneck_dimensions": 2})
x = torch.zeros(3, 4)
assert model.encode(x).shape == (3, 2)
assert model.decode(model.encode(x)).shape == x.shape
assert model(x).shape == x.shape
```

Pass this registry to `TrainAutoencoder(request, model_registry=registry)` and call
`execute()`. For your external class, replace the reference with its real import path.
For a built-in alias, add it to `BUILTIN_AUTOENCODERS` and its description in the owning registry.

```bash
python -m mlx --mode autoencoder --action ls-models
python -m mlx --mode autoencoder --action train --model tiny --input vectors.csv --output tiny-run --bottleneck-dim 2 --epochs 1
python -m mlx --mode autoencoder --action embed --model-path tiny-run/autoencoder.pth --input vectors.csv --output latent.csv
```

External checkpoints require an exact reference in the caller's registry, or
`--trust-checkpoint-code`. This permits importing trusted code; it does not permit unsafe
pickle loading. Built-in checkpoints need no opt-in.

## Files and expansion

Only the architecture, local registration/catalog entry, and tests change. Keep the shape test
above and add a train/checkpoint/reload test like `test_extension_roundtrips.py`.
You can later change encoder/decoder internals without changing embedding or retrieval commands.
