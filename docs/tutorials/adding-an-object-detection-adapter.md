# Adding an object-detection adapter

## Two different meanings

An MLX `DetectionAdapter` translates inference into `DetectionResult`. It is not a trainable
neural adapter. The [empty detector](../../examples/extensions/empty_detector.py) demonstrates
the inference contract with no provider dependency: `predict(frame)` accepts a BGR NumPy image
and returns immutable detection values (pixel xyxy boxes, confidence, class ID, label).

## Attach, execute, and test an inference adapter

```python
import numpy as np
from examples.extensions.empty_detector import EmptyDetector
from mlx.modes.object_detection.commands import RunObjectDetectionStream

class OneFrame:
    def __init__(self):
        self.done = False
    def read(self):
        if self.done:
            return False, None
        self.done = True
        return True, np.zeros((8, 8, 3), dtype=np.uint8)
    def release(self):
        pass

class Sink:
    def show(self, frame):
        return True
    def close(self):
        pass

result = RunObjectDetectionStream(
    detector=EmptyDetector(), frame_source=OneFrame(), frame_sink=Sink(),
    renderer=lambda frame, detections: frame,
).execute()
assert result.frames_processed == 1
```

## Registration, configuration, and discovery

Inference adapters are constructed by a provider's `create_detector(request)`; there is no
separate detector registry. For a new provider, implement the provider capabilities and register
its factory with `ProviderRegistry.register(name, "package.module:factory")`.
Inject that registry into provider-facing commands. Permanent provider names belong in the
provider catalog. `--provider` selects it; `ls-models --names-only` uses its optional
`model_names()` metadata. For Python experiments, direct detector injection needs no registration.

Change the adapter, provider construction/registration if applicable, and tests. Provider objects
stay behind the adapter; do not edit tracking or stream commands.

## Trainable neural adapters

Incremental neural adapters are provider-owned. MLX forwards enable/train-only/type options only
when the selected LibreYOLO train signature or trainer configuration declares support.
The installed version used during this refactor lacked that capability; unsupported requests now
fail before training. Do not advertise an MLX-owned neural-adapter registry that does not exist.

Supported provider integrations must preserve feature-map shapes, load base weights before
attaching new parameters, retain checkpoint adapter metadata, and freeze base parameters and
BatchNorm state for adapter-only training. Adapter-disabled fine-tuning remains unchanged.
Test attachment, disabled equivalence, base-weight loading, and frozen state within the provider;
test MLX option forwarding and capability rejection using fakes. Extending provider architecture
is separate from adding the inference adapter demonstrated above.
