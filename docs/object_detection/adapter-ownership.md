# Research adapter ownership and portable exports

New research adapter implementations belong to MLX's
`mlx.modes.object_detection.feature_adapters`. The legacy `libreyolo.adapters`
package remains unchanged in behavior and usable without MLX. New experiments
use MLX's implementation; historical source snapshots are not rewritten.
The migration preserves tensor names, initialization and mathematical operations.
These modules originate in this project's LibreYOLO worktree, licensed MIT by
Libre YOLO Contributors. The original license is retained beside the modules.

YOLOX target selection belongs to `libreyolo.adapter_targets` inside the MLX
object-detection mode, while generic feature modules have no provider dependency.
See the repository's canonical [architecture](../../ARCHITECTURE.md).

## Loading a complete adapter detector

The `mlx-yolox-adapter-detector-v1` checkpoint includes all foundation and adapter
tensors. It does not need the foundation file at its historical path and does
not pickle a custom module instance. Use MLX's loader, not an unmodified standard
LibreYOLO checkpoint constructor:

```python
from pathlib import Path
from mlx.modes.object_detection.libreyolo.adapter_checkpoint import LoadAdapterDetector

detector = LoadAdapterDetector(
    Path.home() / "Desktop/yolox-l-adapter-drf.pt", device="cuda"
).execute()
result = detector.predict("image.jpg")
```

CUDA requests fail explicitly when unavailable. CPU is supported for inspection
and tests. The exporter selects the highest validation AP50–95 over the requested
seeds, breaks ties by lower seed, verifies reconstruction, and backs up an existing
destination before atomic replacement. Test scores never select the export.

## DAWN priority protocol

The priority study retrains residual fusion and LoRA-8 for seeds 1–5, twenty
epochs each, using the existing DAWN split and original six-class head. The
head and original feature parameters/buffers remain frozen. Both methods use
the same 26 dense neck convolutions. Fusion uses reduction 16/alpha 1; LoRA
uses rank 8/alpha 1. Both adapter initializations are explicitly seeded before
model construction, unlike historical LoRA runs that did not record that policy.

The recipe retains 640-pixel inputs, CUDA AMP, batch/effective batch 8 and
AdamW learning rate 0.0001. Historical train configuration is copied as protocol
provenance. Calibration must accept batch 8; otherwise the run fails without
changing the research configuration. Reports include fresh five-seed paired
differences; historical LoRA scores are not pooled into this comparison.

The launcher lets the active DIOR condition finish while its dispatchers are
held. It restores only matching PID/start-time identities after the DAWN child
exits, including failure paths. This is an in-memory hold, not reboot-proof
checkpoint resumption. SIGKILL or machine shutdown requires manual inspection;
normal cancellation restores queues after the training child has exited.

An optional `include_lora100` recipe flag adds five rank-100/alpha-12.5 runs
after the fusion/LoRA-8 pairs. They have their own calibration, one-epoch smoke,
and artifact root. Combined reports label these rows `lora-r100` and retain a
`source_run` path; the actual backend method remains `lora`. Frozen evaluations
from the additional study are not counted twice. All three methods retain the
same dataset, frozen head, twenty epochs and physical/effective batch eight.
