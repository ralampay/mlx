# Zero-shot target-domain adapter evaluation

This local-only, inference-only workflow compares prior validation-selected models without
training or selecting seeds on the target datasets. See [architecture](../../ARCHITECTURE.md).

Use an explicit JSON study containing `foundation`, `seeds`, `studies` (each `path` and
`methods`), `datasets` (each `name`, native `yaml`, `annotations`, `images` root, `metadata`
equal to `acdc` or `mrtmd`), and these fixed `settings`:

```json
{"image_size":640,"precision":"float32","confidence":0.001,"nms_iou":0.6,
 "max_det":300,"coco_max_det":100,"fixed_confidence":0.25,"fixed_iou":0.5}
```

```bash
python -m mlx --mode object-detection --action adapter-zero-shot \
  --study-config /path/study-input.json --output /path/new-experiment \
  --device cuda --study-phase pilot --format json
python -m mlx --mode object-detection --action adapter-zero-shot \
  --study-config /path/study-input.json --output /path/new-experiment \
  --device cuda --study-phase all --format json
python -m mlx --mode object-detection --action adapter-zero-shot-report \
  --output /path/new-experiment --format json
```

`prepare` copies and verifies models/datasets and calibrates the batch. `pilot` evaluates
24 metadata-stratified images for foundation, LoRA seed 1 and revised hybrid seed 1 on
each dataset. `all` evaluates every configured model, then generates the report/gallery.
No real run changes its batch after OOM. Native validation/test aliases are evaluated once.
The report action needs no GPU and never repeats inference.

For standalone model reconstruction, copy the entire `models` directory anywhere:

```python
from pathlib import Path
from mlx.modes.object_detection.zero_shot.data import read_json
from mlx.modes.object_detection.libreyolo.zero_shot_backend import LibreYOLOTransferEvaluator
bundle = Path('/relocated/models')
entry = next(m for m in read_json(bundle/'manifest.json')['models']
             if m['id'] == 'drax-hybrid/seed-1')
model = LibreYOLOTransferEvaluator('cuda').load(bundle, entry)
```

The shared foundation is necessary for compact adapter checkpoints. Dense head-only
restoration intentionally uses foundation backbone/neck plus selected head state, matching
the original study. `source-config.json` paths are provenance only, not reconstruction inputs.
Use the recorded Python/package versions and the source Git archives plus patches/untracked
archives for code reproducibility. Apply patches to the matching archive; source snapshots
do not include datasets or environments. Raw predictions retain confidence ≥0.001; the
offline gallery displays ≥0.25, including all images, crowd GT, and every method/seed.
