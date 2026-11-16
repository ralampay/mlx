# Local scratch architecture comparison

From the checkout, use `PYTHONPATH=. env/bin/python scripts/experiments/architecture_suite.py ACTION --config recipe.json` with actions
`prepare`, `pilot`, `run`, `status`, and `analyze`. Orchestration and provider boundaries
are documented in [ARCHITECTURE.md](../../ARCHITECTURE.md).

The explicit JSON recipe supplies `output`, ordered `datasets` (name to audited directory),
`models` (two or three registered names), `seeds` (2–20 distinct integers), `epochs`,
`image_size`, `budget_hours`, and `sources` (name to repository directory).
Prepare each dataset using `PrepareComparisonDataset(source, output).execute()` first.
Do not point new output directories at an existing experiment.

Training is scratch-only, CUDA-only, SGD lr=.001, effective batch eight, AMP, warmup five
epochs, final no-augmentation interval fifteen epochs, without early stopping or clipping.
All architectures inherit the provider's YOLOX augmentation and optimizer defaults;
native training arguments and source snapshots record the resolved recipe. Pilots use
label capacity 600 (also used by main runs) to retain dense composite-augmentation targets.
Cached COCO rescoring must match native AP and supplies fixed-threshold precision/recall
at confidence .25 and IoU .5, plus per-class AP. Pilots use
three epochs and validation only. Model seed `s` and data seed `s+10000` are distinct.
Fresh model construction precedes the provider's data RNG reset. Physical batch calibration
uses powers of two through eight, leaving at least 20% free device memory. Small physical
batches retain native BatchNorm behavior; accumulation is not equivalent to larger-batch BN.

The budget selects an ordered dataset prefix using measured pilot timing plus 25% contingency.
This is an estimate, not a hard kill deadline. Existing results, models, and datasets are
never silently overwritten. Exact and conservative near-duplicate groups are removed only
from new dataset copies; thresholding cannot prove source-scene independence.

Manual operational validation: inspect dataset audit exclusions and class counts; confirm
no overlapping training job; check local CUDA computation and disk capacity; execute pilots;
inspect logs for finite loss, saved best checkpoints, reload/evaluation success and runtime
admission. Then launch `run` with `scripts/experiment-job.py` for detached supervision.
An interrupted run directory must be moved to an explicit archive before restarting its seed;
completed run receipts are reused only when checkpoint and experiment identities match.
After successful evaluation, the full best checkpoint is retained and only that new run's
redundant `training/formal/weights/last.pt` is retired, with both hashes recorded. Failed or
interrupted runs retain their terminal checkpoint; no earlier experiment files are removed.

Reports provide descriptive results during training, and exact paired two-sided sign flips
plus paired TOST within ±1 AP point only when every declared pair is complete. One Holm
family covers all comparisons and both claim types. Sign-flip inference assumes symmetry
under the null; t-based TOST assumes approximately normal seed differences. Bootstrap CIs
are unadjusted descriptive intervals. Zero variance is flagged and claims withheld.
These tests measure seed variability on fixed, previously examined datasets, not independent
dataset replication. Neither a 100-epoch schedule nor ten seeds guarantees convergence/power.
