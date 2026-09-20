---
name: experimental-setup
description: "Design a reproducible machine-learning experiment using the current repository and generate a concrete copy-pasteable shell command or shell block to run it. Every experiment must use an explicit output directory and preserve its artifacts under that directory. Use only when explicitly invoked."
---
# Experimental Setup

## Purpose

Turn a research or experimentation idea into a concrete, reproducible experiment
that can be executed using the CURRENT repository.

The primary deliverable of this skill is:

> a complete shell command or shell block that the user can copy and paste into
> a terminal to run the experiment.

The experiment must have an explicit output root.

Examples include:

* benchmarking an embedding model
* comparing raw versus transformed embeddings
* retrieval experiments
* autoencoder representation experiments
* adapter experiments
* model architecture comparisons
* tracker comparisons
* segmentation experiments
* anomaly-detection experiments
* ablation studies
* hyperparameter experiments

The skill should use the repository as an experimental harness rather than
inventing one-off scripts when equivalent repository functionality already
exists.

---

# Invocation policy

This skill is intended for explicit invocation.

Do not use it automatically for ordinary questions.

Typical invocation:

```
$experimental-setup

Benchmark this GGUF embedding model against ArguAna, NFCorpus, and SciFact.
```

or:

```
$experimental-setup

Compare my object detector with and without its adapter.
```

---

# Repository is the source of truth

Before generating commands:

1. Read applicable `AGENTS.md` files.
2. Read `ARCHITECTURE.md` when present.
3. Inspect relevant README and mode documentation.
4. Inspect the CURRENT CLI implementation.
5. Inspect relevant mode/action argument definitions when necessary.
6. Inspect tests if command behavior or artifact contracts are unclear.

Do not rely on previous knowledge of the repository.

Do not invent:

* modes
* actions
* flags
* model identifiers
* registry entries
* artifact names
* dataset formats
* benchmark commands

Only generate commands supported by the CURRENT repository.

If a scientifically useful part of the requested experiment is not implemented,
separate the response into:

```
RUNNABLE NOW
```

and:

```
MISSING CAPABILITY
```

Never fabricate a command for functionality that does not exist.

---

# Primary output requirement

Every invocation MUST produce at least one complete copy-pasteable shell command
or shell block.

Planning or conceptual discussion alone is insufficient.

Whenever practical, generate ONE shell block that performs the complete
experiment.

Prefer:

```bash
set -euo pipefail

MODEL="..."
DATA="..."
OUT="..."

mkdir -p "$OUT"

# sanity checks

# experiment commands

# evaluation commands

# artifact listing
```

The user should be able to paste the block into a shell and execute it.

---

# Mandatory output directory

Every experiment MUST have an explicit experiment output root.

If the user supplies an output directory, use it.

Example:

```
Output:
~/Desktop/experiments/qwen3-retrieval
```

Use:

```bash
OUT="$HOME/Desktop/experiments/qwen3-retrieval"
```

If the user does not provide an output directory, choose a descriptive default
under:

```text
./experiments/<experiment-name>
```

Examples:

```text
./experiments/retrieval-baseline
./experiments/qwen3-retrieval
./experiments/autoencoder-retrieval
./experiments/yolo-adapter-ablation
./experiments/tracker-comparison
```

Do NOT ask a clarification question solely because an output path was omitted.

Choose a sensible descriptive default.

---

# Output flag handling

Do not assume that the repository has an `--output-dir` flag.

Inspect the CURRENT CLI.

The repository may use flags such as:

```text
--output
--output-file
--results-dir
--save-dir
--checkpoint-dir
```

Use whatever flags actually exist.

The experiment root should still be represented by:

```bash
OUT="..."
```

and individual repository commands should place their artifacts underneath it.

Example:

```bash
python -m mlx \
  --mode nlp \
  --action embed \
  ... \
  --output-file "$OUT/arguana/corpus-embeddings.csv"
```

Never invent an output flag for convenience.

---

# Experiment design

Determine:

* research question
* hypothesis
* baseline
* independent variable
* dependent metrics
* controlled variables
* datasets
* model(s)
* training steps
* evaluation steps
* expected artifacts

Prefer experiments that change one meaningful variable at a time.

A good experimental structure is:

```text
baseline
    vs
experimental variant
```

with everything else held constant where possible.

---

# Baseline requirement

Unless the user explicitly requests otherwise, establish a baseline.

Examples:

### Embeddings

```text
raw pretrained embeddings
    vs
transformed embeddings
```

### Adapter experiment

```text
base model
    vs
base model + adapter
```

### Tracking

```text
same detections + tracker A
    vs
same detections + tracker B
```

### Autoencoder

```text
raw embedding
    vs
autoencoder latent embedding
```

The baseline and experimental variants must write to separate artifact
directories.

---

# Controlled variables

Explicitly identify variables that must remain unchanged.

Examples include:

* dataset
* train/test split
* embedding model
* detector outputs
* seed
* training epochs
* optimizer
* batch size
* evaluation metrics
* retrieval K values
* image resolution

Do not vary unrelated settings between baseline and treatment unless the
experiment requires it.

---

# Experiment root structure

Use a clear artifact hierarchy.

Prefer structures such as:

```text
<OUT>/
├── manifest/
├── baseline/
├── variants/
└── results/
```

For multiple datasets:

```text
<OUT>/
├── manifest/
├── arguana/
├── nfcorpus/
├── scifact/
└── comparison/
```

For multiple models:

```text
<OUT>/
├── model-a/
├── model-b/
└── comparison/
```

Adapt the hierarchy to repository-native artifact conventions.

Do not move or rename artifacts produced by the repository unless there is a
good reason.

---

# Preserve intermediate artifacts

Do not unnecessarily regenerate expensive results.

Preserve reusable outputs such as:

* embeddings
* transformed embeddings
* vector databases
* checkpoints
* detector predictions
* normalized detections
* evaluation outputs
* metrics
* benchmark results

Example:

```text
GGUF model
    ↓
raw embeddings
    ├── baseline retrieval
    └── autoencoder training
            ↓
        latent embeddings
            ↓
        retrieval evaluation
```

Raw embeddings should normally remain reusable across several downstream
experiments.

---

# Reproducibility manifest

For substantial experiments, create a small manifest directory.

Prefer:

```bash
mkdir -p "$OUT/manifest"

git rev-parse HEAD > "$OUT/manifest/git-commit.txt"
python --version > "$OUT/manifest/python-version.txt"
```

When useful:

```bash
python -m mlx --help > "$OUT/manifest/mlx-help.txt"
```

Do not create unnecessarily huge environment dumps by default.

If dependency versions are critical, optionally use:

```bash
python -m pip freeze > "$OUT/manifest/python-packages.txt"
```

---

# Dataset inspection

If the user provides a dataset directory but the required internal structure is
not obvious, include lightweight inspection commands before expensive execution.

Examples:

```bash
find "$DATA" -maxdepth 2 -type f | sort | head -100
```

or:

```bash
tree -L 2 "$DATA"
```

Do not destructively modify datasets.

Do not assume the structure of ArguAna, NFCorpus, SciFact, COCO, MOT, or other
known datasets unless the repository's expected format has been verified.

---

# Sanity checks

Before expensive commands, include inexpensive checks where useful.

Examples:

```bash
test -f "$MODEL"
test -d "$DATA"
mkdir -p "$OUT"
```

Repository-native discovery commands should also be used where appropriate.

Examples might include:

```bash
python -m mlx --mode ... --action ls-models
```

but only if the current repository supports them.

---

# Shell safety

Generated shell blocks should normally begin with:

```bash
set -euo pipefail
```

This prevents later stages from running after a failed prerequisite.

Do not use destructive commands such as:

```bash
rm -rf
find ... -delete
```

unless explicitly requested.

Do not overwrite existing experimental results by default.

Prefer new output directories.

---

# Paths

Use shell variables to keep generated experiments readable.

Example:

```bash
MODEL="$HOME/models/model.gguf"
DATA="$HOME/Desktop/datasets/retrieval"
OUT="$HOME/Desktop/experiments/retrieval-baseline"
```

Quote path variables:

```bash
"$MODEL"
"$DATA"
"$OUT"
```

Prefer `$HOME` over literal `~` when assigning paths to shell variables.

Good:

```bash
DATA="$HOME/Desktop/datasets/retrieval"
```

Avoid:

```bash
DATA="~/Desktop/datasets/retrieval"
```

because tilde expansion does not occur inside quoted variable assignments.

---

# Repository location

Do not guess the absolute repository path.

If the user is expected to run the command from anywhere inside the repository,
prefer:

```bash
cd "$(git rev-parse --show-toplevel)"
```

If being in the repository root is already assumed, avoid unnecessary `cd`
commands.

Do not invent virtual-environment activation commands unless the repository
documents them or the user provides the environment path.

---

# Retrieval experiments

For retrieval experiments, identify the required benchmark concepts:

```text
corpus
queries
relevance judgments / qrels
```

A typical experiment may be:

```text
corpus
    ↓
corpus embeddings
    ↓
vector index

queries
    ↓
query embeddings
    ↓
retrieval

retrieved IDs + qrels
    ↓
metrics
```

Prefer preserving corpus embeddings so retrieval evaluation can be rerun without
recomputing them.

For datasets such as:

* ArguAna
* NFCorpus
* SciFact

inspect their actual local layout and the repository's expected dataset format
before generating commands.

---

# GGUF embedding experiments

When testing a GGUF embedding model:

1. verify the model file exists
2. verify the current repository supports the requested GGUF embedding workflow
3. generate baseline embeddings
4. preserve them under the experiment output root
5. evaluate them separately where supported

Use separate directories per dataset.

Example conceptual structure:

```text
$OUT/
├── arguana/
│   ├── embeddings/
│   └── benchmark/
├── nfcorpus/
│   ├── embeddings/
│   └── benchmark/
└── scifact/
    ├── embeddings/
    └── benchmark/
```

---

# Autoencoder embedding experiments

If the repository supports autoencoder-based embedding transformation, use:

```text
frozen source embeddings
        ↓
autoencoder training
        ↓
trained checkpoint
        ↓
latent representation
        ↓
retrieval benchmark
```

Compare at minimum:

```text
raw embeddings
    vs
autoencoder latent embeddings
```

Keep fixed:

* original embedding model
* dataset
* corpus/query split
* evaluation settings
* retrieval metric definitions

Do not rerun the embedding model for every autoencoder variant if the source
embeddings can be reused.

If autoencoder support is not currently implemented, do not invent commands.

Describe the missing capability separately.

---

# Adapter experiments

Prefer:

```text
same pretrained model
same dataset
same evaluation
same seed
same hyperparameters

adapter OFF
    vs
adapter ON
```

Keep checkpoints and results separate.

Do not overwrite the baseline model.

---

# Tracker experiments

Whenever possible, keep detector outputs fixed.

Prefer:

```text
fixed detections
    ├── tracker A
    └── tracker B
```

rather than independently rerunning detection for each tracker.

This isolates tracker choice as the independent variable.

Use repository-supported tracker metrics only.

---

# Ablation studies

For ablation experiments, vary one factor at a time unless the user asks for a
factorial design.

Example:

```text
latent_dim = 512
latent_dim = 256
latent_dim = 128
latent_dim = 64
```

Keep the other settings identical.

Each variant should have its own output directory.

Example:

```text
$OUT/
├── latent-512/
├── latent-256/
├── latent-128/
├── latent-64/
└── comparison/
```

---

# Evaluation metrics

Use metrics actually supported by the repository.

Possible metrics include:

### Retrieval

* Recall@K
* Precision@K
* MRR
* MAP
* nDCG@K

### Classification

* accuracy
* precision
* recall
* F1
* AUROC

### Object detection

* mAP
* precision
* recall

### Tracking

* HOTA
* MOTA
* IDF1

### Segmentation

* IoU
* Dice
* per-class metrics

Do not claim the repository produces a metric unless verified.

If the experiment needs a missing metric, classify it as a missing capability.

---

# Smoke tests

If the full experiment may be expensive and the CLI supports limiting work,
include a small smoke test first.

Examples may include:

* limited samples
* one epoch
* smaller model
* reduced dataset

Only use supported flags.

Do not invent `--limit`, `--dry-run`, or similar options.

---

# Commands before prose

The final response should prioritize runnable commands.

Preferred response order:

## Experiment

One short description.

## Command

A complete copy-pasteable shell block.

## Expected artifacts

A concise artifact tree or list.

## Experimental comparison

Brief explanation of what should be compared.

## Missing capabilities

Only if something required is not currently implemented.

Avoid long theoretical introductions before the commands.

---

# Complete shell block requirement

Whenever practical, produce ONE block similar to:

```bash
set -euo pipefail

MODEL="$HOME/models/model.gguf"
DATA="$HOME/Desktop/datasets/retrieval"
OUT="$HOME/Desktop/experiments/my-experiment"

cd "$(git rev-parse --show-toplevel)"

test -f "$MODEL"
test -d "$DATA"

mkdir -p \
  "$OUT/manifest" \
  "$OUT/baseline" \
  "$OUT/results"

git rev-parse HEAD > "$OUT/manifest/git-commit.txt"
python --version > "$OUT/manifest/python-version.txt"

# Actual repository commands go here.

find "$OUT" -maxdepth 4 -type f | sort
```

Replace comments with actual repository commands when supported.

Do not return placeholder commands when sufficient repository information exists
to construct real commands.

---

# Multi-stage commands

If the experiment requires multiple dependent stages, connect them in logical
order inside one shell block.

Example:

```text
prepare
    ↓
embed
    ↓
train
    ↓
transform
    ↓
benchmark
    ↓
list results
```

Each stage should write underneath `$OUT`.

Do not combine unrelated independent experiments into one giant command unless
the user requests it.

---

# Expected artifacts

After the shell block, state what the user should expect to find.

Prefer an artifact tree.

Example:

```text
experiment/
├── manifest/
│   ├── git-commit.txt
│   └── python-version.txt
├── baseline/
│   └── ...
├── variant/
│   └── ...
└── results/
    └── ...
```

Use actual repository artifact names where known.

Do not invent exact filenames when the application generates dynamic names.

---

# Missing capability handling

If part of the requested experiment cannot currently be run:

1. generate all commands that ARE currently supported
2. clearly identify where the workflow stops
3. name the smallest missing capability
4. explain what artifact should become the input/output boundary

Example:

```text
RUNNABLE NOW

GGUF model
    ↓
raw embedding CSV

MISSING CAPABILITY

raw embedding CSV
    ↓
autoencoder training
    ↓
latent embedding CSV
```

Do not modify source code.

Do not generate speculative CLI syntax.

---

# No repository modification

This skill designs experiments.

Do not change source code unless the user explicitly asks to implement a missing
capability.

If implementation is required, suggest using the repository's development or
architecture workflow separately.

Keep experimental design and framework modification separate.

---

# Final verification

Before returning the command, verify:

* all repository commands exist
* all flags exist
* mode/action combinations are valid
* paths are quoted correctly
* `$OUT` is defined
* all significant outputs live under `$OUT`
* commands are non-destructive
* baseline and variants do not overwrite each other
* the evaluation directly addresses the hypothesis
* unsupported functionality is clearly identified
* the whole shell block can reasonably be pasted into Bash

The final goal is:

> A researcher describes an experiment and receives a reproducible command that
> runs it and leaves behind a well-organized set of artifacts suitable for
> scientific comparison.

