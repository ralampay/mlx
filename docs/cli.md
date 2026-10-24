# CLI and routing

Modules: `mlx.cli`, `mlx.cli_config`, `mlx.cli_options`, and `mlx.cli_routing`.

[All module guides](README.md) · [Architecture](../ARCHITECTURE.md)

## Contents

- [Invocation and discovery](#invocation-and-discovery)
- [Configuration and defaults](#configuration-and-defaults)
- [Results and exit codes](#results-and-exit-codes)
- [Calling from Python](#calling-from-python)
- [Extending the CLI](#extending-the-cli)

## Invocation and discovery

Install the local project using the [installation guide](../README.md#installation).
The installed `mlx` command and `python -m mlx` use the same entrypoint:

```bash
python -m mlx --help
python -m mlx --doctor
python -m mlx --mode image-classification --action ls-models --names-only --format json
python -m mlx --mode text-embedding --action ls-vector-stores
```

`--doctor` runs without `--mode` and reports environment requirements. Ordinary local
workflows require no environment variables. The report lists optional AWS and SAR variables
with set/unset status, never their values; AWS profiles and workload roles can supply
credentials instead. It does not contact AWS or validate credentials. Missing optional
variables do not cause a failure. Use `--doctor --format json` for structured output.

`--mode` selects a workflow; `--action` selects its operation. Modes have different defaults,
including training, smoke testing, and listing. Always specify the action in scripts.
See the [mode directory](modes.md) to choose a workflow. Hyphenated and underscored aliases
are accepted where registered; `track`/`tracking` and `text-embedding`/`nlp` are examples.

`ls-models --names-only` requests metadata where supported. Detailed model listings can
construct models to count parameters. Inventory actions such as `ls-losses`, `ls-metrics`,
and `ls-trackers` are mode-specific; consult `--help` and the selected guide.

## Configuration and defaults

Help shows the shared parser's defaults. A mode can resolve an omitted option to its own
default: detection uses 640 × 640 inputs, while autoencoder training uses batch size 64.
Explicit values are tracked so a caller's choices survive normalization. For reproducibility,
specify the model, dimensions, device, seed, workers, and relevant training settings explicitly.

The meanings of some flags depend on the mode. For example, `--adapter` selects a detector
adapter for LibreYOLO inference or an autoencoder representation transform for text embedding.
`--config` is used by AWS workflows; it is not a universal local configuration-file loader.

The CLI selects local execution by default. `--platform aws` is available only for documented
AWS actions and requires their configuration and credentials. Refer to the mode's AWS guide.

## Results and exit codes

For local `--format json` execution, the top-level CLI serializes the returned result once to
standard output and routes other stdout output to standard error during workflow execution.
Progress/presentation adapters are suppressed by the local runners. The result shape remains
specific to the action; some successful commands return `null` rather than an artifact summary.

| Exit code | Meaning |
| --- | --- |
| `0` | Success, including help |
| `1` | Handled workflow error, unknown mode, or intentional cancellation |
| `2` | Invalid arguments or missing mode |

Handled workflow errors in JSON mode are emitted to standard error as `{"error": "..."}`.
Argument-parsing errors use the usage display. Unexpected exceptions can still propagate.
AWS status/watch presentation has its own output behavior; do not assume it follows the local
single-result JSON contract.

## Calling from Python

Inspect routing metadata without starting a workflow:

```python
from mlx.cli_routing import resolve_mode_descriptor

mode = resolve_mode_descriptor("image-classification")
assert mode.name == "image_classification"
assert "train" in mode.actions
```

For programmatic work, prefer the mode's typed request and command `execute()` API. Use a runner
when you specifically need its configuration normalization and presentation wiring. Neither
requires manufacturing an argparse namespace. See [Python composition](development.md#call-a-workflow-from-python).

## Extending the CLI

Add option metadata to the owning mode's `cli_options.py` inventory, or to `mlx.cli_options`
for genuinely shared flags. Parsing and help consume the same declarations. Add new actions
to the existing descriptor and runner, and keep workflow logic in commands.

The [development guide](development.md#add-a-cli-option-or-action) provides the implementation
steps. [ARCHITECTURE.md](../ARCHITECTURE.md#layers-and-dependency-direction) owns the detailed
routing, validation, and dependency rules.
