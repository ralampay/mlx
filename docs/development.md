# Developing MLX

Use this guide for local setup, changing CLI behavior, and calling workflows from Python.
[ARCHITECTURE.md](../ARCHITECTURE.md) defines ownership and dependency direction;
[AGENTS.md](../AGENTS.md) defines the required implementation and review workflow.
For a new model, tracker, loss, or metric, start with the
[extension tutorials](tutorials/README.md).

## Set up and validate a checkout

Create and activate a virtual environment using the [installation guide](../README.md#installation).
From the repository root, install the editable project and test runner:

```bash
python -m pip install -e .
python -m pip install pytest
python -m mlx --help
python -m pytest -q tests/test_cli_modularity_refinement.py tests/test_runner_configuration.py tests/test_modularity_boundaries.py
```

These focused checks use synthetic inputs and cover configuration, lazy imports, registry
isolation, and workflow collaborators. For broader provider/notebook development, install
`requirements.txt` and any additional workflow extras listed in the installation guide.
Run the full suite before handing off a code change:

```bash
python -m pytest -q
python -m compileall -q mlx
git diff --check
```

Inspect skip reasons with `python -m pytest -q -rs`. A skipped optional integration is not
evidence that its provider works in your environment. Dataset-specific experiments and cloud
jobs are separate manual validation; unit tests should not require their data or credentials.

## Make a change at the owning boundary

Follow `INSPECT -> PLAN -> IMPLEMENT -> VALIDATE -> REVIEW -> REFINE -> REVALIDATE`:

1. Read the relevant architecture section, runner, request types, implementations, and tests.
2. Identify the existing extension point. Put mode-specific behavior under its mode; use
   `mlx.core` only for behavior genuinely shared across modes.
3. Keep dispatch and dependency construction in the runner. Put use-case ordering in a command
   with injected inputs and an `execute()` entrypoint. Return results; attach presentation at
   the composition boundary.
4. Test the public command with small synthetic data and injected collaborators. Cover
   important validation failures, resource cleanup, and compatibility behavior.
5. Review correctness, architecture, portability, and robustness after tests pass. Refine and
   rerun affected checks, inspect the final diff, and update documentation in the same change.

Do not move research feature adapters into LibreYOLO: their implementation and registry are
MLX-owned. See [adapter ownership](object_detection/adapter-ownership.md).

## Add a CLI option or action

The CLI keeps its flat `--mode ... --action ...` syntax. Option declarations belong in the
owning mode's `cli_options.py`; shared flags belong in `mlx/cli_options.py`. For example,
a mode-owned inventory entry can use:

```python
from mlx.core.cli_options import CliOption

sample_limit_option = CliOption(
    ("--sample-limit",),
    {"type": int, "default": None, "help": "Maximum number of samples to process."},
)
```

Add the declaration to that module's `OPTIONS` tuple. If the mode does not yet have an option
inventory, set its `ModeDescriptor.options` reference in `mlx/cli_routing.py`. Keep metadata
modules free of workflow imports. Parser construction imports all registered option inventories;
heavy package initialization would therefore affect every command, including help.

For a new action, update the descriptor's `actions`, add runner dispatch, and construct the
appropriate typed request and command. Put semantic validation in the command or integration
boundary. Descriptor `option_validator`/`validation_options` hooks support checks for opt-in
flags that must reject use in other modes, as with detection distillation.

Help is rendered from parser metadata; do not maintain a second table of flags or defaults.
The displayed defaults are parser defaults, and a mode can supply different effective defaults
for omitted options. Use `with_explicit_options` when normalizing runner inputs so explicitly
supplied Python values and CLI values are preserved. Image-size normalization is shared by
classification and segmentation; its precedence is documented in the architecture guide.

After adding an option, test parsing, typed-request propagation, explicit/default handling, and
its effect through the command. An alias added to an existing model registry does not normally
need a new CLI action or parser change.

## Call a workflow from Python

Prefer typed requests and public command classes over building an argparse namespace. This
example constructs a classifier-training command without accessing data or starting training:

```python
from mlx.modes.image_classification.requests import TrainImageClassificationRequest
from mlx.modes.image_classification.train import TrainImageClassificationModel

request = TrainImageClassificationRequest(
    model="resnet18",
    dataset_path="./dataset",
    output_path="./artifacts/resnet18",
    input_size=(224, 224), width=224, height=224,
    epochs=1, batch_size=2, workers=0, device="cpu",
)
command = TrainImageClassificationModel(request)
```

After preparing the [classification dataset](image_classification/README.md#dataset-expectations),
call `command.execute()`. Zero workers keeps loading in the current process. Direct commands
use null reporters by default; inject a reporter if progress is needed. Set runtime choices
explicitly rather than assuming the CLI's default-resolution step runs for Python commands.

Classification training accepts dataset loaders, an optimizer factory, and artifact storage;
benchmarking accepts checkpoint/data loaders and an artifact writer. Both standard and one-shot
paths support these collaborators. See the canonical
[classification workflow contracts](../ARCHITECTURE.md#classification-workflow-boundaries) and
[synthetic contract tests](../tests/test_cli_modularity_refinement.py) for their signatures and
examples that do not read real images or write artifacts.

## Keep provider selection local

Detection registries are immutable: registration returns a new registry. For ordinary local
runner actions, pass it explicitly. This example lists metadata without constructing a detector:

```python
from mlx.modes.object_detection.providers import BUILTIN_PROVIDER_REGISTRY
from mlx.modes.object_detection.runner import run_object_detection

registry = BUILTIN_PROVIDER_REGISTRY.register(
    "local-yolo", "mlx.modes.object_detection.ultralytics.provider:get_provider",
)
models = run_object_detection(
    {"action": "ls-models", "names_only": True, "provider": "local-yolo", "output_format": "json"},
    provider_registry=registry,
)
assert models
assert "local-yolo" not in BUILTIN_PROVIDER_REGISTRY.entries
```

Commands also accept provider instances directly. An injected ordinary-provider registry does
not configure AWS or research-adapter workflows; those reject the override and use their own
composition boundaries. Legacy `register_provider()` calls without a registry still mutate
process-wide defaults. Avoid them in new integrations. An in-process alias is not automatically
available to another `mlx` process.
