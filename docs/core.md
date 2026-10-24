# Shared core infrastructure

Package: `mlx.core`. This package supports workflows; it is not a selectable CLI mode.

[All module guides](README.md) · [Architecture](../ARCHITECTURE.md)

## Contents

- [Available capabilities](#available-capabilities)
- [Commands and reporting](#commands-and-reporting)
- [Requests and errors](#requests-and-errors)
- [Choosing a shared component](#choosing-a-shared-component)
- [Validation](#validation)

## Available capabilities

| Capability | Modules | Typical caller |
| --- | --- | --- |
| Command and progress contracts | `commands` | Commands and presentation adapters |
| Typed/legacy request bridge | `requests` | Mode-owned request dataclasses |
| Configuration and import references | `configuration`, `extensions`, `cli_options` | Runners, factories, and registries |
| Artifact serialization and source snapshots | `artifacts`, `repository_snapshot` | Checkpoint and experiment boundaries |
| Dataset-source staging | `datasets` | Local/S3 training composition |
| Video frame ports and display adapters | `streaming`, `opencv_display` | Detection, segmentation, and tracking |
| Model inventory and presentation | `model_listing`, `presentation`, `ui` | Discovery and CLI presentation |
| Reproducibility | `random`, `partitions` | Training, data splitting, and resume |
| Shared evaluation/statistics | `binary_metrics`, `paired_statistics`, `paired_superiority` | Mode-owned evaluators and reports |
| Feature, vector, and loss contracts | `image_backbones`, `vector_transforms`, `losses`, `deep_svdd` | Model integration and training components |
| SageMaker lifecycle infrastructure | `aws` | Mode-owned AWS services |

Use the concrete component needed by the workflow, rather than importing all of core.
Optional provider integrations should be constructed at their documented boundary.
[Dataset staging](s3-dataset-training.md) documents cache and archive behavior; mode guides
specify the actual dataset formats.

## Commands and reporting

`Command` is a protocol with one `execute()` entrypoint; implementations do not need to inherit
from it. Commands receive collaborators and runtime settings explicitly. They return results
and emit progress through `WorkflowReporter`, leaving terminal rendering to presentation code.

Capture events without a terminal or heavyweight dependencies:

```python
from mlx.core.commands import CallbackWorkflowReporter, emit

events = []
reporter = CallbackWorkflowReporter(events.append)
emit(reporter, "info", "Prepared inputs", current=1, total=1, payload={"rows": 2})
assert events[0].payload == {"rows": 2}
```

Use `NullWorkflowReporter` when progress is unnecessary. Use mode-specific presentation
adapters when running interactively; reusable commands should not print directly.

## Requests and errors

Mode request dataclasses can inherit `ConfigRequest`. `from_config()` maps declared fields
and retains unknown non-internal values in `extras`; `to_config()` reconstructs the compatibility
mapping. This bridge does not perform semantic validation or run CLI default resolution.

```python
from dataclasses import dataclass
from mlx.core.requests import ConfigRequest

@dataclass(frozen=True)
class ExampleRequest(ConfigRequest):
    workers: int = 0

request = ExampleRequest.from_config({"workers": 2, "note": "example", "_explicit_options": {"workers"}})
assert request.workers == 2
assert request.to_config() == {"workers": 2, "note": "example"}
```

Raise `MLXUserError` from `mlx.core.exceptions` for recoverable user-facing failures.
Use `MLXAbort` for intentional cancellation. Translate library errors at the boundary that has
enough context to give an actionable message, preserving the original exception with
`raise ... from exc`. Avoid wrapping the same failure at every layer.

## Choosing a shared component

Core is appropriate when multiple modes need the same capability and semantics. A new dataset
format, detector attachment policy, checkpoint schema, or mode-specific objective belongs in
its mode. Do not add a second registry or broad service abstraction merely to shorten imports.

For instance, frame reading and S3 extraction are shared infrastructure; segmentation mask
interpretation and detector checkpoint reconstruction are mode-specific. The authoritative
rules are [Shared Primitives, Requests, and Registries](../ARCHITECTURE.md#shared-primitives-requests-and-registries).

## Validation

Use synthetic inputs, temporary directories, fake providers, and injected reporters. Relevant
checks include:

```bash
python -m pytest -q tests/test_core_artifacts.py tests/test_core_presentation.py tests/test_modularity_boundaries.py
```

For implementation review and the full-suite handoff requirement, see the
[development guide](development.md).
