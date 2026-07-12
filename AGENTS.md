# MLX Instructions

This project is a Python CLI for computer-vision workflows. New code should stay consistent with the current package layout in `mlx.core` and `mlx.modes.*`.

## Command Pattern

Business logic should be implemented with command-style classes.

Use this structure for non-trivial workflows:

```python
class CreateInference:
    def __init__(self, params1=None, params2=None):
        self.params1 = params1
        self.params2 = params2

    def execute(self):
        # Logic here
        pass
```

Requirements:

- Put orchestration and business logic inside a dedicated class with injected inputs from `__init__`.
- Use `execute()` as the default entrypoint for command-style workflow classes.
- Keep `runner.py` files thin. They should select the action, prepare config, and call a command or function.
- Use plain functions only for small, stateless helpers or simple transformations.
- If a workflow has multiple steps, split private helpers inside the command class instead of building one large `execute()` method.

## Exception Handling

Exception handling must be explicit and user-focused.

Requirements:

- Raise `MLXUserError` for invalid user input, unsupported actions, missing files, invalid model paths, bad dataset structure, or other recoverable user-facing failures.
- Raise `MLXAbort` only for intentional user cancellation flows.
- Do not swallow exceptions silently.
- Do not use broad `except Exception:` unless you immediately add context and re-raise or convert it into a clear project exception.
- Keep low-level library exceptions close to the integration boundary. Translate them into `MLXUserError` when the failure should be understandable from the CLI.
- Error messages must be actionable and specific. State what failed and what the user should inspect or provide.

Preferred pattern:

```python
from mlx.core.exceptions import MLXUserError


class ExportPredictions:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path

    def execute(self):
        if not self.model_path:
            raise MLXUserError("Missing model path for prediction export.")

        try:
            return self._run_export()
        except FileNotFoundError as exc:
            raise MLXUserError(f"Required file not found: {exc}") from exc
```

## Modularity

Keep modules focused and mode-specific.

Requirements:

- Shared concerns belong in `mlx.core`.
- Mode-specific logic belongs in the corresponding package under `mlx.modes.image_classification`, `mlx.modes.object_detection`, or `mlx.modes.segmentation`.
- Do not place training, inference, data preparation, presentation, and model definitions in one file.
- Prefer one responsibility per module:
  - `runner.py` for dispatch
  - `train.py` for training flows
  - `inference.py` for inference flows
  - `data.py` for dataset preparation and loading
  - `presentation.py` for CLI-facing summaries or formatted output
  - `models/` for model definitions and related building blocks
- Reuse shared helpers instead of copying logic across modes.
- Keep CLI parsing and UI output separate from ML workflow execution.
- Pass configuration into commands and functions explicitly. Do not rely on hidden global state.

## Design Rules

- Prefer composition over inheritance for workflow implementation.
- Keep public APIs small and predictable.
- Name commands using clear verb-first intent such as `TrainSegmentationModel`, `ConvertObjectDetectionModel`, or `RunCameraInference`.
- When a function or class exceeds a single responsibility, split it before adding more branching.
- Add short comments only when the control flow or ML-specific logic is not obvious from the code itself.
