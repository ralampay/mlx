# MLX Instructions

This project is a Python CLI for computer-vision workflows. New code must stay consistent with the current package layout in `mlx.core` and `mlx.modes.*`, preserve existing public behavior unless the task explicitly changes it, and favor implementations that are modular, portable, testable, and reusable across workflows.

## Development Workflow and Refinement Loop

Non-trivial implementation work must not stop at the first working solution. Treat the first implementation as a draft that must be validated and refined.

Use this loop for every meaningful feature, refactor, integration, or bug fix:

`INSPECT -> PLAN -> IMPLEMENT -> VALIDATE -> REVIEW -> REFINE -> REVALIDATE`

Requirements:

- Inspect the relevant code, tests, configuration, `ARCHITECTURE.md`, and nearby implementations before changing code.
- Identify existing patterns and extension points before introducing new abstractions.
- Form a short implementation plan before editing files.
- Implement the smallest coherent solution that fully satisfies the requirement.
- Run the relevant tests, linters, type checks, build checks, smoke tests, or CLI commands available in the repository.
- Review the implementation after validation rather than treating a passing test as completion.
- Refine the implementation when the review exposes architectural, robustness, portability, duplication, naming, testing, or maintainability issues.
- Re-run validation after every meaningful refinement.
- Inspect the final diff before completion.

For non-trivial changes, perform at least these distinct review passes after the initial implementation:

### Correctness Review

Verify that:

- Every requested behavior is implemented.
- Inputs, outputs, failure cases, and edge conditions are handled intentionally.
- The implementation does not depend on accidental behavior or hidden state.
- Existing workflows still behave correctly unless the task explicitly changes them.

### Architecture and Convention Review

Verify that:

- Command-pattern conventions are followed.
- Responsibilities are placed in the correct package and module.
- Dependency direction matches `ARCHITECTURE.md`.
- The implementation reuses existing abstractions where appropriate instead of creating parallel mechanisms.
- CLI, presentation, orchestration, domain logic, and provider/library integration remain separated.

### Portability and Modularity Review

Verify that:

- Core workflow logic is not unnecessarily tied to a specific CLI command, model family, provider, framework, device, file layout, or runtime.
- Replaceable dependencies are injected or isolated behind narrow interfaces.
- Mode-specific behavior remains mode-specific while reusable behavior is promoted to an appropriate shared abstraction.
- New code can be tested independently of heavyweight external systems where practical.
- No hidden globals, singleton state, environment-specific paths, or hard-coded provider assumptions have been introduced.

### Robustness and Test Review

Verify that:

- Errors are translated at the appropriate boundary.
- New behavior has appropriate tests or smoke coverage.
- Existing relevant tests pass.
- Failure paths are exercised where practical.
- Temporary code, debug output, dead branches, commented-out code, placeholders, and unnecessary TODOs are removed.

Passing tests alone is not sufficient to declare a non-trivial task complete. Passing tests begin the review phase; they do not end the task.

If a review pass reveals a material issue, fix it and repeat the relevant validation and review steps until no material issues remain.

## Command Pattern

Business logic and workflow orchestration should be implemented with command-style classes.

Use this structure for non-trivial workflows:

```python
class CreateInference:
    def __init__(self, params1=None, params2=None):
        self.params1 = params1
        self.params2 = params2

    def execute(self):
        # Coordinate the workflow here.
        pass
```

Requirements:

- Put orchestration and use-case-level business logic inside a dedicated command class with required collaborators and inputs injected through `__init__`.
- Use `execute()` as the single default public entrypoint for command-style workflow classes unless an existing project convention explicitly requires otherwise.
- Commands should represent an intent or use case, not a generic utility container.
- Name commands with clear verb-first intent such as `TrainSegmentationModel`, `ConvertObjectDetectionModel`, or `RunCameraInference`.
- Keep `runner.py` files thin. They may parse or normalize already-parsed options, select an action, construct dependencies/configuration, invoke a command, and pass results to presentation code.
- Do not put training loops, inference logic, data transformation logic, model-specific behavior, provider calls, or substantial branching directly in `runner.py`.
- Use plain functions only for small stateless helpers, pure transformations, or narrowly scoped utilities.
- If a workflow contains multiple logical steps, split them into focused private methods or collaborating components rather than building one large `execute()` method.
- A command may coordinate collaborators, but it should not absorb every implementation detail belonging to repositories, adapters, model wrappers, loaders, serializers, evaluators, or presenters.
- Prefer dependency injection over constructing replaceable infrastructure deep inside `execute()`.
- Do not read CLI arguments, environment variables, or global configuration directly from deep workflow code when those values can be passed explicitly.
- Do not print directly from reusable command logic. Return structured results and let presentation or CLI layers render them.
- Commands should be callable from tests, another Python module, or a future interface without requiring CLI execution.

### Command Boundary Rules

A command is responsible for coordinating a use case. It may:

- Validate use-case-level preconditions.
- Coordinate several focused components.
- Determine workflow order.
- Translate lower-level failures when the command is the correct user-facing boundary.
- Return a result object, value, or documented data structure.

A command should not:

- Parse `argparse` or Typer/Click state directly.
- Contain CLI-specific formatting.
- Depend on terminal state.
- Hard-code model-provider selection that belongs in a registry, factory, adapter, or injected collaborator.
- Mix unrelated workflows because they happen to share a CLI action.
- Become a generic god object with many public methods.

When multiple commands share substantial behavior, extract the shared capability into a focused service, helper, adapter, protocol, or core component rather than creating command inheritance solely to reuse implementation.

## Portability and Dependency Isolation

New implementations should be portable across environments and replaceable where practical.

Requirements:

- Keep project-owned workflow logic independent from third-party library details whenever a narrow integration boundary is practical.
- Isolate framework/provider-specific code near its integration boundary.
- Prefer adapters, wrappers, factories, registries, protocols, or injected callables for dependencies that may reasonably vary, such as model backends, trackers, exporters, storage providers, device selection, or inference engines.
- Do not spread provider-specific conditionals throughout commands or core workflow code.
- Centralize backend/provider selection in a deliberate construction layer when multiple implementations are supported.
- Avoid absolute paths and machine-specific assumptions.
- Accept paths, devices, provider names, thresholds, and other runtime choices through explicit configuration or injected parameters.
- Do not require a GPU unless the feature inherently requires one. Respect existing device-selection conventions.
- Do not couple reusable logic to a particular dataset directory layout when a loader or dataset abstraction can represent that boundary.
- Avoid import-time side effects, expensive model loading, global mutable registries, or network calls.
- Heavy resources should be created deliberately and as late as practical.
- Keep serialization formats and external schemas isolated from internal domain/workflow structures when translation is useful.

When adding a new provider or implementation, prefer code shaped like:

```text
CLI / caller
    -> runner or composition root
        -> command
            -> project-owned interface / focused service
                -> provider-specific adapter
```

rather than:

```text
command
    -> scattered provider checks
    -> direct CLI reads
    -> framework-specific calls throughout the workflow
```

## Modularity

Keep modules focused and mode-specific.

Requirements:

- Shared concerns belong in `mlx.core` only when they are genuinely reusable across modes or represent cross-cutting infrastructure.
- Mode-specific logic belongs in the corresponding package under `mlx.modes.image_classification`, `mlx.modes.object_detection`, or `mlx.modes.segmentation`.
- Do not move code into `mlx.core` merely to make imports convenient.
- Do not place training, inference, data preparation, presentation, and model definitions in one file.
- Prefer one responsibility per module:
  - `runner.py` for dispatch and dependency construction
  - `train.py` for training commands/flows
  - `inference.py` for inference commands/flows
  - `data.py` for dataset preparation and loading
  - `presentation.py` for CLI-facing summaries or formatted output
  - `models/` for model definitions and related building blocks
  - provider/adapter modules for third-party integration when separation is warranted
- Reuse shared helpers instead of copying logic across modes.
- Before extracting shared code, confirm the behavior is conceptually the same rather than merely similar-looking.
- Keep CLI parsing and UI output separate from ML workflow execution.
- Pass configuration into commands and functions explicitly. Do not rely on hidden global state.
- Prefer small composable units over deeply nested inheritance hierarchies.
- Keep cross-module dependencies directional and intentional.
- Avoid circular imports. If two modules need each other's internals, reconsider responsibility boundaries.
- Keep public module surfaces narrow; implementation details should remain private where practical.

## Extension and Reuse Conventions

When adding a new model, tracker, data source, exporter, trainer, or inference backend:

- First inspect whether the project already has a registry, factory, adapter, protocol, base abstraction, or configuration convention for that category.
- Extend the existing mechanism instead of adding a second dispatch system.
- Keep selection logic out of workflow internals.
- Prefer a stable project-owned interface around third-party APIs when the external API would otherwise leak broadly through the codebase.
- Ensure a new implementation can be added with localized changes whenever reasonably possible.
- Do not require callers to know unnecessary third-party implementation details.
- Preserve backward compatibility for existing configuration and commands unless the requested change explicitly allows a breaking change.

A good extension should usually require changes near the extension point, not edits scattered across unrelated workflows.

## Exception Handling

Exception handling must be explicit and user-focused.

Requirements:

- Raise `MLXUserError` for invalid user input, unsupported actions, missing files, invalid model paths, bad dataset structure, or other recoverable user-facing failures.
- Raise `MLXAbort` only for intentional user cancellation flows.
- Do not swallow exceptions silently.
- Do not use broad `except Exception:` unless you immediately add context and re-raise or convert it into a clear project exception.
- Keep low-level library exceptions close to the integration boundary. Translate them into `MLXUserError` when the failure should be understandable from the CLI.
- Do not translate exceptions repeatedly at every layer. Convert them at the boundary that has enough context to produce a useful message.
- Preserve the original exception with `raise ... from exc` when wrapping failures.
- Error messages must be actionable and specific. State what failed and what the user should inspect or provide.
- Reusable low-level components should not print errors directly.

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

## Configuration and State

- Prefer explicit configuration objects or explicit constructor parameters over implicit global state.
- Keep configuration normalization near the composition/runner boundary.
- Commands should receive values in forms appropriate for the workflow rather than repeatedly decoding raw CLI strings.
- Environment-variable access should be centralized where practical and passed inward as configuration.
- Avoid mutating shared configuration objects during execution unless mutation is explicitly part of their contract.
- Avoid module-level mutable state.

## Testing Expectations

- Add or update tests for material behavior changes when the repository has an applicable testing structure.
- Prefer testing commands through their public `execute()` entrypoint.
- Inject lightweight fakes/stubs for expensive or external collaborators where practical.
- Keep provider-specific tests separate from project-owned workflow tests when that distinction improves clarity.
- Test user-facing failure behavior for important validation paths.
- For integrations that cannot be fully unit tested, add a focused smoke test or document/run a reproducible validation command.
- A bug fix should include a regression test when practical.
- Do not weaken or delete a valid test merely to make an implementation pass.

## Design Rules

- Prefer composition over inheritance for workflow implementation.
- Use inheritance only when there is a genuine substitutable type relationship and it simplifies rather than hides behavior.
- Keep public APIs small and predictable.
- Name commands using clear verb-first intent.
- Name services/components by responsibility rather than vague names such as `Manager`, `Helper`, or `Utils` when a more specific name is available.
- When a function or class exceeds a single responsibility, split it before adding more branching.
- Prefer explicit data flow over hidden callbacks or shared mutable state.
- Prefer deterministic, side-effect-light components where practical.
- Add short comments only when the control flow or ML-specific logic is not obvious from the code itself.
- Do not add abstractions speculatively. Introduce them when they clarify an existing boundary, enable a required extension, or remove real duplication/coupling.
- Avoid broad refactors unrelated to the requested task, but make localized refactors when needed to preserve architecture and maintainability.

## Completion Criteria

Before declaring a non-trivial task complete:

- Re-read the original request and verify each requirement individually.
- Confirm the command-pattern boundary is appropriate.
- Confirm reusable logic is not unnecessarily coupled to CLI or third-party implementation details.
- Confirm module placement and dependency direction are correct.
- Run all relevant validation available for the changed area.
- Inspect the final diff for accidental changes.
- Remove debug code, temporary files, dead imports, placeholders, and stale comments.
- Check for duplicated logic introduced by the change.
- Check whether newly introduced interfaces or behavior require architecture documentation updates.

Ask during final review:

- Does this solve the complete requested use case rather than only the happy path?
- Would this command work if invoked from Python instead of the CLI?
- Could a relevant backend/provider be replaced without rewriting the workflow?
- Are responsibilities located in the correct modules?
- Is any new coupling avoidable?
- Are failures actionable to the caller/user?
- Is the implementation simpler or more maintainable after refinement than after the first pass?
- Would this diff be acceptable in a production code review?

If any answer exposes a material issue, continue the refinement loop before finishing.

## Architecture Documentation

- Treat `ARCHITECTURE.md` as the canonical architecture reference.
- Review `ARCHITECTURE.md` for every code or configuration update.
- Update it in the same change whenever package ownership, dependency direction, command inventory,
  public interfaces, providers, presentation boundaries, error behavior, or workflow data flow changes.
- Document new extension points, adapters, registries, factories, or command boundaries when they materially affect how future code should be added.
- Keep architecture detail in `ARCHITECTURE.md`; README files should link to it and focus on setup and usage.
