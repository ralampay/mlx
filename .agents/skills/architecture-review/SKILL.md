---
name: architecture-review
description: >
  Perform an explicit, conservative architecture and refactoring review of the
  current repository. Use only when explicitly invoked. Inspect repository-local
  AGENTS.md and architecture documentation first, identify concrete architectural
  problems, refactor only where justified, validate changes, and prefer
  simplification over speculative abstraction.
---

# Architecture Review and Refactor

## Purpose

Perform a deliberate architecture review of the CURRENT repository and refactor
the code only when concrete improvements are justified.

The purpose of this skill is not to redesign the repository.

The goals are:

- preserve correctness
- reduce unnecessary coupling
- reduce cognitive load
- improve locality of change
- improve extension ergonomics
- preserve useful abstractions
- remove accidental complexity
- keep dependency direction explicit
- protect portability
- keep documentation synchronized with the implementation

Optimize for maintainability and experimentation rather than maximum abstraction.

---

# Repository authority

Before performing any architectural analysis:

1. Find and read the applicable `AGENTS.md` files.
2. Find and read architecture documentation such as `ARCHITECTURE.md`.
3. Read the relevant README and developer documentation.
4. Inspect the CURRENT implementation.

Repository-local instructions override this skill.

Do not rely on assumptions from previous versions of the repository.

If the repository defines its own:

- architectural principles
- dependency rules
- extension mechanisms
- testing conventions
- coding standards
- workflow requirements

follow those rules.

This skill supplies the review procedure, not the repository architecture.

---

# Core question

Throughout the review, ask:

> Does the current architecture make the intended development and
> experimentation workflows easier to understand, extend, test, and maintain?

For extensible systems, also ask:

> If someone adds another implementation of this concept tomorrow, how much of
> the repository must they understand and modify?

Prefer localized extension paths such as:

    implement
        -> register
        -> configure
        -> run
        -> test

when that pattern makes sense for the repository.

Do not force this pattern onto components that are not genuine extension points.

---

# Operating mode

This is a REVIEW-AND-REFACTOR skill.

Follow this workflow:

    INSPECT
        -> ANALYZE
        -> PLAN
        -> IMPLEMENT
        -> VALIDATE
        -> REVIEW
        -> REFINE
        -> REVALIDATE

Do not begin refactoring before understanding the relevant architecture.

Do not stop immediately after tests pass.

Review the resulting design and diff separately after implementation.

---

# 1. Inspect

Start by understanding the repository.

Inspect relevant:

- package/module structure
- entry points
- public APIs
- commands
- runners
- composition roots
- request/configuration objects
- models
- registries
- factories/resolvers
- adapters
- providers
- protocols/interfaces
- persistence/artifact boundaries
- presentation code
- tests
- documentation
- compatibility layers

Trace representative workflows end-to-end.

For example:

    user/API/CLI input
        -> configuration
        -> composition root
        -> command/use case
        -> domain/model/component
        -> integration/provider
        -> structured result
        -> presentation/artifact

Adapt this model to the repository rather than assuming every repository uses
these exact layers.

---

# 2. Analyze architectural boundaries

Review actual dependency direction.

Look for problems such as:

- lower-level modules importing orchestration code
- generic modules importing application-specific modules
- domain logic depending on presentation
- provider-specific objects leaking through neutral APIs
- CLI concerns appearing deep inside reusable code
- configuration parsing occurring throughout the system
- circular dependencies
- hidden imports used to avoid dependency cycles
- unrelated modules depending directly on implementation internals

Prefer explicit dependency flow.

Do not move code merely to make directories look symmetrical.

Ownership should follow responsibility.

---

# 3. Review extension locality

For each meaningful extension point, determine exactly what must change to add a
new implementation.

Examples may include:

- model
- architecture
- tracker
- loss
- metric
- adapter
- provider
- backend
- dataset
- transform
- exporter
- storage implementation
- algorithm

For each applicable category identify:

- implementation location
- contract/interface
- registration mechanism
- construction mechanism
- configuration mechanism
- discovery mechanism
- tests
- unrelated files that currently require modification

Prefer changes that are local to the extension.

Flag extension scatter such as requiring changes to:

- CLI dispatch
- several runners
- unrelated commands
- multiple factories
- unrelated mode packages
- presentation code

when those edits are not inherently required.

Do not introduce a registry just because a concept has two implementations.

Use extension infrastructure only where extensibility is intentional and useful.

---

# 4. Review registries and factories

Treat registries and factories as separate concepts.

A registry normally answers:

> Which implementation corresponds to this identifier?

A factory normally answers:

> How should this implementation be constructed?

Review for:

- mutable process-wide registries
- import-time side effects
- unnecessary decorators
- duplicated alias handling
- repeated normalization
- heavyweight imports during discovery
- factories containing workflow logic
- giant conditional dispatch blocks
- generic registries that erase useful domain-specific metadata

Prefer explicit and testable mechanisms.

Do not replace several understandable domain-specific registries with one
universal plugin system unless there is strong evidence that doing so simplifies
the repository.

---

# 5. Review abstraction cost

Treat every abstraction as something that must justify its existence.

For significant abstractions ask:

1. What concrete problem does this solve?
2. Does it reduce coupling?
3. Does it improve replaceability?
4. Does it improve testing?
5. Does it reduce repeated logic?
6. Does it isolate an external dependency?
7. Does it reduce the number of files needed for common changes?
8. Is its public API substantially simpler than the implementation it hides?
9. Could this abstraction be removed while retaining the same capabilities?
10. Is it solving a current problem or a hypothetical future problem?

Prefer deletion and simplification where appropriate.

Do not preserve abstraction merely because it was introduced in a previous
refactor.

---

# 6. Favor composition

Prefer small composable collaborators over deep inheritance hierarchies.

Use inheritance when there is a meaningful substitutable relationship.

Otherwise prefer:

- composition
- dependency injection
- narrow protocols/interfaces
- simple callables
- explicit collaborators

Avoid giant base classes containing optional behavior for unrelated
implementations.

Avoid interfaces containing methods that most implementations do not need.

---

# 7. Review configuration flow

Trace important configuration values from their origin to their eventual use.

Look for:

- repeated parsing
- repeated normalization
- different defaults at different layers
- string values repeatedly converted to enums or objects
- environment variables read deep inside workflows
- implementation-specific details exposed unnecessarily to users
- large configuration objects passed everywhere
- mutable configuration objects

Prefer normalization near composition boundaries.

Internal code should receive values in the form most meaningful to that layer.

---

# 8. Review commands and orchestration

Commands/use cases should coordinate meaningful work.

Composition roots should construct dependencies.

Look for commands that have accumulated:

- implementation selection
- architecture construction
- provider details
- rendering
- raw serialization
- dataset parsing
- unrelated training policy
- giant conditional trees

Also look for the opposite problem:

- tiny wrapper classes
- one-line services
- excessive delegation
- layers that provide no useful boundary

Do not extract code merely to shorten functions.

Extract coherent responsibilities.

---

# 9. Review integrations and optional dependencies

Keep third-party integrations at deliberate boundaries.

Look for unnecessary coupling to:

- ML frameworks
- inference runtimes
- databases
- cloud SDKs
- visualization packages
- CLI libraries
- network services

Where dependencies are optional, avoid importing or initializing them merely
because the package, CLI, registry, or discovery mechanism is loaded.

Prefer lazy integration boundaries.

Do not wrap stable third-party APIs method-for-method unless the wrapper provides
actual architectural isolation.

---

# 10. Review public API ergonomics

Consider the repository from the perspective of a developer using it as a
library.

Ask:

- Is the intended entry point discoverable?
- Can useful functionality be called without going through the CLI?
- Can components be replaced during tests or experiments?
- Are callers forced to understand provider implementation details?
- Are return values structured?
- Do callers need to parse console output?
- Are constructor dependencies understandable?
- Is configuration meaningful at the appropriate abstraction level?

Refactor only where the improvement is concrete.

---

# 11. Review results and presentation

Reusable workflows should generally return structured results rather than
requiring callers to inspect terminal output.

Keep presentation concerns such as:

- terminal formatting
- progress displays
- tables
- interactive windows
- UI prompts

outside reusable algorithm/workflow logic where practical.

However, do not introduce an elaborate event/reporting system merely to avoid a
small amount of output code.

Choose the simplest boundary that supports actual callers and testing.

---

# 12. Review tests as architecture

Tests should protect important architectural behavior.

Where appropriate, add tests for:

- extension registration
- resolver/factory behavior
- dependency injection
- configuration normalization
- structured command results
- backward compatibility
- lazy optional dependencies
- provider isolation
- extension locality

Prefer small focused tests.

Use lightweight fake implementations rather than initializing expensive
external systems where possible.

---

# 13. Review documentation

Documentation is part of the architecture.

Verify that architectural and extension documentation matches the CURRENT code.

Check:

- imports
- filenames
- class names
- registry APIs
- configuration examples
- CLI examples
- tutorial steps
- tests shown in tutorials

If the repository contains extension tutorials, ensure they use deliberately
simple examples.

Tutorials should teach the extension mechanism before teaching sophisticated
domain algorithms.

When architecture changes, update the canonical architecture documentation.

Avoid duplicating large architectural explanations across many files.

---

# 14. Search for architectural smells

Search for potentially interesting patterns including:

    if model_name ==
    if provider ==
    if tracker ==
    if adapter ==
    if loss ==
    import *
    __import__
    importlib
    sys.path
    globals()
    mutable module-level dict
    broad except Exception
    print(
    argparse
    os.environ

Also inspect:

- large files
- large execute/run methods
- many optional parameters
- constructors with many dependencies
- duplicated parsing
- duplicated dispatch
- repeated factory logic
- repeated serializers
- repeated artifact logic
- deeply nested conditionals

These patterns are not automatically wrong.

Evaluate them in context.

---

# 15. Plan before refactoring

After inspection and analysis, create a concise internal implementation plan.

Prioritize:

1. dependency violations
2. correctness risks
3. extension scatter
4. duplicated dispatch
5. unnecessary coupling
6. accidental complexity
7. documentation drift
8. cosmetic consistency

Do not perform broad cosmetic rewrites unless they directly support the
architectural improvement.

Keep changes incremental and reviewable.

---

# 16. Implement conservatively

Make changes only when the review identified a concrete reason.

Preserve public behavior wherever practical.

Avoid unrelated cleanup.

Avoid speculative abstractions.

Avoid implementing infrastructure solely because it may someday become useful.

When compatibility behavior is required, keep compatibility adapters thin.

Do not allow compatibility layers to become new homes for application logic.

---

# 17. Validate incrementally

After each coherent change:

1. run focused tests
2. inspect failures
3. inspect the diff
4. verify dependency direction
5. verify public behavior

Then run the broadest practical test suite after the refactor.

Also run relevant smoke tests and import checks where appropriate.

---

# 18. Review the final diff

After implementation and successful tests, review the resulting code as if
reviewing another developer's pull request.

Ask:

- Is the new architecture simpler?
- Did the change reduce coupling?
- Did extension locality improve?
- Did we introduce unnecessary abstractions?
- Did we increase indirection?
- Did public APIs become harder to understand?
- Are optional dependencies still isolated?
- Can any newly introduced class/function be removed?
- Did compatibility remain intact?
- Does documentation match the implementation?

Refine again if necessary.

Then re-run affected validation.

---

# 19. Stop condition

Stop refactoring when:

- correctness is protected
- dependency boundaries are sound
- important extensions are sufficiently local
- duplicated architectural behavior has been addressed where worthwhile
- optional dependencies remain isolated
- documentation matches the implementation
- tests protect the important behavior
- remaining issues are primarily stylistic, subjective, or speculative

Do not continue refactoring just because more abstraction is possible.

---

# Guiding principles

Prioritize, roughly in this order:

1. Correctness
2. Understandability
3. Locality of change
4. Explicit dependency flow
5. Testability
6. Replaceability
7. Portability
8. Discoverability
9. Backward compatibility
10. Consistency

Do not optimize for:

- maximum abstraction
- maximum use of design patterns
- minimum line count
- perfect directory symmetry
- speculative future extensibility

Optimize for:

> The smallest coherent architecture that makes the repository easy to
> understand, modify, experiment with, test, and extend.

---

# Completion report

When finished, provide a concise report containing:

## Findings

Concrete architectural issues that were found.

## Changes

Refactors performed and why.

## Simplifications

Unnecessary abstraction, duplication, or indirection removed.

## Extension impact

Where applicable, show how adding representative implementations works after
the refactor.

## Tests

Tests added, changed, and executed.

## Documentation

Architecture or tutorial documentation updated.

## Compatibility

Public behavior deliberately preserved or any unavoidable compatibility issues.

## Remaining debt

Only legitimate architectural issues intentionally left unresolved.

## Validation

Actual tests and smoke checks performed.

Do not invent issues to make the report appear more substantial.
