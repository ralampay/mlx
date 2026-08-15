# MLX Architecture

This document is the canonical architectural reference for MLX. User-facing commands and
dataset formats remain in `README.md` and the mode guides under `docs/`.

## Design Goals

MLX is organized so that machine-learning workflows can be invoked from the CLI, Python,
tests, or another application without rewriting their orchestration. The governing rules are:

- non-trivial workflows are command classes with injected inputs and an `execute()` entrypoint;
- runners parse and normalize configuration, select actions, attach presentation adapters, and
  invoke commands;
- shared infrastructure belongs in `mlx.core`, while task-specific behavior stays under its mode;
- third-party libraries are integration details behind mode-owned protocols and adapters;
- recoverable failures cross the application boundary as actionable `MLXUserError` instances;
- old Python entry points remain thin compatibility wrappers when commands replace functions.

## Layers and Dependency Direction

```text
CLI (`mlx.cli`)
    ↓ lazy mode lookup
mode runner (`mlx.modes.<mode>.runner`)
    ↓ typed request + presentation adapters
command (`execute()`)
    ↓ composition
models / data / metrics / artifacts / provider protocols
    ↓ integration boundary
PyTorch, torchvision, OpenCV, Ultralytics, ONNX Runtime, llama.cpp
```

Dependencies point downward. Models, datasets, and providers must not import runners or CLI
parsing. A provider-specific package may import neutral mode contracts; neutral contracts must
not import a provider implementation.

`mlx.core.commands` defines the common `Command`, `WorkflowReporter`, and `WorkflowEvent`
contracts. `mlx.core.requests.ConfigRequest` supplies a lossless bridge between typed request
dataclasses and legacy configuration dictionaries. New application APIs should accept a typed
request; dictionary entry points exist only for CLI and compatibility use.

## Package Topology and Commands

```text
mlx/
├── core/                         shared commands, requests, errors, UI, seeds, model summaries
└── modes/
    ├── image_classification/     classification data, models, OOD, training, inference, CAM
    ├── segmentation/             paired-mask data, U-Net models, metrics, research artifacts
    ├── object_detection/
    │   ├── models.py             provider-neutral detection values and detector protocol
    │   ├── providers.py          lazy provider registry and provider protocol
    │   ├── commands.py           neutral train, create, convert, list, and stream commands
    │   ├── artifacts.py          shared checkpoint discovery and export-path rules
    │   ├── streaming.py          frame-source/frame-sink ports and OpenCV adapters
    │   ├── tracking/             provider-neutral tracking contracts and algorithms
    │   ├── libreyolo/            LibreYOLO implementation using the Ralampay fork
    │   └── ultralytics/          Ultralytics implementation and compatibility exports
    └── nlp/                      GGUF-backed CSV embedding workflows
```

The primary workflow commands are:

| Mode | Commands |
| --- | --- |
| Image classification | `TrainImageClassificationModel`, `SmokeTestImageClassificationModel`, `BenchmarkImageClassification`, `InferImageClassification`, `GenerateImageClassificationCams`, `BuildImageClassificationDataset`, `ListImageClassificationModels` |
| Segmentation | `TrainSegmentationModel`, `SmokeTestSegmentationModel`, `BenchmarkSegmentation`, `InferSegmentationImage`, `RunSegmentationStreamInference`, `BuildSegmentationDataset`, `ListSegmentationModels` |
| Object detection | `TrainObjectDetectionModel`, `CreateObjectDetector`, `ConvertObjectDetectionModel`, `ListObjectDetectionModels`, `RunObjectDetectionStream` |
| Tracking | `RunObjectDetectionTrackingCommand`, `RunTrackByDetectionCommand` |
| NLP | `EmbedCsvCommand` (`EmbedCsv` is the legacy path-returning API) |

Large commands should keep `execute()` readable by delegating cohesive steps to private methods
or focused helpers. Stateless tensor transforms, metrics, serialization helpers, and model
builders remain functions.

## Object-Detection Providers

Object detection is selected with `--provider`; `ultralytics` is the default and `libreyolo` is
the alternative. The CLI routes to
`mlx.modes.object_detection.runner`, which resolves providers through a string registry only when
the selected action executes. Importing the CLI or another mode therefore does not require either
provider to be installed.

All providers normalize predictions to `DetectionResult` containing `Detection` values. Tracking,
annotation, streaming, and downstream callers depend only on that contract. The provider protocol
supports four capabilities:

1. train from `TrainObjectDetectionRequest`;
2. create a `DetectionAdapter` from `ObjectDetectionRequest`;
3. export from `ConvertObjectDetectionRequest`;
4. list models from `ListObjectDetectionModelsRequest`.

To add another provider:

1. create `mlx.modes.object_detection.<provider>/provider.py` without importing it globally;
2. implement `ObjectDetectionProvider`, translating library results to the neutral detection types;
3. register a lazy `module:function` factory in `PROVIDER_REGISTRY`;
4. translate dependency and model/data errors into `MLXUserError` at the provider boundary;
5. add fake-provider contract tests plus provider-specific decoding and integration tests;
6. document supported models, formats, training semantics, and install dependencies.

Provider dependencies are named optional extras so users can install only the integration they
need. `object-detection-ultralytics` installs the Ralampay Ultralytics fork,
`object-detection-libreyolo` follows the `release` branch of the Ralampay LibreYOLO fork with its
ONNX dependencies, and `object-detection` installs both. Provider packages must stay lazy so these
extras remain independent.

The Ultralytics provider lists `yolo26`, `draxnet-ave-yolo26`, and
`draxnet-sknet-yolo26` as canonical architectures. The compatibility alias
`draxnet-yolo26` resolves to the fixed-average DraxNet variant. Built-in model aliases resolve to
YAML files shipped in the installed provider package so listing and training use the same
definitions; explicit filesystem paths remain supported for custom architectures.

LibreYOLO training and listing are first-class for `yolo9-t`, `yolo9-s`, `yolo9-m`, `yolo9-c`,
and `yolo9-s-drax-b5`. A shared model specification keeps listing and scratch training aligned;
the Drax alias selects the release branch's first supported experiment configuration: YOLOv9-S,
B5 only, attention and efficient mode enabled, average fusion, and zero drop path. Inference and
conversion adapters may accept other axis-aligned detection checkpoints supported by the fork,
but non-detection tasks and cross-provider checkpoint loading are outside the neutral provider
contract.

## Presentation, Errors, and Compatibility

Commands expose structured values and provider-neutral protocols. Streaming is fully headless:
`RunObjectDetectionStream` accepts injected detector, frame source, frame sink, renderer, and
reporter objects. The CLI supplies OpenCV and Rich adapters. Other modes keep their output
formatters in `presentation.py`; ongoing changes must move new terminal/window behavior toward
the same injected-adapter boundary rather than adding UI work to model, data, or metric modules.

Invalid CLI input, absent files, unsupported actions/providers, missing optional libraries, bad
dataset layouts, and camera/video failures raise `MLXUserError`. Model-internal invariant failures
may use `ValueError` or `RuntimeError` when they indicate programmer errors rather than recoverable
user input. `MLXAbort` is reserved for intentional cancellation.

Compatibility functions such as `train_image_classification`, `infer_segmentation_image`, and
`convert_object_detection_model` construct a typed request or provider command and call
`execute()`. Former Ultralytics-owned detection and tracking imports are re-exported from their old
paths. Compatibility wrappers must not accumulate new business logic.

## Testing and Change Rules

- Unit-test commands with fake models, providers, reporters, frame sources, and frame sinks.
- Test each provider against the neutral contract; provider-independent tests must not import its
  third-party package.
- Keep runner tests focused on defaults, action dispatch, request construction, and presentation
  wiring.
- Verify user-facing failures at integration boundaries and run `python -m pytest -q` before handoff.
- Any code or configuration change must review this document. Update it in the same change whenever
  command inventory, package ownership, dependencies, interfaces, provider behavior, or data flow
  changes.
