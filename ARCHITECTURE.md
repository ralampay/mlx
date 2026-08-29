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

`ModeDescriptor` is the source of canonical names, aliases, default actions, action inventories,
purposes, and lazy runner paths. The CLI applies the selected default before dispatch. Local
`--format json` attaches null reporters and suppresses Rich/display presenters; the top-level
boundary serializes the returned structured result once.

## Package Topology and Commands

```text
mlx/
├── cli.py                        CLI parser and top-level error/JSON boundary
├── cli_config.py                 pure parsed-option normalization
├── cli_routing.py                immutable mode descriptors and lazy runner resolution
├── core/                         shared commands, requests, errors, UI, seeds, model summaries
│   ├── artifacts.py             atomic serialization, hashes, and JSON normalization
│   ├── aws/                     shared SageMaker lifecycle infrastructure
│   ├── datasets.py              S3 ZIP staging, safe extraction, cache, resolver protocol
│   ├── presentation.py          shared Rich rendering for core infrastructure events
│   ├── image_backbones.py        neutral penultimate-image-feature contracts
│   ├── deep_svdd.py              shared Deep-SVDD score/calibration semantics
│   └── streaming.py              neutral frame-source contracts and OpenCV decoder
└── modes/
    ├── image_classification/     classification data, models, OOD, training, inference, CAM
    │   └── aws/                  SageMaker Spot lifecycle and native checkpoint recovery
    ├── segmentation/             paired-mask data, U-Net models, metrics, research artifacts
    │   ├── streaming.py          injected frame source/sink contracts and OpenCV adapters
    │   ├── visualization.py      pure mask coloring, blending, and view composition
    │   └── models/backbone_factory.py  isolated classifier-backbone adapter
    ├── object_detection/
    │   ├── models.py             provider-neutral detection values and detector protocol
    │   ├── providers.py          lazy provider registry and provider protocol
    │   ├── commands.py           neutral train, benchmark, create, convert, list, stream commands
    │   ├── evaluation.py         normalized benchmark metrics and research-artifact contract
    │   ├── artifacts.py          shared checkpoint discovery and export-path rules
    │   ├── streaming.py          frame-sink adapter and compatibility frame-source re-exports
    │   ├── aws/                  SageMaker Spot submission, lifecycle, and recovery boundary
    │   ├── tracking/             tracking, MOT evaluation, replay export, registry, algorithms
    │   ├── libreyolo/            LibreYOLO implementation using the Ralampay fork
    │   └── ultralytics/          Ultralytics implementation and compatibility exports
    ├── video_anomaly_detection/  normal-only clip data, 3D/legacy backbones, SVDD, research artifacts
    └── nlp/                      GGUF-backed CSV embedding command and Rich reporter
```

The primary workflow commands are:

| Mode | Commands |
| --- | --- |
| Image classification | `TrainImageClassificationModel`, `SmokeTestImageClassificationModel`, `BenchmarkImageClassification`, `InferImageClassification`, `GenerateImageClassificationCams`, `BuildImageClassificationDataset`, `ListImageClassificationModels`, AWS submit/status/stop/resume commands |
| Segmentation | `TrainSegmentationModel`, `SmokeTestSegmentationModel`, `BenchmarkSegmentation`, `InferSegmentationImage`, `RunSegmentationStreamInference`, `BuildSegmentationDataset`, `ListSegmentationModels` |
| Video anomaly detection | `TrainVideoAnomalyModel`, `BenchmarkVideoAnomalyModel`, `InferVideoAnomaly`, `ListVideoAnomalyModels` |
| Object detection | `TrainObjectDetectionModel`, `BenchmarkObjectDetectionModel`, `CreateObjectDetector`, `ConvertObjectDetectionModel`, `ListObjectDetectionModels`, `RunObjectDetectionStream`, AWS submit/status/stop/resume commands |
| Tracking | `CreateTrackingAlgorithm`, `RunObjectDetectionTrackingCommand`, `RunTrackByDetectionCommand`, `RunTrackingVideo`, `ExportMOTFromClassAwareTracking`, `BenchmarkMOTTracking`, `ExportTrackingReplay` |
| NLP | `EmbedCsvCommand` (`EmbedCsv` is the legacy path-returning API) |

Large commands should keep `execute()` readable by delegating cohesive steps to private methods
or focused helpers. Stateless tensor transforms, metrics, serialization helpers, and model
builders remain functions.

## Portable Training Dataset Sources

Every train-capable mode accepts either its existing local `dataset_path` or an S3 ZIP through
the shared `TrainWithDatasetSource` command. Runners remain composition roots: they inject the
mode's existing training command, its dataset-root contract, the reporter, and the artifact
directory resolver. The wrapper changes only the typed request's resolved `dataset_path`; model
trainers and loaders therefore remain storage-provider neutral. S3/Boto3 details do not enter
mode commands or dataset implementations.

`DatasetSourceSpec` makes the local-versus-S3 choice explicit before staging. It derives typed
request defaults instead of comparing a CLI path sentinel. CLI explicit-option bookkeeping is
consumed at the integration boundary and is not retained as domain request metadata.

`StageS3Dataset` owns the local staging lifecycle. It validates the S3 URI, inspects object
identity, downloads through an injected S3 client, computes SHA-256, securely extracts into a
temporary sibling directory, resolves the mode-specific root, writes a completion manifest, and
atomically publishes a persistent cache entry under `~/.cache/mlx/datasets` by default. Cache
identity includes bucket/key plus VersionId or ETag provenance and object size. Incomplete entries
are never treated as valid. Training artifacts receive `dataset_source.json`; credentials and
profile names are deliberately excluded.

S3 downloads emit structured `dataset_download` lifecycle events rather than terminal output.
The reusable `RichDatasetDownloadProgress` renderer consumes those infrastructure events as one
in-place progress line with byte count, percentage, transfer speed, and remaining time. Each
train-capable mode composes it through `RichInfrastructureEventRenderer`; JSON and direct Python
use keep their null or injected reporters and remain terminal-independent.

The shared streaming ZIP extractor rejects traversal, absolute and Windows-drive paths,
symbolic links, special files, normalized duplicates, and file/directory conflicts. It checks
declared uncompressed size against free space and an optional caller limit. Dataset semantics
remain mode owned through injected root resolvers located in each mode's data module: object detection requires exactly one
`data.yaml`; classification requires one `train`/`val` root; segmentation additionally requires
paired image/mask directories; video anomaly detection requires normal train/validation roots.
The same extractor and applicable root resolver are used by SageMaker container entrypoints.

The CLI rejects an explicitly supplied local dataset together with `--dataset-s3-uri`, rejects
S3 input for non-training actions, and requires persistent `--output` for local S3 training.
`--profile` is resolved only at the Boto3 construction boundary. For SageMaker object detection
and image classification, an explicit CLI S3 URI overrides the YAML URI for a new submission;
resume validation continues to enforce the original run-spec URI.
User-facing archive contracts and operational guidance live in
[`docs/s3-dataset-training.md`](docs/s3-dataset-training.md).

## Object-Detection Providers

Object detection is selected with `--provider`; `ultralytics` is the default and `libreyolo` is
the alternative. The CLI routes to
`mlx.modes.object_detection.runner`, which resolves providers through a string registry only when
the selected action executes. Importing the CLI or another mode therefore does not require either
provider to be installed.

All providers normalize predictions to `DetectionResult` containing `Detection` values with
floating-point `xyxy` boxes. Tracking,
annotation, streaming, and downstream callers depend only on that contract. The provider protocol
supports five capabilities:

1. train from `TrainObjectDetectionRequest`;
2. benchmark from `BenchmarkObjectDetectionRequest`;
3. create a `DetectionAdapter` from `ObjectDetectionRequest`;
4. export from `ConvertObjectDetectionRequest`;
5. list models from `ListObjectDetectionModelsRequest`.

`BenchmarkObjectDetectionModel` normalizes both provider validators to `precision`, `recall`,
`f1`, `map_50`, and `map_50_95`. Provider adapters retain responsibility for model loading,
dataset integration, prediction JSON, native plots, and exception translation. The neutral
artifact writer owns `metrics.json`, `metrics.csv`, `native_metrics.json`, and
`run_metadata.json`, including model hashing and evaluator provenance. This gives standalone
benchmarks and optional post-training validation the same result schema without leaking either
provider API into the command.

`TrainObjectDetectionModel` composes the same benchmark capability when
`validate_after_training` is enabled. It benchmarks the selected best/last checkpoint and returns
an `ObjectDetectionTrainingResult`; ordinary training preserves its former provider-native return
value. Validation is opt-in because it performs an additional full dataset pass.

The `track` CLI mode routes to the nested tracking runner because tracking-by-detection remains
owned by object detection. `RunTrackingVideo` composes the selected provider's `DetectionAdapter`,
an OpenCV frame source, a registry-selected `TrackingAlgorithm`, composed streaming class-aware
JSONL and MOT output, optional MOT evaluation, portable replay export, and an optional injected
frame sink/renderer pair. Trackers receive only normalized
`TrackingDetection` values and may be selected by
built-in alias or an external `package.module:ClassName`; constructor keyword arguments come from
an optional JSON configuration. The built-in registry is immutable, and applications extend it by
creating and injecting a new `TrackerRegistry` rather than changing process-wide state. SORT and
ByteTrack are the built-in reference implementations.

Tracking output has two synchronized projections with 1-based frame and track IDs.
`tracks.jsonl` is the versioned, provider-neutral source that retains class ID, optional label,
confidence, and `xyxy` geometry. `tracks.txt` is a strict headerless 10-column MOTChallenge file
and deliberately has no nonstandard class column. `TrackingResultWriter` composes the focused
writers so tracker classes remain unaware of serialization. `ExportMOTFromClassAwareTracking`
validates the JSONL and can select classes while recreating the standard MOT projection. Only
confirmed tracks observed in the current frame are persisted; lost and tentative state remain
algorithm details. Benchmarking ignores unavailable world coordinates and reports MOTA, mean
matched IoU, IDF1, precision, recall, false positives, misses, and identity switches. Session
memory is bounded by active/lost tracks and current-frame detections; video frames and complete
trajectory histories are not retained.

`ExportTrackingReplay` is downstream of tracking serialization and does not depend on a detector,
tracker, OpenCV, or source video. It writes a versioned `replay.json` projection plus a
self-contained `replay.html` browser player. The JSON preserves canvas/FPS metadata, run settings,
prediction boxes and class metadata, optional ground-truth boxes, and optional metrics, but omits
provider objects and absolute video paths. It validates that the optional class-aware sidecar and
MOT predictions describe identical rows before combining them. `OpenCVFrameSource` exposes FPS
and geometry through the optional
`MetadataFrameSource` capability; commands still accept minimal `FrameSource` implementations,
and decoded frame shapes remain authoritative for replay canvas dimensions. This interface
segregation keeps fake, camera, and future non-OpenCV sources portable.

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

## Video Anomaly Detection

`video_anomaly_detection` is a first-class normal-only workflow. Its runner constructs typed
requests and presentation adapters; commands own training, benchmarking, listing, and headless
video inference. Mode-owned data, backbone, metric, and artifact modules remain independent of
CLI state. The default tensor flow is:

```text
[B,T,C,H,W] -> [B,C,T,H,W] -> family-aware inflated 3D backbone
              -> global 3D pool -> [B,D] -> SVDD projection -> [B]
```

The image-classification registry remains authoritative for compatible aliases and 2D source
weights. Cross-mode access is isolated in
`video_anomaly_detection.models.classification_compat`; other video model, training, and listing
modules depend on this gateway rather than classifier internals. It supplies frame feature
backbones, an inflation-safe feature wrapper, capability lookup, source provenance, and custom
block classification. The video-owned 3D factory maps each standard family to a dedicated clip-native class,
inflates spatial kernels, replaces normalization/pooling with 3D equivalents, keeps pointwise
temporal kernels at one, and enforces temporal stride one. Its registry contains only 3D
construction strategy; it does not redefine model-family capability. Standard pretrained weights
are fully inflated, while Drax-specific branches remain freshly initialized and are recorded as
partial provenance. Both Drax families preserve `average` and `sknet` fusion.

Checkpoint version 2 records `backbone_mode`, concrete 3D class, inflation kernel, stride policy,
pooling, and pretrained provenance. Missing `backbone_mode` identifies a version-1 checkpoint,
which is reconstructed through the retained frame-wise `build_image_feature_backbone` plus
registered `TEMPORAL_ENCODERS` path. New requests default to `3d`; the legacy path remains an
explicit compatibility boundary rather than being folded into the 3D contract.

Deep-SVDD squared-distance and quantile semantics are shared in `mlx.core.deep_svdd`; image
classification and video anomaly models retain mode-owned projection/checkpoint structures. The
video center is initialized once from normal training embeddings and stored as a fixed buffer.
Optimization consumes only normal clips. The deployment threshold is calibrated from normal
validation scores and persisted. Resume restores the exact center, optimizer, history, and RNG
state; benchmark and inference reject checkpoints without a stored center or finite threshold and
never recalibrate using test data.

The training command emits batch-level phase events while initializing the center, optimizing an
epoch, validating, and calibrating final thresholds. `RichVideoAnomalyReporter` renders each phase
as a transient in-place progress line and writes one persistent epoch summary containing training
SVDD loss, validation loss, current best validation objective, and learning rate. The command stays
terminal-neutral, and callback/null reporters retain the same structured lifecycle for Python,
tests, automation adapters, and JSON execution.

The generic dataset owns deterministic complete windows over
`<split>/{normal,anomaly}/<source>/<frames>`. Training and validation expose only `normal`; an
anomaly source under training is an actionable error. Source/frame metadata travels through
evaluation into per-window and aggregated per-frame records. Direct compressed-video decoding is
provided for inference through `mlx.core.streaming.OpenCVFrameSource`; frame-sequence extraction
is the current training-data boundary. The same core frame-source API is re-exported by object
detection to preserve its existing public imports.

Benchmark artifacts are first-class outputs: structured clip/frame metrics, prediction records,
ROC/PR data, plots, provenance including checkpoint SHA-256, and a deterministic Markdown report.
`VideoAnomalyBenchmarkArtifactWriter` owns serialization, plotting, and report generation while
the command coordinates evaluation. Commands emit `WorkflowEvent` values; all Rich rendering
remains in the mode presentation module.

## AWS SageMaker Training

Object-detection training selects an execution platform independently from its model provider.
`local` remains the default and never reads AWS configuration. `--platform aws` routes to the
mode-owned `mlx.modes.object_detection.aws` package, where command classes coordinate injected
S3, ECR, IAM, SageMaker, CloudWatch, Docker, and checkpoint components. The neutral
`TrainObjectDetectionModel` still owns provider selection inside the training container.
Local AWS authentication is selected at the composition boundary: an explicit `--profile`
overrides `aws.profile` in YAML, which otherwise delegates to Boto3's standard credential chain.
Credential material never crosses into commands, manifests, images, or training hyperparameters.

AWS training uses one logical MLX run across one or more SageMaker job attempts. Managed Spot is
the default. SageMaker restores `/opt/ml/checkpoints` after an interruption; the container then
validates two alternating recovery slots by epoch and SHA-256 before reconstructing the
provider's `last.pt`. Manual resume creates a new SageMaker job attempt for the same logical run,
reuses the original immutable image reference, and permits only capacity/runtime changes and a
higher total epoch target. It also preserves the original serialized training payload, changing
only the epoch target, so a newer local CLI cannot introduce default fields that are unknown to
the immutable training image. Provider/model/dataset changes are rejected at that boundary.

Users own the dataset ZIP and shared checkpoint S3 bucket or prefix. MLX never creates, deletes,
empties, or attaches lifecycle policies to those buckets. Logical runs and attempts receive
separate prefixes under the configured checkpoint base. MLX may create and reuse a tagged ECR
repository and narrowly scoped SageMaker execution role when the caller does not provide them.
Stopping compute preserves checkpoints and returns a resumable job identity.

Shared AWS infrastructure lives under `mlx.core.aws`: lifecycle values and commands, Boto3 client
construction, source hashing, Docker/ECR publication, and side-effect-free status translation.
Mode AWS modules retain compatibility re-exports. Each mode continues to own configuration and
IAM policy, training payload serialization, checkpoint/recovery codecs, container entrypoints,
and service composition; these distinct ML semantics are deliberately not folded into a generic
service.

Recovery is intentionally a completed-epoch guarantee. Provider `last.pt` files include model,
epoch, optimizer, scaler, EMA, and available scheduler/RNG state; a project-owned publisher only
announces CloudWatch progress after copying and validating a complete checkpoint into the
inactive recovery slot. If the newest slot is corrupt, the prior slot is used. Work from an
incomplete interrupted epoch may be repeated.

When `training.validate_after_training` is enabled, the training container runs the neutral
benchmark against `validation_split` after selecting the checkpoint. SageMaker stages the common
research artifacts and provider-native plots/predictions under `benchmark/` in `model.tar.gz`, in
addition to the selected checkpoint and training summary. CloudWatch epoch/progress/ETA metrics
remain operational signals rather than model-quality metrics.

Image classification exposes the same asynchronous lifecycle through its mode-owned `aws`
package for standard, joint Deep-SVDD, and one-shot Siamese training. It uses the classifier's
native `{model}.last.pth` full-state checkpoint, keeping model, optimizer, completed epoch,
history, random state, labels, dimensions, family, and OOD state under the classification
boundary. Two checksum-validated recovery slots and the best deployable checkpoint synchronize
through SageMaker's checkpoint directory. Manual resume retains the original training payload
and immutable image, allowing only a higher total epoch target plus capacity/runtime changes.
Final model artifacts include the best checkpoint, resumable checkpoint, training CSV, and a
sanitized summary.

## Shared Primitives, Requests, and Registries

`mlx.core.artifacts` contains only behavior that is identical across modes: JSON-safe value
normalization, atomic JSON/PyTorch writes, CSV serialization, and SHA-256 hashing. RNG capture and
restore live in `mlx.core.random`. Checkpoint schemas, compatibility checks, naming, plots, and
reports remain mode owned. Deep-SVDD sharing remains similarly narrow in `mlx.core.deep_svdd`.

Image classification and segmentation expose action-specific request subclasses while retaining
their former umbrella request types for Python compatibility. `ConfigRequest` keeps unknown
public compatibility values but discards underscore-prefixed CLI bookkeeping. Commands may still
adapt a typed request to a mapping at a legacy boundary; runners are responsible for selecting
the action-specific type.

Tracking, object-detection providers, image-classification custom models, temporal encoders, and
3D video backbones expose immutable registry mappings or registry value objects. Extension APIs
return a new registry for dependency injection. Historic registration calls also update their
default registry for compatibility, while exported mappings remain read-only to callers.

NLP (`pandas`, `llama-cpp-python`) and Grad-CAM are optional package extras. Their adapters remain
lazy and raise actionable `MLXUserError` messages when the selected capability is not installed.

## Presentation, Errors, and Compatibility

Commands expose structured values and provider-neutral protocols. Long-running training,
benchmark, dataset-build, conversion, CAM, and embedding commands report structured
`WorkflowEvent` values; task-specific Rich tables, progress bars, prompts, and panels belong to
mode-owned `presentation.py` adapters. Rendering for shared infrastructure events may live in
`mlx.core.presentation` and is composed by those mode adapters. Compatibility functions may
attach the adapters, while direct
command construction defaults to a no-op reporter and remains suitable for Python and tests.
Detection streaming and
tracking video execution support headless use through injected presentation boundaries:
`RunObjectDetectionStream` accepts injected detector, frame source, frame sink, renderer, and
reporter objects. `RunTrackingVideo` accepts an optional paired frame sink and tracking renderer;
without them it writes tracking artifacts headlessly. The tracking CLI injects an OpenCV sink and
a mode-owned renderer by default, while `--no-display` leaves both absent. The renderer consumes
only `TrackingFrameResult` values and draws current observations with stable track-ID colors,
boxes, identity/class/confidence/status labels, and a frame summary. User-stopped playback
finalizes partial MOT output but skips whole-video benchmarking. The CLI supplies OpenCV and Rich
adapters. Other modes keep their output
formatters in `presentation.py`; ongoing changes must move new terminal/window behavior toward
the same injected-adapter boundary rather than adding UI work to model, data, or metric modules.
Segmentation's reusable visualization transforms live in `visualization.py`; only window display
and prompts remain in presentation. Its encoder consumes a `ClassificationBackboneFactory`, with
the existing image-classification implementation isolated in the default compatibility adapter
instead of being imported by the encoder itself.

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
