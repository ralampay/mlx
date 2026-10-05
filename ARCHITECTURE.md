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
PyTorch, torchvision, OpenCV, Ultralytics, ONNX Runtime, llama.cpp, ChromaDB
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
│   ├── paired_statistics.py     exact paired superiority, bootstrap interval, and TOST equivalence analysis
│   ├── vector_transforms.py     provider-neutral batch representation-transform contract
│   ├── aws/                     shared SageMaker lifecycle infrastructure
│   ├── datasets.py              S3 ZIP staging, safe extraction, cache, resolver protocol
│   ├── presentation.py          shared Rich rendering for core infrastructure events
│   ├── image_backbones.py        neutral penultimate-image-feature contracts
│   ├── binary_metrics.py         shared higher-is-positive binary score evaluation
│   ├── deep_svdd.py              shared Deep-SVDD score/calibration semantics
│   └── streaming.py              neutral frame I/O contracts and lazy OpenCV adapters
└── modes/
    ├── image_classification/     classification data, models, OOD, training, inference, CAM
    │   └── aws/                  SageMaker Spot lifecycle and native checkpoint recovery
    ├── image_recognition_oc/     normal-only image recognition algorithms and research artifacts
    │   └── aws/                  single/all-backbone SageMaker training and benchmarking
    ├── segmentation/             paired transforms/masks, U-Net models/groups, metrics, samples, research artifacts
    │   ├── streaming.py          compatibility names for shared frame ports and OpenCV adapters
    │   ├── visualization.py      pure mask coloring, blending, and view composition
    │   └── models/backbone_factory.py  isolated classifier-backbone adapter
    ├── saliency_mapping/         still-image SOD data, loss, metrics, training, inference, and artifacts
    │   └── models/               saliency registry with segmentation compatibility builders
    ├── object_detection/
    │   ├── models.py             provider-neutral detection values and detector protocol
    │   ├── providers.py          lazy provider registry and provider protocol
    │   ├── commands.py           neutral train, benchmark, create, convert, list, stream commands
    │   ├── evaluation.py         normalized benchmark metrics and research-artifact contract
    │   ├── adapter_data.py       deterministic DAWN conversion, class mapping, and stratified split
    │   ├── adapter_metrics.py    fixed-threshold detection precision and recall
    │   ├── adapter_experiment.py typed request and checkpoint verification/training/evaluation commands
    │   ├── adapter_report.py     comparative CSV and paired-seed Markdown report command
    │   ├── adapter_slices.py     post-training prediction cache and targeted slice-analysis commands
    │   ├── comparison_experiment.py paired low-data YOLOX/CSP-Drax preparation, execution, and statistical report commands
    │   ├── artifacts.py          shared checkpoint discovery and export-path rules
    │   ├── streaming.py          frame-sink adapter and compatibility frame-source re-exports
    │   ├── aws/                  SageMaker submission, recovery, and comparison commands
    │   ├── tracking/             tracking, MOT evaluation, replay export, registry, algorithms
    │   ├── libreyolo/            LibreYOLO implementation using the Ralampay fork
    │   │   ├── adapter_backend.py strict checkpoint/CUDA/environment/calibration and BN-safe trainer boundary
    │   │   ├── adapter_loading.py standalone foundation/adapter loading and shared strict adapter restoration
    │   │   └── adapter_slice_backend.py strict checkpoint reconstruction and prediction-cache boundary
    │   └── ultralytics/          Ultralytics implementation and compatibility exports
    ├── video_anomaly_detection/  normal-only clip data, 3D/legacy backbones, SVDD, research artifacts
    │   └── aws/                  sequential all-model SageMaker lifecycle and recovery
    ├── text_embedding/           BEIR embedding, vector indexing, retrieval metrics, research artifacts
    │   ├── embedding/            neutral provider protocol and lazy llama.cpp adapter
    │   └── vector_store/         neutral store protocol, immutable registry, lazy Chroma adapter
    ├── autoencoder/              generic 1D reconstruction training and bottleneck adapters
    └── nlp/                      legacy GGUF-backed CSV embedding compatibility API
```

The primary workflow commands are:

| Mode | Commands |
| --- | --- |
| Image classification | `TrainImageClassificationModel`, `SmokeTestImageClassificationModel`, `BenchmarkImageClassification`, `InferImageClassification`, `GenerateImageClassificationCams`, `BuildImageClassificationDataset`, `ListImageClassificationModels`, AWS submit/status/stop/resume commands |
| One-class image recognition | `TrainImageOneClassModel`, `BenchmarkImageOneClass`, `InferImageOneClass`, `ListImageOneClassModels`, AWS submit/status/stop/resume commands |
| Segmentation | `TrainSegmentationModel`, `TrainAllSegmentationModels`, `GenerateSegmentationSamples`, `SmokeTestSegmentationModel`, `BenchmarkSegmentation`, `InferSegmentationImage`, `RunSegmentationStreamInference`, `BuildSegmentationDataset`, `ListSegmentationModels` |
| Saliency mapping | `TrainSaliencyModel`, `TrainSaliencyModelGroup`, `GenerateSaliencySamples`, `SmokeTestSaliencyModels`, `BenchmarkSaliencyMapping`, `BenchmarkSaliencyModelGroup`, `InferSaliencyImage`, `BuildSaliencyDataset`, `ListSaliencyModels` |
| Video anomaly detection | `TrainVideoAnomalyModel`, `BenchmarkVideoAnomalyModel`, `InferVideoAnomaly`, `ListVideoAnomalyModels`, AWS all-model submit/status/resume commands |
| Object detection | `TrainObjectDetectionModel`, `FineTuneObjectDetectionModel`, `BenchmarkObjectDetectionModel`, `CreateObjectDetector`, `ConvertObjectDetectionModel`, `ListObjectDetectionModels`, `RunObjectDetectionStream`, `PrepareDawnAdapterDataset`, `VerifyFoundationCheckpoint`, `CalibrateAdapterBatchSize`, `RunAdapterExperiment`, `GenerateAdapterReport`, `CacheAdapterSlicePredictions`, `GenerateAdapterSliceReport`, `PrepareCSPDraxComparison`, `RunCSPDraxComparison`, `AnalyzeCSPDraxComparison`, AWS submit/status/stop/resume and best-model locator commands |
| Tracking | `CreateTrackingAlgorithm`, `RunObjectDetectionTrackingCommand`, `RunTrackByDetectionCommand`, `RunTrackingVideo`, `PrepareTrackingBenchmarks`, `CompileTrackingVideo`, `BenchmarkTrackingDataset`, `ExportMOTFromClassAwareTracking`, `BenchmarkMOTTracking`, `ExportTrackingReplay` |
| Text embedding | `EmbedTextCommand`, `BenchmarkTextEmbeddingCommand`, `PrepareRetrievalDatasets`, `BenchmarkAutoencoderRetrieval`, `TransformEmbeddingArtifacts`, `AnalyzeAutoencoderRetrieval`, `BenchmarkConfiguredAutoencoders`, `SelectAutoencoderExperimentSettings`; legacy `EmbedCsvCommand` remains supported |
| Autoencoder | `TrainAutoencoder`, `EmbedAutoencoder`, `ListAutoencoderModels`, `ListAutoencoderLosses` |

Large commands should keep `execute()` readable by delegating cohesive steps to private methods
or focused helpers. Stateless tensor transforms, metrics, serialization helpers, and model
builders remain functions.

Saliency owns an immutable `SaliencyModelRegistry` of `(name, config) -> model` builders producing
one-channel logits. Built-in entries retain segmentation identifiers and groups through
`saliency_mapping.compatibility`; custom saliency models need no segmentation registration.
That gateway also contains the deliberate reuse of segmentation image policies and sample
selection. Saliency-owned data, sigmoid application, BCE/SSIM/IoU loss, SOD metrics, checkpoints,
artifacts, and presentation do not flow back into segmentation.

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

Segmentation `--model all` and `--model all-small` keep the shared staging command outside the
batch command, so an S3 ZIP is inspected, downloaded, and extracted once before the resolved local
root is injected into every model request. Single-model and batch commands therefore remain
storage-provider neutral.
The segmentation data boundary owns paired spatial transforms. Resize remains backward-compatible;
random crops share coordinates between image and mask and resolve to deterministic center crops for
validation, test, benchmarking, and samples. Crop padding uses zero for both inputs and class-index
masks. Model groups are explicit registry-owned sets so membership remains stable when model
implementations change.

The runner removes its implicit local dataset default when an S3 URI is explicitly supplied while
continuing to reject callers that explicitly provide both sources.

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
provider API into the command. Benchmark requests enable provider-native progress by default;
the CLI composition boundary applies that action-specific default and preserves an explicit
`--no-verbose` override.

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

Tracking's `build-dataset` action invokes `PrepareTrackingBenchmarks`: source-specific
MOT image and PersonPath22 adapters describe media, normalized annotations, and conversion
provenance. The command copies into temporary sequence directories and publishes them only
after manifest/MOT validation. Inputs are independent copies, not links. The preparation report
records unavailable, unlabeled, and invalid sequences; conflicting duplicate identities are
rejected rather than arbitrarily merged. Dataset preparation never downloads missing sources.

`data.TrackingSequence` is the version-1 manifest contract: dataset/split/name, relative media
and ground-truth paths, media kind, geometry, FPS, and increasing zero-based source frame indices.
Evaluation frame IDs are contiguous and one-based. `SequenceFrameSource` owns numeric image
ordering and sequential video selection. PersonPath22 selects explicitly annotated frames and
visible person boxes; manifests document the simplified policy and identity mappings.

`BenchmarkTrackingDataset` accepts a typed `TrackingBenchmarkRequest` and composes existing
tracking sessions and MOT evaluation per sequence, reusing the detector but resetting the tracker.
The runner injects display and trajectory rendering factories. In the default visual path,
`CompileTrackingVideo` writes FFV1 AVI and verifies its decoded pixels against the selected source
frames before inference. The video integration boundary owns encoding and the composed saved/live
frame sink. `TrackingTrajectoryRenderer` owns only bounded presentation history (60 observations
per identity, expired after 60 absent frames). No histories enter tracking algorithms.

`real_time_results=False` bypasses compilation and all rendering while retaining the same frame
selection and evaluation. `display=False` suppresses only the live window; JSON CLI output also
suppresses the window. Each sequence retains the existing tracking/replay/metric artifacts;
visual runs add `compiled.avi` and `annotated.mp4`. Root JSON/CSV summaries contain per-sequence
results without averaging percentages, and metadata records settings, manifest/model hashes,
and completion status. Cancellation skips the interrupted sequence's metrics and stops the batch.
The evaluator remains MLX's existing MOT evaluator, not an implementation of official dataset
protocols or crowd-region suppression. Input, codec, annotation, and filesystem failures are
reported as contextual `MLXUserError` instances. `track run` remains compatible.

To add another provider:

1. create `mlx.modes.object_detection.<provider>/provider.py` without importing it globally;
2. implement `ObjectDetectionProvider`, translating library results to the neutral detection types;
3. register a lazy `module:function` factory with `ProviderRegistry.register` and inject the returned
   registry (built-in defaults live in `providers.py`);
4. translate dependency and model/data errors into `MLXUserError` at the provider boundary;
5. add fake-provider contract tests plus provider-specific decoding and integration tests;
6. document supported models, formats, training semantics, and install dependencies.

Provider dependencies are named optional extras so users can install only the integration they
need. `object-detection-ultralytics` installs the Ralampay Ultralytics fork,
`object-detection-libreyolo` pins the CSP-capable Ralampay LibreYOLO fork commit with its
ONNX dependencies, and `object-detection` installs both. Provider packages must stay lazy so these
extras remain independent.

The Ultralytics provider lists `yolo26`, `draxnet-ave-yolo26`, and
`draxnet-sknet-yolo26` as canonical architectures. The compatibility alias
`draxnet-yolo26` resolves to the fixed-average DraxNet variant. Built-in model aliases resolve to
YAML files shipped in the installed provider package so listing and training use the same
definitions; explicit filesystem paths remain supported for custom architectures.

LibreYOLO training and listing share an immutable model specification inventory and a lazy
provider-local `model_factory.build_scratch_model` construction boundary. The inventory selects
public library classes: `LibreYOLO9` for `yolo9-{t,s,m,c}` and `yolo9-s-drax-b5`, `LibreYOLOX`
for `yolox-{n,t,s,m,l,x}`, `LibreYOLO9DraxMobileNetV3Large` for
`yolo9-drax-mobilenet-v3-large-{t,s,m,c}`, and `LibreYOLOXDraxMobileNetV3Large` for
`yolox-drax-mobilenet-v3-large-{n,t,s,m,l,x}`, and `LibreYOLOXDraxCSPM` for
`yolox-drax-csp-m`. The MobileNet
Drax MobileNet pyramid variant fixes the compact pyramid backbone with a 0.67-deep, 0.875-wide
YOLOX neck/head. The CSP-Drax model keeps the YOLOX-M CSPDarknet backbone and
uses a P3 refiner, a compact P5 Drax block, and a narrower PAN/head. These variants have
separate checkpoint families.
New architectures extend this inventory rather than
adding selection branches to training or listing commands. Missing public classes raise an
actionable provider-build error; importing MLX does not import the provider library.

Only `yolo9-s-drax-b5` receives `DraxConfig`: B5 only, attention and efficient mode enabled,
average fusion, and zero drop path. MobileNet variants own their fixed Drax configuration in
LibreYOLO. Their `pretrained=True` training initializes ImageNet backbone features only;
construction and listing do not download weights. Warm starts, resume, inference, benchmarking,
and conversion use the generic `LibreYOLO` checkpoint loader, preserving checkpoint architecture
and provider-owned preprocessing. Non-detection tasks and cross-provider checkpoint loading
remain outside the neutral provider contract. ONNX export relocation supports destinations on
another filesystem and translates file-transfer errors at the conversion boundary.

For `yolox-drax-mobilenet-v3-large`, the training request forwards the optional
`incremental_adapter`, `incremental_adapter_train_only`, and
`incremental_adapter_type` controls to the LibreYOLO provider.
The provider validates their dependency, while LibreYOLO owns adapter attachment, foundation
freezing, BatchNorm mode handling, and optimizer construction. Other providers do not implement
this family-specific training behavior.

Local detector feature distillation is configured by typed training-request fields
`distiller`, `distill_loss`, `distill_weight`, `distill_temperature`, and
`distill_mask_ratio`. Mode-owned validation rejects unsupported actions/platforms
and providers before dataset staging. `TrainLibreYOLOObjectDetection` composes
`PrepareDistillationRun`, which resolves teacher identity, validates resume
provenance, and translates options to LibreYOLO training arguments. LibreYOLO owns
teacher freezing, feature taps, losses, and optimizer integration; no teacher is
added to MLX inference or model definitions. `distillation.json` records teacher
SHA-256 and effective loss settings beside the run checkpoints. Scratch/fine-tune
script entrypoints protect against accidental automatic recovery. The local shell
entrypoint accepts model, teacher/student paths, local or S3 data, and training
options; CLI overrides environment defaults, and relative paths remain relative
to the caller. It only composes the existing CLI workflow. See
[local distillation](docs/object_detection/distillation.md) for usage.

## One-Class Image Recognition

`image_recognition_oc` owns normal-only still-image training, single-image inference, binary
benchmarking, and model/backbone listing. `--model` selects the one-class algorithm and `--backbone`
selects a standard image-classification feature model. The mode's immutable, injectable
`OneClassAlgorithmRegistry` initially contains `deep-svdd`; its provider owns construction,
initialization, loss, scoring, calibration, and algorithm checkpoint metadata so future algorithms
do not enter command branching.

The image-classification registry remains authoritative for backbone aliases and pretrained
weights. All cross-mode imports are isolated in `image_recognition_oc.backbones`, which exposes the
neutral `ImageFeatureBackbone` contract and excludes Siamese families. Other one-class modules do
not depend on classifier internals.

Training and validation datasets recursively index `<split>/normal` and reject anomaly leakage.
The Deep-SVDD center is initialized from unaugmented normal training embeddings and remains fixed;
the full backbone and projection are optimized. Best selection uses mean normal-validation score,
then best and resumable checkpoints are independently calibrated from the configured validation
quantile. Versioned checkpoints store algorithm/backbone identity, preprocessing, center,
threshold, dimensions, and state; resumable checkpoints additionally preserve optimizer, history,
best objective, completed epoch, and RNG state.

Benchmarking requires `test/{normal,anomaly}`, treats anomaly as the positive class, never
recalibrates, and writes image-level metrics, predictions, ROC/PR data, plots, provenance, and a
Markdown report. Higher-is-positive binary metric calculation is shared in
`mlx.core.binary_metrics`; mode-specific artifact and presentation policies remain mode owned.

The mode-owned AWS boundary runs the same training and benchmark commands inside SageMaker.
Single training freezes one algorithm/backbone variant; `train-all` freezes every standard
backbone for `deep-svdd`, expanding both Drax fusion modes, then executes them sequentially after
one dataset extraction. Per-variant state distinguishes completed training from completed
benchmarking so a failed optional benchmark can resume without retraining a verified model.
Standalone AWS benchmarking stages an exact checkpoint through a separate managed input channel.

## Video Anomaly Detection

`video_anomaly_detection` is a first-class normal-only workflow. Its runner constructs typed
requests and presentation adapters; commands own training, benchmarking, listing, and video
inference. Mode-owned data, backbone, metric, and artifact modules remain independent of
CLI state. The default tensor flow is:

```text
[B,T,C,H,W] -> [B,C,T,H,W] -> family-aware inflated 3D backbone
              -> global 3D pool -> [B,D] -> SVDD projection -> [B]
```

`InferVideoAnomaly` treats deployment-checkpoint metadata as authoritative and only validates a
model alias when the caller explicitly supplies one. It decodes the requested video through the
shared frame-source interface, scores complete sliding windows, and optionally sends annotated
frames through an injected shared frame sink. The mode-owned renderer labels the temporal window
ending at each frame without coupling the command to OpenCV presentation calls. The command
returns a `VideoAnomalyInferenceResult` containing a video-level verdict, aggregate score/count
fields, display completion state, artifact paths, and the complete per-window timeline. Any window
whose score is strictly greater than the stored normal-validation threshold marks the video
anomalous. Display mode scores each complete window immediately so its ending frame receives the
corresponding verdict; headless mode retains batching. The command writes the same summary to
`summary.json` and retains detailed JSONL and CSV predictions; reporters own terminal rendering.

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
classification, one-class image recognition, and video anomaly models retain mode-owned
projection/checkpoint structures. The video center is initialized once from normal training
embeddings and stored as a fixed buffer.
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

Object-detection comparisons use `SubmitDetectionComparison`,
`GetDetectionComparisonStatus`, and `LaunchDetectionComparisonTest` in the mode's AWS package.
`--models` or YAML `comparison.models` supplies two or more names validated against the selected
MLX provider catalog. Submission records the ordered model list in an S3 manifest and starts
the first job. `WatchDetectionComparison` invokes `AdvanceDetectionComparison` after each
completed job to submit the next, keeping instance demand at one; a restarted watcher can
continue the manifest. Each job uses the same dataset ZIP and configured training settings. Each
container publishes validation metrics to the comparison prefix. The test command requires
all jobs to have completed and stages each model's `best.pt` and final full-state `last.pt`
in separate SageMaker channels. The last checkpoint is selected from the active recovery slot
only when its metadata reaches the configured final epoch. The evaluation-only container
benchmarks both checkpoints on the `test` split and publishes JSON and CSV results under
`{model}-best` and `{model}-last`; stored validation metrics belong to the best checkpoint.
The runner selects commands and renders records; provider benchmarking stays behind the neutral
`BenchmarkObjectDetectionModel` command. Training and comparison use the same
object-detection SageMaker image and provider adapter boundary.
The submission passes its instance type to the container. The entrypoint resolves `auto`
through PyTorch CUDA availability and raises a user-facing error when a GPU instance cannot
expose CUDA, preventing an unnoticed CPU training run.

Object-detection fine-tuning is an explicit training use case rather than provider-specific
branching. Local callers supply a required `.pt` checkpoint to
`FineTuneObjectDetectionModel`; the command validates the initialization contract and delegates
the common provider-neutral training flow. AWS callers supply an exact S3 model URI. SageMaker
stages it through a separate managed input channel, and the container passes the staged file to
the same fine-tuning command only on the first attempt. Recovery checkpoints take precedence on
Spot or manual resume, preventing the initial model from being reapplied. Run manifests and
compatibility matching treat the initial-model URI as immutable run identity.

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

`LocateBestAwsObjectDetectionModel` resolves a downloadable best checkpoint from the AWS training
YAML without requiring a job name. The mode-owned service scans completed MLX SageMaker jobs in
newest-first order, matches immutable run manifests against the configured resource prefix,
dataset, checkpoint base, provider, and model, and validates the `recovery/best.json` metadata and
`best.pt` object before returning a structured S3 location. It does not download the model or use
the SageMaker `model.tar.gz`, whose selected packaged checkpoint may differ from `best.pt`.

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

One-class image recognition exposes AWS `train`, `train-all`, `benchmark`, `status`, `stop`, and
`resume`. Its YAML separates common training values from optional benchmark values and combines
`aws.output_s3_uri` with `aws.resource_prefix` before adding logical run or benchmark IDs.
Training checkpoints, research artifacts, benchmark outputs, frozen specs, and attempt outputs
therefore remain grouped without requiring callers to construct per-run paths. Standalone
benchmark jobs are restarted rather than resumed; training jobs use per-variant rotating
full-state checkpoints and hash-verified completed artifact manifests.

Video anomaly detection exposes `train-all`, `status`, and `resume` through its mode-owned AWS
package. One SageMaker attempt extracts the dataset once and trains the frozen live 3D variant
inventory sequentially; Drax aliases expand to both fusion modes. A batch manifest records each
variant as pending, running, completed, or failed. Completed artifact directories and per-variant
rotating full-state checkpoints synchronize directly under the batch S3 prefix. The workflow is
fail-fast, and a later attempt integrity-checks completed artifacts, skips them, and restores the
active variant. Attempt model archives use a sibling prefix so old archives are not downloaded as
checkpoint input. Local single-model training and the reusable trainer remain AWS-independent.

## Shared Primitives, Requests, and Registries

`mlx.core.artifacts` contains only behavior that is identical across modes: JSON-safe value
normalization, atomic JSON/PyTorch writes, CSV serialization, and SHA-256 hashing. RNG capture and
restore live in `mlx.core.random`. Checkpoint schemas, compatibility checks, naming, plots, and
reports remain mode owned. Deep-SVDD sharing remains similarly narrow in `mlx.core.deep_svdd`;
higher-is-positive binary score metrics shared by image and video anomaly workflows live in
`mlx.core.binary_metrics`.

Image classification and segmentation expose action-specific request subclasses while retaining
their former umbrella request types for Python compatibility. `ConfigRequest` keeps unknown
public compatibility values but discards underscore-prefixed CLI bookkeeping. Commands may still
adapt a typed request to a mapping at a legacy boundary; runners are responsible for selecting
the action-specific type.
Runners that distinguish explicit options normalize their input through
`core.configuration.with_explicit_options` before adding mode defaults. CLI-provided option
metadata remains authoritative, including an empty set; a plain Python mapping treats its
supplied public keys as explicit. Normalization copies the mapping and option set, so callers'
configuration is not mutated. Typed requests continue to discard this boundary-only metadata.

Tracking, object-detection providers, image-classification custom models, temporal encoders, and
3D video backbones expose immutable registry mappings or registry value objects. Extension APIs
return a new registry for dependency injection. Historic registration calls also update their
default registry for compatibility, while exported mappings remain read-only to callers.

Metadata-only discovery uses `ListComponentNames` and structured `ComponentSummary` values.
Classification, segmentation, one-class recognition, video anomaly detection, and detection
support `ls-models --names-only` without constructing models. Existing detailed model listings
retain their parameter-count behavior. Detection providers expose optional `model_names()`
metadata; providers without that capability fail explicitly instead of loading models as a
fallback. Metadata discovery may still import a mode's framework modules; it does not initialize
models or external services.

Classification and segmentation training accept injected model registries and loss factories.
Registry injection also crosses discovery, smoke testing, checkpoint loading, inference, CAM,
benchmarking, and segmentation/saliency post-training sample generation. Batch segmentation and
saliency commands bind their default child-command factories to the same registry. An
injected model must remain resolvable when its checkpoint is reloaded; registry objects themselves
are not serialized into checkpoints. Video 3D backbone selection uses its own capability registry,
so adding a native 3D implementation does not require a classification-model registration.

Text embedding (`llama-cpp-python`, ChromaDB), legacy NLP CSV embedding (`pandas`,
`llama-cpp-python`), and Grad-CAM are optional package extras. Their adapters remain lazy and
raise actionable `MLXUserError` messages when the selected capability is not installed.

## Text Embedding and Retrieval

`text_embedding` is the canonical mode; `text-embedding` and `nlp` route to the same descriptor,
while legacy NLP CSV options select the retained compatibility command. `EmbedTextCommand` loads a
validated BEIR dataset, sends batched query and document text through `TextEmbeddingProvider`,
applies optional provider-neutral L2 normalization, streams CSV batches, and inserts corpus
vectors through `VectorStore`. The llama.cpp adapter owns GGUF loading and sequence-vector
validation. The Chroma adapter owns persistence and converts cosine distance to a normalized
higher-is-better score. Neither third-party package is imported by unrelated modes.

Embedding backend metadata and provider/store factories are resolved before output creation or
model construction. The command closes an acquired vector store even when CSV initialization
fails; artifact setup and embedding share the same cleanup boundary.

BEIR parsing produces immutable corpus, query, relevance-judgment, and dataset values independently
of embedding and retrieval. Document title/body composition and configurable query/document
role formatting live at the workflow boundary rather than in the llama.cpp adapter. Embedding artifacts
include portable copied qrels, manifests, model SHA-256, explicit normalization/prefix metadata,
and a generic representation name. This serialization boundary allows future PCA or autoencoder
tools to load, transform, export, and re-index vectors without changing embedding providers.

`BenchmarkTextEmbeddingCommand` consumes the embedding artifact directory and persistent store; it
never reloads the GGUF model. Provider-neutral metrics calculate precision, recall, reciprocal
rank, average precision, DCG, and nDCG from normalized search results and parsed qrels. A focused
writer owns aggregate/per-query tables, rankings, failures, provenance, and the deterministic
Markdown report. A future `PgVectorStore` implements the same protocol and is registered in the
immutable vector-store registry; neither command, dataset parsing, nor metric code changes.

Benchmark evaluation includes only exported queries present in the selected qrels. Unjudged
queries remain in embedding exports but are excluded from aggregate metrics and failure lists;
the benchmark summary and manifest report their count. Artifact readers validate manifest
structures and exported IDs before index access. Retrieval results must have unique known IDs,
finite best-first scores, and respect the requested depth. Chroma reload checks cosine metadata.

Text embedding accepts `pooling` (`auto`, `mean`, `cls`, `last`, `none`) and
`prompt_format` (`auto`, `none`, `e5`, `embeddinggemma`) on `EmbedTextRequest`. Both default to `auto`.
The existing immutable backend registry declares optional `supports_pooling`; opted-in factories
accept a `pooling` keyword for explicit choices. Default construction retains the single-path
factory contract. Unsupported explicit pooling is rejected before output creation.
The llama.cpp adapter alone maps named constants, omits the override for auto, validates sequence
vectors, and exposes optional `runtime_metadata()` returning `pooling_effective`, `context_length`,
and `llama_cpp_python_version`. Providers without this method remain compatible and report unknown
runtime settings. No token averaging or alternate-pooling retry is permitted.

The mode-local `RetrievalTextFormatter` receives document title/body separately and provides
legacy title/body composition with identity/custom-prefix or E5 role prefixes, or explicit
EmbeddingGemma title-aware formatting.
Prompt-format auto deliberately resolves to none without filename detection. E5 conflicts with
nonempty custom prefixes and fails before model construction. Dataset values remain unmodified.

Additive `embedding_configuration` provenance is stored in embedding/run manifests and copied into
benchmark manifests, run metadata, and JSON summaries; benchmark tables/reports expose pooling and
prompt-format choices. Runtime pooling is read through the binding's public accessor, with
`model/default` for unresolved auto and `unknown` for unresolved explicit pooling. Missing context
and version are null. Source embedding dimension and existing transformed dimensions remain distinct.
Old artifact schemas remain readable without fabricated configuration. Benchmark manifests also
retain the original embedding settings (including custom prefixes and normalization).
The CLI renders chained errors to stderr under `--verbose`; model errors retain their original
cause and actionable pooling guidance. Legacy CSV embedding rejects nondefault new options.

## Vector Autoencoders and Representation Transforms

`mlx.core.vector_transforms.VectorRepresentationTransformer` is the narrow cross-mode contract for
batch vector transformations. It exposes input/output dimensions, portable provenance, and a
numeric `transform` operation. Text embedding accepts this contract after provider embedding and
before final normalization, serialization, and indexing; it does not import PyTorch or an
autoencoder model. The text runner lazily constructs the autoencoder adapter only when `--adapter`
is supplied.

The `autoencoder` mode owns numeric CSV parsing, reconstruction training, checkpoint schemas,
model/loss registries, standalone bottleneck export, and Rich presentation. Its immutable lazy
registries support built-in aliases and explicit `package.module:DefinitionClass` references.
Both registries use the mode-owned `autoencoder.definitions` constructor; neither registry
depends on the other. Each registry validates its own definition protocol.
Definitions build models or loss modules from validated mappings, allowing custom experiments
without adding selection branches to commands. Checkpoints retain the exact architecture import
path and preprocessing contract so later encoding reconstructs the correct implementation.

Autoencoder checkpoints use restricted tensor loading, never an unsafe-pickle fallback. Before
importing architecture code, loading validates the checkpoint structure and dimensions. Built-in
references and exact references in a caller-supplied registry are trusted; other references require
`trust_checkpoint_code=True` (`--trust-checkpoint-code`). This authorizes Python imports, not a
sandbox, and must be used only for trusted code. Existing built-in checkpoints remain compatible.

Autoencoder loss definitions may expose `default_config`; the composition in `TrainAutoencoder`
merges explicit options over these defaults and persists the effective mapping. Existing
reconstruction losses keep their `(prediction, target)` interface and `model(inputs)` path.
Loss modules opting into `requires_latent` receive `(prediction, target, *, latent)` after one
`encode()` and corresponding `decode()` call. This opt-in contract requires reconstruction to
follow `decode(encode(inputs))`; dispatch is capability-based, not tied to a loss name.

The mode-owned `mse-similarity` objective combines reconstruction MSE with off-diagonal cosine
matrix matching between detached input vectors and latent vectors. Its default similarity weight
is 1.0. Positive weight requests `minimum_batch_size = 2`; zero weight uses the legacy MSE path.
Loss modules may request minimum batch size 1 (default) or 2. `MergeSingletonBatchSampler` in the
mode's data module merges trailing singletons without dropping samples. Commands validate batch
and partition sizes before output creation and use the same loss evaluator in training and
validation. Input CSVs and inference contracts are unchanged; stored total validation loss remains
the checkpoint-selection criterion, rather than a measured retrieval score.

Native classification, segmentation, and autoencoder training share scalar tensor loss validation
in `mlx.core.losses`; objective definitions and target semantics remain mode-owned.

Input normalization is part of adapter provenance. Training detects normalized MLX embedding
artifacts or accepts an explicit override; live GGUF vectors receive the same preprocessing before
`encode()`. The existing text-embedding normalization option remains a final-representation step.
Thus corpus exports, query exports, and vector-store records always share identical latent-vector
semantics, while retrieval metrics and vector-store implementations remain unchanged.

## Presentation, Errors, and Compatibility

Commands expose structured values and provider-neutral protocols. Long-running training,
benchmark, dataset-build, conversion, CAM, and embedding commands report structured
`WorkflowEvent` values; task-specific Rich tables, progress bars, prompts, and panels belong to
mode-owned `presentation.py` adapters. Rendering for shared infrastructure events may live in
`mlx.core.presentation` and is composed by those mode adapters. Compatibility functions may
attach the adapters, while direct
command construction defaults to a no-op reporter and remains suitable for Python and tests.
The shared training-metrics renderer consumes mode-configured metric definitions and writes one
persistent row per epoch, including direction-aware deltas. Commands continue to emit raw metric
values and checkpoint state; optimization direction, labels, colors, and terminal formatting stay
in the presentation layer. Segmentation and image classification compose this renderer while
provider-native object-detection progress remains owned by its provider.
Detection streaming and
tracking video execution support headless use through injected presentation boundaries:
`RunObjectDetectionStream` accepts injected detector, frame source, frame sink, renderer, and
reporter objects. The detection runner owns frame-port cleanup until command construction
succeeds, including display-setup failures. The stream command then owns both ports and attempts
sink cleanup even if source release fails; setup and execution errors continue to propagate.
`RunTrackingVideo` accepts an optional paired frame sink and tracking renderer;
without them it writes tracking artifacts headlessly. The tracking CLI injects an OpenCV sink and
a mode-owned renderer by default, while `--no-display` leaves both absent. The renderer consumes
only `TrackingFrameResult` values and draws current observations with stable track-ID colors,
boxes, identity/class/confidence/status labels, and a frame summary. User-stopped playback
finalizes partial MOT output but skips whole-video benchmarking. The CLI supplies OpenCV and Rich
adapters. Other modes keep their output
formatters in `presentation.py`; ongoing changes must move new terminal/window behavior toward
the same injected-adapter boundary rather than adding UI work to model, data, or metric modules.
Segmentation streaming reuses the lazy OpenCV adapters and frame ports in `mlx.core.streaming`;
its historic adapter names and public capture attribute remain available. The stream command
releases its injected source and closes its sink on model-setup failures as well as loop exit,
and attempts sink cleanup even if source release fails.
Segmentation's reusable visualization transforms live in `visualization.py`; only window display
and prompts remain in presentation. Its encoder consumes a `ClassificationBackboneFactory`, with
the existing image-classification implementation isolated in the default compatibility adapter
instead of being imported by the encoder itself.

Segmentation training treats `test/` as an optional qualitative-artifact boundary. A completely
absent test split does not affect training, while a partially defined or invalid paired split is
rejected before model work begins. After training, `GenerateSegmentationSamples` reloads the
best-foreground-Dice checkpoint (falling back to best validation loss), selects at most 16
deterministic samples across the sorted split, and refreshes mode-owned original,
ground-truth, prediction, overlay, and labeled-panel artifacts. Full test metrics remain the
responsibility of `BenchmarkSegmentation`.

The small segmentation DRAX comparison keeps MobileNetV3's five-stage encoder and shared U-Net
decoder. `SkipRefinedMobileNetEncoder` refines only the 1/16-resolution encoder skip through a
64-channel adapter; its registered variants use convolution-only, original DRAX, or balanced-branch
DRAX refinement. The existing final-feature DRAX MobileNet remains distinct. The optional
`DraxBlock.balanced_branch_scale` initializes convolution and attention delta scales equally;
omitting it retains existing classifier and detector behavior. `unet-compact` uses the native U-Net
with narrower widths. The mode-owned `cross-entropy-dice` loss combines cross entropy and
foreground soft Dice for sparse binary masks. These additions have stable model and loss names for
checkpoint reload, while the historical `all-small` group retains its explicit membership.

`BenchmarkSegmentation` feeds each inference batch into a mode-owned metrics accumulator rather
than retaining dataset-wide target, prediction, and probability tensors. Confusion, calibration,
loss, MCC, and configured binary-threshold statistics remain exact; bounded per-class score
histograms produce approximate ROC/PR curves and AUC/AP values with configurable resolution. Its
memory use therefore depends on batch size, class count, and histogram resolution rather than
dataset pixel count.

`TrainAllSegmentationModels` freezes the sorted segmentation registry for one sequential,
fail-fast local run. It requires scratch initialization, a complete test partition, and a new or
empty directory output. Each model owns a child directory; its lowest-validation-loss checkpoint
is benchmarked on test while its best-Dice checkpoint remains the source of qualitative training
samples. Root `all-models.json` and `leaderboard.csv` artifacts are refreshed after every completed
model and rank finite test results by mean foreground Dice. Commands release model references and
available accelerator cache between variants. Segmentation batch training is not a SageMaker
workflow.

Invalid CLI input, absent files, unsupported actions/providers, missing optional libraries, bad
dataset layouts, and camera/video failures raise `MLXUserError`. Model-internal invariant failures
may use `ValueError` or `RuntimeError` when they indicate programmer errors rather than recoverable
user input. `MLXAbort` is reserved for intentional cancellation.

Compatibility functions such as `train_image_classification`, `infer_segmentation_image`, and
`convert_object_detection_model` construct a typed request or provider command and call
`execute()`. Former Ultralytics-owned detection and tracking imports are re-exported from their old
paths. Compatibility wrappers must not accumulate new business logic.

## Extension Review and Tutorials

The [extension inventory](docs/extensions.md) records current ownership, selection, discovery,
testing, and remaining limitations. [Executable tutorials](docs/tutorials/README.md) demonstrate
local extension without changing runners. Registries remain mode-owned; shared import-reference
validation and JSON option loading in core store no registrations. Tracking reuses those helpers.
Classification feature-head removal is catalog metadata rather than another model-name dispatch.

Classification, segmentation, saliency, autoencoder, text embedding, one-class recognition,
and video anomaly package exports are lazy compatibility surfaces. Importing their requests or
registries does not load workflow commands, runners, or presentation merely through package
initialization. Public command names resolve on demand to their owning implementation modules.
Inspecting a registry must not import visualization or optional provider integrations. Detailed
parameter-count listings may construct models; metadata-only discovery must not do so.
Classification and segmentation runners normalize loss options before constructing typed training
requests; legacy Python factories continue accepting JSON paths or mappings.

LibreYOLO incremental neural adapters remain provider-owned, distinct from inference adapters.
Before adapter training, the integration verifies that either the train signature or the trainer's
configuration fields explicitly declare all requested adapter controls. Generic `**kwargs` alone
is not evidence of support. Unsupported versions fail with `MLXUserError` rather than silently
performing full-model training. MLX does not invent a neural-adapter registry or provider hook.

Detection inference accepts an optional `ObjectDetectionRequest.adapter` path through
`--adapter`. The LibreYOLO provider constructs `LoadAdaptedYOLOX`, which reads the adapter's
`config`/`state` artifact, verifies its standard YOLOX model identity and exact foundation
SHA-256 against `--model-path`, then invokes `ApplyYOLOXAdapter` before wrapping the model
for normalized frame prediction. Saved method, rank, reduction, alpha, head-training flag,
and injection topology determine restoration; original study directories are not required.
`ApplyYOLOXAdapter` and `resolve_adapter_targets` are also used by study reconstruction to
keep injection and strict trainable-state loading at one provider boundary. Recorded LoRA
and hybrid convolution paths are honored, including historical hybrid topology; feature
adapter paths are checked against the provider's target policy. Camera/video streaming
and presentation remain unchanged. ONNX foundations, incompatible checksums, malformed
adapter artifacts, and Ultralytics inference with `--adapter` raise `MLXUserError`.
This option does not apply standalone adapters during detection training, conversion, or
benchmarking; adapter-study actions retain their existing method-selection semantics.

The YOLOX-L feature-adapter study is a separate local research workflow. LibreYOLO owns the
generic feature adapters, registry, injection, YOLOX target policy, standard model, and strict
checkpoint compatibility. MLX owns DAWN Parquet-to-YOLO conversion, the seed-42 stratified split,
CUDA-required execution, batch calibration, training/evaluation orchestration, memory/timing,
per-seed artifacts, and aggregate statistics. The workflow loads the unmodified foundation state
dict before injection and defaults to the three PAFPN outputs at strides 8, 16, and 32. It never
silently falls back from a requested CUDA device or changes a configured batch after OOM.

`PrepareDawnAdapterDataset.execute()` writes only below the caller-supplied dataset destination;
experiment output contains metadata and run artifacts, not dataset copies. `RunAdapterExperiment`
uses LibreYOLO's existing AMP and nominal-batch accumulation contracts, holds the physical/effective
batch fixed across methods, verifies identity before training and frozen/trainable tensors after
training, and records failures without retrying altered conditions. `GenerateAdapterReport` emits
analysis-ready CSV/JSON plus descriptive mean, standard deviation, paired differences, and 95%
intervals. Single-seed output is explicitly exploratory.

The `drax-hybrid` method is implemented only in LibreYOLO: it shares LoRA's
convolution target selector (26 dense 1x1 neck convolutions for YOLOX-L), combining
exact LoRA weight updates and compressed two-scale spatial bypasses. MLX forwards
rank/reduction/alpha and records actual placement. Prediction reconstruction uses
the saved hybrid injection paths so earlier three-projection checkpoints remain
loadable. Experiment resume rejects a completed hybrid with a different placement.
`LoadAdapterBaseline` owns read-only reuse through `--baseline-study`: it verifies
the complete seed matrix and matching checkpoint, split, precision, batch,
optimizer and epoch conditions, then pins the source metadata checksums in
`baseline.json`. Training reuses frozen results; reports read the prior runs by
reference. Slice caching verifies source checkpoint hashes, evaluator settings,
ground truth and prediction checksums before copying small prediction artifacts
into the new analysis directory. Datasets and model checkpoints are not copied.
`--comparison-method` selects the candidate for both aggregate and slice reports;
its default remains `drax`. Existing baseline outputs are never modified.
Frozen BN buffers are verified against the non-EMA training checkpoint alongside
parameters. Hybrid gradient observation uses hooks without CPU tensor transfers.

Post-training targeted evaluation is split into two command boundaries. `CacheAdapterSlicePredictions`
strictly reconstructs each validation-selected model, requires CUDA, verifies unsliced COCO metrics
against the original run, and stores checksum-addressed per-image predictions without invoking
training. `GenerateAdapterSliceReport` consumes those caches on CPU to report DAWN weather and pooled
weather groups, COCO object sizes, classes, class-weather intersections, and validation-defined
frozen-baseline difficulty. Seed-paired Drax comparisons and clustered image-bootstrap intervals are
analysis outputs; low-support weather slices remain explicitly descriptive.

## Testing and Change Rules

### Inference-only target-domain adapter transfer

`object_detection.zero_shot` owns the `adapter-zero-shot` and `adapter-zero-shot-report`
actions. The composition boundary reads an explicit study JSON and injects the CUDA
provider and strict foundation verifier into `RunTransferStudy`. `SnapshotTransferModels`
copies validation-selected states into a hash-verified relative-path bundle; source runs
remain read-only. `ReconstructAdapterModel` is the common provider reconstruction command
used by both legacy DAWN slices and target-domain inference, including head-only state
merging and recorded hybrid placements.

`LibreYOLOTransferEvaluator` is the only new torch/LibreYOLO integration boundary. It
reuses native detection validation and adds synchronized FP32 forward timing, explicit
calibration and batch-one latency samples. `ScoreTransferPredictions` matches native COCO
annotations once, retains crowd/area semantics, and re-accumulates weather/sequence/video
slices from those matches. JSON prediction/match caches and completion receipts permit
verified resume without altered batch settings. No training or target-driven selection is
part of this workflow. `GenerateTransferReport` and `GenerateTransferGallery` consume
caches without CUDA, with complete seed-paired superiority statistics and all-image boxes.
`core.paired_superiority` supplies two-sided paired t intervals/tests, exact sign flips and
Holm adjustment independently of the existing equivalence-analysis API. Five training
seeds are the inference unit; correlated target frames are not treated as independent seeds.

The CLI requires a dedicated output path; runtime dataset/model paths remain explicit
configuration, never package constants. Source snapshots, hashes, environment, fixed
protocol, pilot, calibration and run-level failures stay under that output directory.
See `docs/object_detection/zero-shot-adapters.md` for configuration and reconstruction.

- Unit-test commands with fake models, providers, reporters, frame sources, and frame sinks.
- Test each provider against the neutral contract; provider-independent tests must not import its
  third-party package.
- Keep runner tests focused on defaults, action dispatch, request construction, and presentation
  wiring.
- Verify user-facing failures at integration boundaries and run `python -m pytest -q` before handoff.
- Any code or configuration change must review this document. Update it in the same change whenever
  command inventory, package ownership, dependencies, interfaces, provider behavior, or data flow
  changes.

## Corpus-Adapted Autoencoder Retrieval Experiments

`text_embedding` adds `PrepareRetrievalDatasets` (`prepare-datasets`) and
`BenchmarkAutoencoderRetrieval` (`benchmark-autoencoders`). The latter accepts a typed
`AutoencoderRetrievalRequest`; its runner injects `TrainAutoencoder` and
`AutoencoderRepresentationTransformer` factories, keeping torch/model ownership in `autoencoder`.
The experiment command coordinates existing embedding and benchmark commands, the
provider-neutral `TransformEmbeddingArtifacts`, and `AnalyzeAutoencoderRetrieval`. It does not
implement training or model-provider selection. Reports remain artifact/presentation concerns.
The CLI uses action-specific typed defaults without changing standalone command defaults.

`retrieval_datasets` owns the immutable, revision-pinned `laptop-ae-v1` source inventory and an
injected Hub/Parquet boundary. Downloads convert into temporary BEIR directories before validated
atomic publication. Nano judgments are binary and the source storage split named `train` becomes
`qrels/test.tsv`; manifests make this evaluation mapping explicit. Corpus/query IDs are preserved,
including empty Nano corpus text through the loader's opt-in `allow_empty_documents` policy.
No judgments or documents are silently discarded. Existing source directories require matching
revision and content hashes. Missing or malformed inputs raise actionable `MLXUserError` values.

The embedding formatter adds explicit `embeddinggemma` selection using separate title and body
fields; legacy none/E5 composition is unchanged. Backend registry capability
`supports_context_length` allows optional `EmbedTextRequest.context_length`; the llama.cpp adapter
sets context/batch capacity and records token-prefix truncation provenance. Omitted context values
preserve binding defaults. Optional provider `close()` is called on success and failure after
runtime metadata capture. No filename inference or pooling fallback is introduced.

The immutable vector-store registry adds `exact`, a float32 cosine index with memory-mapped
vectors, bounded score chunks, and ID-stable ties. `BenchmarkTextEmbeddingRequest` adds opt-in
`exclude_self_matches` and records the policy in benchmark artifacts. The cached transform command
uses `VectorRepresentationTransformer` to transform both exports, normalize final vectors, and
rebuild an index while retaining original model/dataset provenance. It never imports autoencoder
models or reruns text embedding.

`AutoencoderTrainRequest.minimum_batch_size` permits callers to raise the batching minimum to two
(default one), coordinating the same singleton-merging policy across loss ablations. Loss-declared
minimums remain authoritative lower bounds. Model initialization, partitions, and ordering remain
seeded; validation objective remains the checkpoint-selection rule.

`ExperimentStages` owns experiment identity, content verification, duration recording, failure
artifacts, and atomic stage publication. Configuration, source/model/dataset hashes and versions
must match for resume. Partial stages restart; modified completed stages are rejected. Acquisition
and resume are explicit options, and no output is printed by reusable workflow code. Sequential
execution bounds resources; callers must not share an output directory concurrently.

Statistical inference uses dataset-level paired mean deltas after averaging query and seed
repetitions. One-sided t-tests at the negative non-inferiority margin receive Holm correction over
the primary loss/dimension family. Secondary similarity-versus-MSE tests form a separate family.
Intervals, multiplicity policy, assumptions, seed variation, and transductive/Nano limitations are
reported explicitly. Missing cells prevent analysis; degenerate variance is inconclusive. This
boundary does not redefine existing retrieval metrics. See
[`docs/retrieval-autoencoders.md`](docs/retrieval-autoencoders.md) for the reproducible protocol.

## Configured Autoencoder Retrieval Experiments (v2)

The existing `benchmark-autoencoders` command accepts an optional `--experiment-config` JSON
recipe. Legacy flags and layouts remain supported when no recipe is supplied. With a recipe,
`BenchmarkConfiguredAutoencoders` coordinates named variants through injected training,
validation, embedding, and transformation collaborators; it does not import torch or own model
selection. Shared protocol validation lives in `experiment_validation`. Recipes separate training
widths from evaluation widths, so one ordered checkpoint or PCA fit can serve several outputs.
`--dry-run` validates the recipe, local datasets, model, and any requested embedding source without
creating output. Explicit legacy variant flags conflict with recipes; explicit runtime training
flags override recipe training defaults. Resolved configuration is hashed for resume.

The autoencoder registry adds `simple-spectral` and `ordered-simple`. Both reuse the existing
GELU MLP; spectral decoder linear layers use parametrizations with five power iterations. Ordered
models declare supported prefixes but retain deterministic full-width `encode()` and `decode()`.
`objectives` isolates scalar/component evaluation from training: `ReconstructionObjective` wraps
existing two-argument and latent-aware losses; `OrderedReconstructionObjective` masks a prefix
with its own seeded generator in training and averages every declared prefix during validation.
Ordinary losses retain their public signature. Ordered models initially permit MSE only.

`regularization` owns `mse-least-volume` and `mse-covariance`, including validation and training-only
calibration. Least Volume requires a constrained-decoder capability. The covariance objective is
an explicitly paper-inspired mean squared off-diagonal feature-covariance penalty, not a paper
reproduction or a sample-similarity loss. Both persist calibration scale/raw terms, formula version,
and validation components alongside seed and split hash. Zero coefficient follows MSE without
calibration. Checkpoint schema remains compatible; additional metadata is optional for old models.
The representation adapter accepts a supported output prefix and records training and output widths.

`mlx.core.partitions` owns the shared seeded row split and split hash used by AE training and PCA.
It preserves the existing torch permutation and loader-generator advancement; torch is imported
lazily at this narrow boundary. Corpus text loading and vector CSV parsing remain mode-owned.
`compression_controls` owns training-only full-SVD, centered/unwhitened PCA and native vector
truncation. Both implement the existing vector-transform contract. Export normalizes final vectors
and records the actual representation name, including PCA and truncation rather than an AE label.

`ValidateEmbeddingSource` verifies external original stages against content, dataset, model, and
embedding-protocol/runtime provenance. External stages remain read-only; a v2 manifest records
resolved references and their hashes. Missing or mismatched requested sources fail rather than
silently recomputing embeddings. v2 manifests and atomic stages distinguish a training run from
its evaluations. Training costs and model/PCA bytes are reported once per run, while vector bytes,
transformation and search times are per evaluation. PCA and deterministic truncation have explicit
kind/seed semantics; deterministic controls are not replicated to inflate statistical sample size.

`SelectAutoencoderExperimentSettings` is exposed as `select-autoencoder-settings`. It verifies a
complete pilot and selects coefficients solely from held-out document reconstruction ratios against
matched controls; it never reads retrieval metrics. It emits frozen confirmation settings and an
auditable selection record. `AnalyzeConfiguredAutoencoders` requires every expected cell and reports
prespecified non-inferiority and superiority families with separate Holm corrections. The analysis
unit remains the dataset. See `docs/retrieval-autoencoders.md` for protocol, counts, and launchers.

### Orthogonal tied projections and supervised experiment processes

The autoencoder registry adds `orthogonal-tied`, a bias-free linear encoder whose single
weight is reused transposed for reconstruction. Its model owns training-only uncentered
SVD initialization, signed reduced-QR projection after optimizer updates, and orthogonality
diagnostics. `TrainAutoencoder` recognizes optional `initialize_from_training_values`,
`project_parameters_`, and `training_diagnostics` model capabilities. Initialization receives
only the already-normalized training partition, before loss calibration and optimization.
Initialized models receive an epoch-zero evaluation eligible for best-checkpoint selection;
legacy models retain their original epoch sequence. Checkpoint restoration never initializes
from data, and initialization provenance is persisted separately from construction options.

`mse-cosine` is a reconstruction-only loss: MSE plus `cosine_weight / input_width` times
per-row cosine error. It creates no sample-pair matrices. Reconstruction objectives collect
component diagnostics from both latent-aware and reconstruction-only losses.

Configured experiments additionally accept `kind: svd`. `FitLinearProjection` and
`LinearProjectionVectors` own centered PCA and uncentered SVD behind the same fitted
projection boundary. `FitPcaVectors` and `PcaVectors` retain their centered-only compatibility
interfaces. Controls fit training rows only and can evaluate prefixes of one fitted basis.
SVD fit counts, artifacts, and provenance remain distinct from PCA. Selection metadata is
retained in reports; two-sided difference tests have a separate Holm family from
non-inferiority and prespecified secondary tests. Confidence intervals remain unadjusted.

The orthogonal experiment recipe is exploratory: the existing ten datasets have already
informed model selection. Its launch scripts use `scripts/experiment-job.py`, a POSIX-only
process supervisor outside ML workflow code. Foreground and background execution share
an advisory per-output lock. Detached workers use nohup, a new session, disconnected stdin,
timestamped logs, PID/child PID metadata, explicit thread settings, and exit-status records.
The experiment child inherits the lock descriptor so a lost supervisor does not permit a
duplicate writer. PID files are informational; lock ownership determines whether a launch
is allowed. Existing experiment stage identity and hash checks govern resume.

`retrieval_datasets` also provides the pinned `held-out-nano-v1` source inventory for
ClimateFEVER, FEVER, and NQ. It uses the same Parquet-to-BEIR conversion command and
source-manifest verification as `laptop-ae-v1`. The held-out confirmation recipe
adds a matched `simple` MSE autoencoder and compares it with the orthogonal MSE model,
PCA, and full Gemma embeddings; SVD and truncation remain contextual controls.
`AnalyzeReductionMetricComparison` reads completed baseline and per-seed metric CSVs,
requires complete 512D cells, averages seeds within datasets, and performs five
separate eight-metric paired-test families with Holm correction. Its frozen margins
and variant identities are supplied by the caller; analysis writes outside the
immutable benchmark stage. The script in `scripts/analyze-reduction-metrics.py` is
only a thin entrypoint for the experiment protocol.

### Answer-level RAG reduction experiments

`PrepareRagEvaluationDatasets` converts pinned SQuAD v2, HotpotQA, and FinQA
sources into BEIR corpora, relevance judgments, answer labels, and a frozen
generation subset. Source hashes and converted files are verified on reuse.
The existing configured retrieval command produces embeddings, reduction
checkpoints, exact-search rankings, and retrieval metrics for each embedding
model. The `qwen3` retrieval formatter applies the upstream query instruction
while leaving document text unprefixed.

`EvaluateRagReduction` reads only completed rankings and injects a generator
through its narrow `generate(prompt)` interface. `LlamaCppRagGenerator` owns
the local GGUF chat integration and loads the model lazily. The evaluator
uses identical prompt and context rules for every reduction, caches identical
prompts across variants, and returns answer and gold-evidence proxy metrics.
`AnalyzeRagReduction` calculates equal-dataset means and paired question
bootstrap intervals conditional on the fixed datasets. CLI scripts under
`scripts/` supply protocol paths and dependencies; RAG orchestration and
statistics remain mode-owned rather than in the runners.

## YOLOX MobileNet Drax architecture ablations

LibreYOLO's YOLOX-Drax-MobileNetV3 wrapper owns the optional `architecture_variant`
constructor preset. MLX extends the existing immutable model inventory with three
L-size ablation aliases ending in `-refine-p3p4`, `-spp-p5`, and `-balanced-drax`,
plus the integrated M-size `-pyramid-drax` candidate used for parameter-constrained
comparisons. Its lazy factory injects the preset and rejects provider versions that
would silently ignore it. Neural blocks, preprocessing, strict checkpoint reconstruction, class-count
rebuilding, and distributed worker reconstruction remain provider-owned. The
unchanged family is the default; variants require the matching LibreYOLO build.

`TrainObjectDetectionRequest` adds optional `workers`, `eval_interval`,
`no_aug_epochs`, and `patience` values, forwarded by the LibreYOLO boundary after
non-negative integer validation. Omitted Python values preserve provider defaults.
CLI `--workers` retains its existing default of four; `--eval-interval`,
`--no-aug-epochs`, and `--patience` default to unset. Other providers retain their
existing behavior. Research experiment supervisors compose these training and
benchmark commands externally; they do not embed model logic in MLX runners.
