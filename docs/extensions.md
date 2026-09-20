# Extension inventory and second-pass review

The governing rules are in [ARCHITECTURE.md](../ARCHITECTURE.md).
[Executable tutorials](tutorials/README.md) demonstrate small implementations.
Paths below are relative to `mlx/modes/` unless stated otherwise.

## Zero-scatter inventory

GOOD means normal implementations stay local. ACCEPTABLE means the listed deliberate constraints
remain. These are extension-locality judgments, not a score of model quality. Registry injection
is a Python API; a local alias is not automatically installed into a separate CLI process.

| Category / status | Contract and implementation | Registration / construction | Configuration and discovery | Tests / unrelated edits |
| --- | --- | --- | --- | --- |
| Tracker — GOOD | `object_detection/tracking/protocols.py`: tracking lifecycle and neutral frame values; `algorithms/` | Immutable `TrackerRegistry`, `CreateTrackingAlgorithm`; explicit class references | `--tracker`, JSON options, `ls-trackers` | Tracking contract tests and tutorial reset/update exercise; no CLI/command edits |
| Detection provider — GOOD | `object_detection/providers.py`: provider capabilities; provider-owned adapters return neutral detections | `ProviderRegistry.register`, lazy factory references | `--provider`; metadata `ls-models --names-only` | Fake provider tests and integration tests; default alias registration only |
| Detection inference adapter — GOOD | `DetectionAdapter.predict` in `object_detection/models.py`; provider adapter modules | Provider creates detector, or stream command receives it directly | Provider/model request; no separate adapter catalog | Empty-detector tutorial runs stream with fake source/sink; no tracking/stream edits |
| Detection neural adapter — ACCEPTABLE | Provider-specific feature-module attachment, shape/freeze/checkpoint policy | LibreYOLO provider owns selection; MLX validates capability before forwarding flags | Incremental-adapter flags; no MLX neural registry/discovery | Provider signature/config fakes; real attachment tests belong to supporting provider; unsupported versions reject |
| Classification model — GOOD | `image_classification/models/`: builder returning module; optional feature adapter | `models/standard.py:StandardModelRegistry`; built-ins in `catalog.py` | `--model` alias/import reference; `ls-models` and names-only | Model/command tests and mean-classifier tutorial; no runner edits; feature-based workflows additionally need feature adapter |
| Segmentation model — GOOD | `segmentation/models/`: builder with output class count | `models/registry.py:SegmentationModelRegistry` | `--model` alias/import reference, listing; registry propagated across reload and batch commands | Pixel-model train/save/infer test; no inference, sampling, or runner edits |
| Saliency model — GOOD | `saliency_mapping/models/`: builder producing single-channel logits | Mode-owned `SaliencyModelRegistry`; built-ins use `compatibility.py` | `--model` alias/import reference, detailed listing and explicit groups | Pixel-model train/save/infer test; no segmentation registration or command edits |
| One-class algorithm — ACCEPTABLE | `image_recognition_oc/algorithms.py:OneClassAlgorithm` lifecycle, calibration and checkpoint semantics | Immutable algorithm registry; injected algorithm/backbone collaborators | `--model`, `--backbone`, `ls-models` | One-class fake algorithm tests; Python loaders must bind the same registry; no generic loss registry for intrinsic algorithm objective |
| Temporal encoder — ACCEPTABLE | `video_anomaly_detection/models/temporal.py:TemporalEncoder` | Immutable `TemporalEncoderRegistry`, model factory | Temporal encoder option; registry keys available in Python (no dedicated discovery action) | Video factory/contract tests; bind model/checkpoint factory for Python experiments; CLI built-in names require local registration |
| Video 3D backbone — ACCEPTABLE | `video_anomaly_detection/models/backbone3d.py`: module with dimension contract | `Backbone3DRegistry`, independent of classification catalog | `--model`, video names-only listing | 3D backbone and video tests; custom checkpoint loader must use same bound factory; no classification registration |
| Selectable loss — GOOD | Native classification/segmentation loss modules; AE definition returns scalar tensor loss | Mode-owned catalogs, `core.losses.build_scalar_loss`; AE `ReconstructionLossRegistry` | `--loss`, `--loss-config`; AE `ls-loss-functions`; native modes `ls-losses`; injectable factories | Scalar/gradient checks and custom AE loss training; no training branches |
| Retrieval metric — GOOD | `text_embedding/metrics.py`: ranking/judgment function | `metric_registry.py:RetrievalMetricRegistry` | Inject registry into benchmark; `--metrics`, cutoff config, mode-local `ls-metrics` | Hand calculations and custom metric tutorial; no store/command changes |
| Embedding backend — ACCEPTABLE | `text_embedding/embedding/protocol.py:TextEmbeddingProvider` | Immutable `EmbeddingBackendRegistry`, factory; llama.cpp adapter | `--embedding-backend`, `ls-embedding-backends`; current CLI model contract remains local GGUF | Fake provider tests; generic Python DI works; a non-file/remote CLI workflow needs an intentional input-contract change |
| Vector store — GOOD | `text_embedding/vector_store/protocol.py`: add/query/close, best-first scores | Immutable `VectorStoreRegistry`, lazy factory | `--vector-store`, `ls-vector-stores`, persisted provider manifest | Chroma fake and optional persistence tests; future pgvector adds adapter + registration, not command/metrics changes |
| Autoencoder — GOOD | `autoencoder/architectures/`: encode/decode/forward and dimensions | `model_registry.py:AutoencoderRegistry`; definition constructs module | `--model`, dimension/JSON options, `ls-models` | Tiny-model shape test, loss train/reload test; custom checkpoint reference needs caller registry or explicit trust |
| Dataset — ACCEPTABLE | Mode-owned loaders and immutable values; `core.datasets` stages local/S3 input | Inject loaders/services where supported; format selection remains mode-specific | Dataset flags and typed requests | Loader validation tests; new format may require mode-owned loader changes, not a speculative universal dataset registry |
| Representation transform — GOOD | `core.vector_transforms` numeric protocol, AE adapter | Inject transformer; text runner binds optional AE adapter | `--adapter`, normalization and representation provenance | Vector contract and AE integration tests; future PCA needs no metrics/store changes |
| Spatial transforms — ACCEPTABLE | Mode-owned augmentation policies | Small fixed mode-local selection | Transform option | Existing data tests; not promoted to a global registry for three fixed policies |
| Export/inference backend — ACCEPTABLE | Detection provider convert/create-detector capabilities; ONNX/OpenCV boundary modules | Provider factories, injected detector and frame ports | Provider, format and input flags | Provider conversion/stream tests; backend-specific capabilities stay provider-owned |

## Findings and chosen resolutions

| Location | Concrete issue | Principle / severity | Resolution |
| --- | --- | --- | --- |
| Classification/segmentation/saliency commands | Registry injection stopped at discovery, smoke, reload or grouped workflows | Locality / high | Forward the same registry across child commands and checkpoints; test real tiny round trips |
| Saliency models and data | New saliency architecture required segmentation participation; scattered cross-mode imports | Ownership / high | Independent saliency registry; deliberate compatibility gateway for existing shared behavior |
| Text embedding benchmark | Unjudged exported queries diluted metrics; malformed manifests/rankings reached index code | Correctness / high | Evaluate judged cohort, record exclusions, validate schema/counts/IDs/order/finiteness and cosine metadata |
| Autoencoder checkpoint loading | Pickle loading and checkpoint-supplied import references were implicit trust boundaries | Explicitness / high | Restricted tensor loading; exact caller registration or explicit code-trust flag before external imports |
| LibreYOLO training | Generic kwargs could silently accept unsupported adapter controls | Correctness / high | Require signature or trainer-config evidence, otherwise actionable error before training |
| Mode package exports | Registry import eagerly pulled visualization modules | Portability / medium | Lazy compatibility exports; subprocess import blockers protect optional boundaries |
| Tracking options/import loading | Reimplemented existing reference validation and JSON-object loading | Understandability / medium | Reuse narrow core helpers, preserve tracker lifecycle/registry |
| Classification feature heads | Duplicate model-name selection beside catalog | Locality / medium | Head-removal paths become catalog metadata |
| Native loss factories/training | Selectable losses could return vectors, nonfinite values or detached tensors | Testability / medium | Shared scalar contract checks; target semantics remain mode-owned |
| Tutorials | Extension mechanics lacked executable end-to-end evidence | Discoverability / medium | Seven executable tutorials and deliberately tiny reference components |

## Core and abstraction decisions

Keep core commands/events, request compatibility bridge, exceptions, artifact primitives,
configuration/reference loading, binary metrics, vector transforms, dataset staging, and AWS
lifecycle infrastructure: multiple conceptually different modes use each for the same purpose.
Shared scalar-loss validation is justified by three native training modes. Do not move checkpoint
schemas, research reports, saliency objectives, model catalogs, or provider adapter attachment to
core. Existing feature and streaming ports are intentional CV boundaries, not universal ML types.

Keep domain-specific registries separate. Tracker lifecycle metadata, architecture definitions,
and vector-store factories do not have identical semantics. There is no filesystem scanning,
decorator registration, universal service locator, or new plugin framework. Factories construct;
commands coordinate. Legacy global registration entrypoints remain compatibility-only; new Python
examples always create independent immutable registries.

Keep action-specific requests and legacy mapping bridges: removing them would break callers.
Likewise retain task-specific results rather than introduce an unproven universal ExperimentResult.
Metadata discovery may import torch/framework definitions; it must not initialize models,
external services, or optional providers. Detailed parameter-count discovery intentionally builds
models. Existing training engines remain mode-owned where objectives/checkpoints differ.

## Extension exercises

| Exercise | Implementation | Registration/configuration | Test | Orchestration edits |
| --- | --- | --- | --- | --- |
| Tracker | Reuse minimal `DetectionAsTrack`, or copy into a local algorithm module | One `TrackerRegistry.register` call; options mapping | Tutorial update/reset contract | None |
| Autoencoder | `architectures/tiny.py` demonstrates minimal encode/decode/forward | One architecture definition reference; dimensions | Tutorial shape + train/reload exercise | None |
| Loss | `examples/extensions/absolute_error.py` | One local loss-registry entry (or direct reference) | Scalar/backward tutorial + AE training | None |
| Segmentation/saliency | `examples/extensions/pixel_model.py` | One entry in the owning registry | Train, checkpoint, reload, infer | None |

## Remaining deliberate limits

- Legacy global registration APIs retain historical process-wide behavior. Removing them needs
  deprecation; do not use them for new experiments.
- Custom checkpoint architecture code must be installed/importable; registries are not serialized.
  AE trust controls are not a general checkpoint sandbox for all existing CV providers.
- Video/one-class Python experiments bind registries into checkpoint/model factories explicitly.
  This is explicit composition, not automatic process-wide plugin installation.
- Neural adapter implementation and pretrained-weight compatibility require a supporting provider.
  MLX's forwarding tests do not prove a provider's neural attachment algorithm.
- Current text CLI embedding input remains GGUF-focused; remote backends warrant a separate,
  explicit configuration design if introduced.
- Spatial augmentation selection and mode-specific training engines remain intentionally separate.

No new runtime dependency or checkpoint schema migration is introduced by this refinement.
