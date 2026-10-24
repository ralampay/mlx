# Workflow modes

Package: `mlx.modes`. Each child package owns a task-specific workflow; this namespace
is not itself a CLI mode and does not provide a universal training API.

[All module guides](README.md) · [CLI guide](cli.md) · [Architecture](../ARCHITECTURE.md)

## Contents

- [Choose a workflow](#choose-a-workflow)
- [Shared workflow conventions](#shared-workflow-conventions)
- [Discovery and extension](#discovery-and-extension)

## Choose a workflow

| Goal | Module and guide | Primary data |
| --- | --- | --- |
| Locate objects with bounding boxes | [Object detection](object_detection/README.md) | Provider-supported detection datasets, images/video |
| Maintain object identities over time | [Tracking](../TRACKING.md) | Video, detector results, optional MOT ground truth |
| Predict labels or compare image pairs | [Image classification](image_classification/README.md) | Label-organized image directories |
| Learn normal images and flag anomalies | [One-class image recognition](image_recognition_oc/README.md) | Normal-only training/validation images |
| Detect anomalous video clips | [Video anomaly detection](video_anomaly_detection/README.md) | Normal training clips; labelled evaluation sequences |
| Assign semantic classes to pixels | [Segmentation](segmentation/README.md) | Paired images and class-index masks |
| Estimate continuous foreground saliency | [Saliency mapping](saliency-mapping/README.md) | Paired images and grayscale saliency masks |
| Embed and evaluate document retrieval | [Text embedding](text-embedding.md) | BEIR corpus, queries, and relevance judgments |
| Compress fixed-width vectors | [Autoencoder](autoencoder.md) | CSV rows containing numeric embedding vectors |
| Maintain historical CSV embedding integrations | [Legacy NLP](nlp.md) | CSV text column and local GGUF model |

Classification requires labels or pair semantics; one-class recognition learns only normal
examples. Segmentation masks encode classes, while saliency masks encode continuous foreground
targets. Tracking consumes detections and maintains state across frames; it is not another
detector implementation. Choose the data contract before selecting a model.

## Shared workflow conventions

Most modes provide typed request objects and command classes with `execute()`. Use these APIs
for Python integration and inject models, datasets, reporters, or storage where the command
supports them. Runners normalize configuration, select actions, construct dependencies, and
attach presentation. The exact collaborator contracts and result shapes remain mode-specific.

Training, inference, and benchmarking are separate operations. Checkpoints and artifacts are
not interchangeable between providers or modes unless an explicit adapter supports the
conversion. Preserve the selected provider and model metadata when loading a checkpoint.

Some image/video modes support [S3 ZIP staging](s3-dataset-training.md) and documented AWS
execution. Support for an action locally does not imply support on SageMaker. Autoencoders
and retrieval workflows use their own vector/text input contracts.

## Discovery and extension

Use the [CLI guide](cli.md) for aliases, effective defaults, JSON behavior, and metadata listing.
Mode-specific guides below it document datasets, artifacts, and validation. Use
[the development guide](development.md) for new actions, and the
[extension tutorials](tutorials/README.md) for models, trackers, losses, and metrics.

The mode descriptor inventory lives in `mlx.cli_routing`; adding a directory beneath
`mlx.modes` does not automatically register a CLI mode. Keep new model-family behavior in its
own mode and reuse [core infrastructure](core.md) only when its semantics are genuinely shared.
