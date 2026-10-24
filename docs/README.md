# Documentation

## Start here

- **Install and verify MLX:** [installation](../README.md#installation), including base,
  provider-specific, and development environments.
- **Find a command:** [CLI overview](../README.md#command-line-interface), then
  `python -m mlx --help` for current options and actions.
- **Train from S3:** [dataset ZIP guide](s3-dataset-training.md) for supported image/video
  modes, archive layouts, caching, provenance, and AWS differences.
- **Develop a feature or fix:** [development guide](development.md) for local setup,
  checks, CLI additions, and Python composition.
- **Add a model, loss, tracker, or adapter:** [extension inventory](extensions.md)
  and [executable tutorials](tutorials/README.md).
- **Understand ownership and contracts:** [ARCHITECTURE.md](../ARCHITECTURE.md).
- **Run retrieval experiments:** [autoencoder retrieval](retrieval-autoencoders.md).

## Main modules

Start with a shared-layer guide or select a task-specific package:

| Mode Package | Purpose | Docs |
| --- | --- | --- |
| CLI modules | Parsing, defaults, help, dispatch, and result handling | [CLI and routing](cli.md) |
| SAR | ASF Sentinel-1 discovery, downloads, and offline preparation | [SAR ingestion](sar.md) |
| `mlx.core` | Shared workflow infrastructure and contracts | [Core infrastructure](core.md) |
| `mlx.modes` | Task selection and shared workflow conventions | [Workflow modes](modes.md) |
| `mlx.modes.object_detection` | Provider-neutral detection, Ultralytics/LibreYOLO integrations, and tracking | [Object detection](./object_detection/README.md) |
| `mlx.modes.object_detection.tracking` | Tracking, MOT benchmarking, and replay export | [Tracking](../TRACKING.md), [custom trackers](../CUSTOM_TRACKING.md) |
| `mlx.modes.image_classification` | Image-classification workflows for one-shot and standard classifiers | [Image classification](./image_classification/README.md) |
| `mlx.modes.image_recognition_oc` | Normal-only still-image one-class recognition with replaceable backbones and SageMaker execution | [One-class image recognition](./image_recognition_oc/README.md), [AWS execution](./image_recognition_oc/aws-sagemaker.md) |
| `mlx.modes.video_anomaly_detection` | Normal-only clip-native 3D CNN, Deep SVDD, and sequential SageMaker workflows | [Video anomaly detection](./video_anomaly_detection/README.md), [AWS training](./video_anomaly_detection/aws-sagemaker-training.md) |
| `mlx.modes.segmentation` | Semantic segmentation workflows for U-Net style models | [Segmentation](./segmentation/README.md) |
| `mlx.modes.text_embedding` | GGUF embedding, exact/Chroma indexing, and BEIR retrieval benchmarking | [Text embedding](./text-embedding.md) |
| `mlx.modes.autoencoder` | Generic 1D vector autoencoder training and bottleneck export | [Autoencoder](./autoencoder.md) |
| `mlx.modes.saliency_mapping` | Still-image single-channel salient-object detection | [Saliency mapping](./saliency-mapping/README.md) |
| `mlx.modes.nlp` | Historical text-column CSV embedding compatibility | [Legacy NLP](nlp.md) |

## CLI Mapping

| CLI Mode | Backing Package |
| --- | --- |
| `object_detection`, `object-detection` | `mlx.modes.object_detection` |
| `track`, `tracking` | `mlx.modes.object_detection.tracking` |
| `image_classification`, `image-classification` | `mlx.modes.image_classification` |
| `image_recognition_oc`, `image-recognition-oc` | `mlx.modes.image_recognition_oc` |
| `video_anomaly_detection`, `video-anomaly-detection` | `mlx.modes.video_anomaly_detection` |
| `segmentation` | `mlx.modes.segmentation` |
| `text_embedding`, `text-embedding`, `nlp` | `mlx.modes.text_embedding` (`nlp` also retains legacy CSV options) |
| `autoencoder` | `mlx.modes.autoencoder` |
| `saliency_mapping`, `saliency-mapping` | `mlx.modes.saliency_mapping` |

## Extending MLX

Start with the [extension inventory](extensions.md) and [small executable tutorials](tutorials/README.md).
The canonical dependency and ownership rules remain in [ARCHITECTURE.md](../ARCHITECTURE.md).
