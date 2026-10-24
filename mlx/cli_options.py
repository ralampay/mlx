"""Lightweight CLI option declarations; shared flags retain historical defaults."""
import argparse
from mlx.core.cli_options import CliOption

OPTIONS = (
    CliOption(('-h', '--help'), {
        'action': 'store_true',
        'dest': 'help',
        'help': 'Show this help screen.',
    }),
    CliOption(('--mode',), {
        'default': None,
        'help': (
            'Mode to run: autoencoder, object_detection, track, image_classification, '
            'image_recognition_oc, segmentation, saliency_mapping, video_anomaly_detection, '
            'text_embedding.'
        ),
    }),
    CliOption(('--platform',), {
        'choices': ('local', 'aws'),
        'default': 'local',
        'help': (
            'Execution platform. AWS supports object detection, image classification, one-class image '
            'recognition, and sequential video-anomaly training.'
        ),
    }),
    CliOption(('--config',), {
        'default': None,
        'dest': 'config_path',
        'help': 'AWS YAML job configuration. Local execution does not read this file.',
    }),
    CliOption(('--profile',), {
        'default': None,
        'help': 'AWS shared-credentials profile. An explicit value overrides aws.profile in YAML.',
    }),
    CliOption(('--job-name',), {
        'default': None,
        'dest': 'job_name',
        'help': 'SageMaker job name used by status, stop, and resume.',
    }),
    CliOption(('--instance-type',), {
        'default': None,
        'dest': 'instance_type',
        'help': 'Explicit AWS instance-type override for train or resume.',
    }),
    CliOption(('--watch',), {
        'action': 'store_true',
        'default': False,
        'help': 'Poll AWS job status until it reaches a terminal state.',
    }),
    CliOption(('--poll-interval',), {
        'type': float,
        'default': 30.0,
        'dest': 'poll_interval',
    }),
    CliOption(('--rebuild-image',), {
        'action': 'store_true',
        'default': False,
        'dest': 'rebuild_image',
        'help': 'Force a new content-tagged SageMaker image build and ECR push.',
    }),
    CliOption(('--format',), {
        'choices': ('table', 'json'),
        'default': 'table',
        'dest': 'output_format',
        'help': 'Render interactive/Rich output or emit one structured JSON result.',
    }),
    CliOption(('--model',), {
        'default': None,
        'help': (
            'Provider-specific model identifier or architecture name; autoencoder accepts aliases or '
            'package.module:DefinitionClass.'
        ),
    }),
    CliOption(('--input',), {
        'default': None,
        'dest': 'input_path',
        'help': 'BEIR/artifact input for text embedding, or an embedding CSV for autoencoder workflows.',
    }),
    CliOption(('--names-only',), {
        'action': 'store_true',
        'dest': 'names_only',
        'help': 'List model names without constructing models in supported modes.',
    }),
    CliOption(('--adapter',), {
        'default': None,
        'help': (
            'YOLOX adapter checkpoint for LibreYOLO inference, or autoencoder checkpoint for text '
            'embedding.'
        ),
    }),
    CliOption(('--trust-checkpoint-code',), {
        'action': 'store_true',
        'help': 'Allow trusted external autoencoder architecture code; never enables unsafe pickle.',
    }),
    CliOption(('--loss',), {
        'default': None,
        'help': 'Autoencoder reconstruction loss alias or package.module:DefinitionClass.',
    }),
    CliOption(('--loss-config',), {
        'default': None,
        'dest': 'loss_config',
        'help': 'Optional JSON object with reconstruction-loss options.',
    }),
    CliOption(('--backbone',), {
        'default': None,
        'help': 'Feature backbone used by one-class image recognition.',
    }),
    CliOption(('--height',), {
        'type': int,
        'default': 256,
        'help': 'Input height; object detection defaults to 640.',
    }),
    CliOption(('--width',), {
        'type': int,
        'default': 256,
        'help': 'Input width; object detection defaults to 640.',
    }),
    CliOption(('--transform',), {
        'choices': ('resize', 'random-crop', 'center-crop'),
        'default': 'resize',
        'help': (
            'Segmentation spatial transform: resize, random-crop, or center-crop. Random crops become '
            'center crops for validation and test.'
        ),
    }),
    CliOption(('--device',), {
        'default': 'cpu',
        'help': 'Execution device such as cpu or cuda:0.',
    }),
    CliOption(('--action',), {
        'default': None,
        'help': 'Sub-action such as train, ls-models, infer-video, convert, benchmark, or build-dataset.',
    }),
    CliOption(('--drax-fusion-mode',), {
        'choices': ('average', 'sknet'),
        'default': 'average',
        'dest': 'drax_fusion_mode',
        'help': 'Drax residual fusion: fixed average or adaptive SKNet channel weighting.',
    }),
    CliOption(('--batch-size',), {
        'type': int,
        'default': 1,
        'dest': 'batch_size',
        'help': 'Training or evaluation batch size; autoencoder mode defaults to 64.',
    }),
    CliOption(('--dataset', '--dataset-path'), {
        'default': './tmp/dataset',
        'dest': 'dataset_path',
        'help': 'Local dataset source. Its layout and supported aliases are mode-specific.',
    }),
    CliOption(('--dataset-s3-uri',), {
        'default': None,
        'dest': 'dataset_s3_uri',
        'help': (
            'Training-only S3 URI of a ZIP dataset. Local training caches and extracts it before '
            'training; AWS training uses the managed input channel.'
        ),
    }),
    CliOption(('--dataset-cache-dir',), {
        'default': '~/.cache/mlx/datasets',
        'dest': 'dataset_cache_dir',
        'help': 'Persistent content cache for locally staged S3 dataset ZIPs.',
    }),
    CliOption(('--epochs',), {
        'type': int,
        'default': 100,
        'help': 'Training epoch count; autoencoder mode defaults to 50.',
    }),
    CliOption(('--output',), {
        'default': None,
        'dest': 'output_path',
        'help': (
            'Output directory written by training, benchmarks, or tracking. Detection uses it as the '
            'provider project directory, or as the ONNX export destination for convert.'
        ),
    }),
    CliOption(('--train-count',), {
        'type': int,
        'default': None,
        'dest': 'train_count',
        'help': 'Images per label assigned to the train split when building classification datasets.',
    }),
    CliOption(('--val-count',), {
        'type': int,
        'default': None,
        'dest': 'val_count',
        'help': 'Images per label assigned to the val split when building classification datasets.',
    }),
    CliOption(('--test-count',), {
        'type': int,
        'default': None,
        'dest': 'test_count',
        'help': 'Images per label assigned to the test split when building classification datasets.',
    }),
    CliOption(('--train-ratio',), {
        'type': float,
        'default': None,
        'dest': 'train_ratio',
        'help': 'Train split ratio applied within each label when building classification datasets.',
    }),
    CliOption(('--val-ratio',), {
        'type': float,
        'default': None,
        'dest': 'val_ratio',
        'help': 'Validation split ratio for autoencoder training or classification dataset construction.',
    }),
    CliOption(('--test-ratio',), {
        'type': float,
        'default': None,
        'dest': 'test_ratio',
        'help': 'Test split ratio applied within each label when building classification datasets.',
    }),
    CliOption(('--split-mode',), {
        'choices': ('counts', 'ratios'),
        'default': None,
        'dest': 'split_mode',
        'help': (
            'Build-dataset split mode: counts or ratios. Ratio mode splits each label independently '
            'using the provided ratios.'
        ),
    }),
    CliOption(('--overwrite',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'help': 'Allow supported workflows to replace existing output artifacts without prompting.',
    }),
    CliOption(('--model-path', '--checkpoint'), {
        'default': None,
        'dest': 'model_path',
        'help': 'Model checkpoint; LibreYOLO inference also accepts self-contained MLX adapter detectors.',
    }),
    CliOption(('--file-path',), {
        'default': None,
        'dest': 'file_path',
        'help': 'Video path for file-based inference.',
    }),
    CliOption(('--input-img',), {
        'default': '/tmp/image.jpg',
        'dest': 'input_img',
        'help': 'Input image for classification or one-class recognition inference.',
    }),
    CliOption(('--workers',), {
        'type': int,
        'default': 4,
        'help': 'Evaluation dataloader worker count.',
    }),
    CliOption(('--camera-index',), {
        'type': int,
        'default': 0,
        'dest': 'camera_index',
        'help': 'Camera index for webcam inference.',
    }),
    CliOption(('--pretrained',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'help': 'Toggle supported pretrained model initialization.',
    }),
    CliOption(('--colored',), {
        'action': argparse.BooleanOptionalAction,
        'default': True,
        'help': 'Use RGB input; disable for supported grayscale workflows.',
    }),
    CliOption(('--plots',), {
        'action': argparse.BooleanOptionalAction,
        'default': True,
        'help': 'Write supported provider-native training or benchmark plots.',
    }),
    CliOption(('--verbose',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'help': 'Show per-epoch live progress bars when supported.',
    }),
    CliOption(('--use-best',), {
        'action': argparse.BooleanOptionalAction,
        'default': True,
        'dest': 'use_best',
        'help': (
            'Use the best validation checkpoint when training supports it. For image classification '
            'this saves only the best validation-loss checkpoint; for object detection this selects '
            'weights/best.pt after training.'
        ),
    }),
    CliOption(('--apply-transformations',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'dest': 'apply_transformations',
        'help': 'Apply supported image-training augmentations: RandomHorizontalFlip and RandomRotation(10).',
    }),
    CliOption(('--lr',), {
        'type': float,
        'default': None,
        'help': 'Learning rate for supported training workflows; autoencoder mode defaults to 0.001.',
    }),
    CliOption(('--amp',), {
        'action': argparse.BooleanOptionalAction,
        'default': True,
        'help': 'Toggle mixed precision for supported training providers.',
    }),
    CliOption(('--loss-clip',), {
        'type': float,
        'default': None,
        'dest': 'loss_clip',
        'help': 'Optional gradient clipping value.',
    }),
    CliOption(('--svdd-weight',), {
        'type': float,
        'default': 0.05,
        'dest': 'svdd_weight',
        'help': 'Weight applied to the joint Deep SVDD loss.',
    }),
    CliOption(('--svdd-dim',), {
        'type': int,
        'default': 128,
        'dest': 'svdd_dim',
        'help': 'Deep SVDD embedding width.',
    }),
    CliOption(('--svdd-hidden-dim',), {
        'type': int,
        'default': 256,
        'dest': 'svdd_hidden_dim',
        'help': 'Deep SVDD projection hidden width.',
    }),
    CliOption(('--svdd-quantile',), {
        'type': float,
        'default': 0.95,
        'dest': 'svdd_quantile',
        'help': 'Normal-validation score quantile used for rejection calibration.',
    }),
    CliOption(('--svdd-warmup-epochs',), {
        'type': int,
        'default': 0,
        'dest': 'svdd_warmup_epochs',
        'help': 'Classification-only epochs before applying the SVDD loss.',
    }),
    CliOption(('--seed', '--random-seed'), {
        'type': int,
        'default': None,
        'dest': 'random_seed',
        'help': 'Global random seed applied across Python, NumPy, and PyTorch.',
    }),
    CliOption(('--run-name',), {
        'default': None,
        'dest': 'run_name',
        'help': 'Optional provider run folder name.',
    }),
    CliOption(('--num-classes',), {
        'type': int,
        'default': 2,
        'dest': 'num_classes',
        'help': 'Number of image-classification or segmentation output classes.',
    }),
    CliOption(('--class-names',), {
        'default': None,
        'dest': 'class_names',
        'help': 'Comma-separated segmentation class names stored in checkpoints and research artifacts.',
    }),
    CliOption(('--split',), {
        'choices': ('train', 'val', 'test'),
        'default': 'test',
        'help': 'Object-detection or segmentation dataset split used by benchmark: train, val, or test.',
    }),
    CliOption(('--boundary-tolerance',), {
        'type': int,
        'default': 2,
        'dest': 'boundary_tolerance',
        'help': 'Boundary-metric matching tolerance in resized-image pixels.',
    }),
    CliOption(('--calibration-bins',), {
        'type': int,
        'default': 15,
        'dest': 'calibration_bins',
        'help': 'Confidence bins used for segmentation calibration metrics.',
    }),
    CliOption(('--curve-bins',), {
        'type': int,
        'default': 4096,
        'dest': 'curve_bins',
        'help': 'Score bins used for bounded-memory segmentation ROC/PR metrics.',
    }),
    CliOption(('--threshold-steps',), {
        'type': int,
        'default': 101,
        'dest': 'threshold_steps',
        'help': 'Number of binary segmentation thresholds evaluated by benchmark.',
    }),
    CliOption(('--mask-threshold',), {
        'type': float,
        'default': 0.5,
        'dest': 'mask_threshold',
        'help': 'Threshold used when rendering binary segmentation masks.',
    }),
    CliOption(('--overlay-alpha',), {
        'type': float,
        'default': 0.45,
        'dest': 'overlay_alpha',
        'help': 'Blend strength for segmentation overlays.',
    }),
    CliOption(('--display',), {
        'action': argparse.BooleanOptionalAction,
        'default': True,
        'help': 'Show live tracking overlays or rendered CAM images in OpenCV windows.',
    }),
)
