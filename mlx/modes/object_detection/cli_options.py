"""Lightweight CLI option declarations; shared flags retain historical defaults."""
import argparse
from mlx.core.cli_options import CliOption

OPTIONS = (
    CliOption(('--provider',), {
        'default': 'ultralytics',
        'help': 'Object-detection provider: ultralytics or libreyolo.',
    }),
    CliOption(('--methods',), {
        'default': None,
        'help': 'Comma-separated YOLOX adapter experiment methods.',
    }),
    CliOption(('--experiment-seeds',), {
        'default': None,
        'help': 'Comma-separated seeds for adapter experiments.',
    }),
    CliOption(('--baseline-study',), {
        'default': None,
        'help': 'Read-only completed adapter study to reuse for comparison.',
    }),
    CliOption(('--study-config',), {
        'default': None,
        'help': 'JSON specification for an inference-only transfer study.',
    }),
    CliOption(('--study-phase',), {
        'choices': ('prepare', 'pilot', 'all'),
        'default': 'all',
    }),
    CliOption(('--comparison-method',), {
        'default': 'drax',
        'help': 'Candidate method for paired adapter reports.',
    }),
    CliOption(('--bootstrap-samples',), {
        'type': int,
        'default': 2000,
        'help': 'Deterministic image-bootstrap samples for adapter slice reports.',
    }),
    CliOption(('--adapter-reduction',), {
        'type': int,
        'default': 8,
        'help': 'Feature-adapter channel reduction factor.',
    }),
    CliOption(('--adapter-rank',), {
        'type': int,
        'default': 8,
        'help': 'LoRA rank for YOLOX adapter experiments.',
    }),
    CliOption(('--adapter-alpha',), {
        'type': float,
        'default': 1.0,
        'help': 'Adapter residual scale or LoRA alpha.',
    }),
    CliOption(('--adapter-target',), {
        'choices': ('backbone', 'neck', 'backbone+neck'),
        'default': 'neck',
        'help': 'Adapter placement: backbone, neck, or backbone+neck.',
    }),
    CliOption(('--train-head',), {
        'action': 'store_true',
        'help': 'Train the detection head alongside an injected adapter.',
    }),
    CliOption(('--head-policy',), {
        'choices': ('preserve', 'reset-classifiers'),
        'default': 'preserve',
    }),
    CliOption(('--condition-id',), {
        'help': 'Distinct identity for a single taxonomy-transfer condition',
    }),
    CliOption(('--gradient-accumulation',), {
        'type': int,
        'default': 1,
        'help': 'Optimizer accumulation steps; effective batch is physical batch times this value.',
    }),
    CliOption(('--eval-interval',), {
        'type': int,
        'default': None,
    }),
    CliOption(('--no-aug-epochs',), {
        'type': int,
        'default': None,
    }),
    CliOption(('--patience',), {
        'type': int,
        'default': None,
    }),
    CliOption(('--model-s3-uri',), {
        'default': None,
        'dest': 'model_s3_uri',
        'help': 'Exact S3 .pt checkpoint used to initialize AWS object-detection fine-tuning.',
    }),
    CliOption(('--experiment-id',), {
        'default': None,
        'dest': 'experiment_id',
        'help': 'Comparison run ID returned by the AWS submission action.',
    }),
    CliOption(('--models',), {
        'default': None,
        'help': 'Comma-separated MLX model names for comparison.',
    }),
    CliOption(('--confidence',), {
        'type': float,
        'default': 0.25,
        'help': 'Detection confidence threshold.',
    }),
    CliOption(('--iou',), {
        'type': float,
        'default': 0.6,
        'help': 'NMS IoU threshold for object-detection benchmarks.',
    }),
    CliOption(('--max-detections',), {
        'type': int,
        'default': 300,
        'dest': 'max_detections',
        'help': 'Maximum detections per image during object-detection benchmarking.',
    }),
    CliOption(('--save-predictions',), {
        'action': argparse.BooleanOptionalAction,
        'default': True,
        'dest': 'save_predictions',
        'help': 'Write provider-native prediction JSON during object-detection benchmarking.',
    }),
    CliOption(('--validate-after-training',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'dest': 'validate_after_training',
        'help': 'Run the provider-neutral benchmark after object-detection training.',
    }),
    CliOption(('--validation-split',), {
        'choices': ('train', 'val', 'test'),
        'default': 'val',
        'dest': 'validation_split',
        'help': 'Dataset split benchmarked after object-detection training.',
    }),
    CliOption(('--validation-confidence',), {
        'type': float,
        'default': 0.001,
        'dest': 'validation_confidence',
    }),
    CliOption(('--validation-iou',), {
        'type': float,
        'default': 0.6,
        'dest': 'validation_iou',
    }),
    CliOption(('--validation-max-detections',), {
        'type': int,
        'default': 300,
        'dest': 'validation_max_detections',
    }),
    CliOption(('--distiller',), {
        'default': None,
        'help': 'Local teacher .pt checkpoint for LibreYOLO feature distillation.',
    }),
    CliOption(('--distill-loss',), {
        'choices': ('cwd', 'mgd'),
        'default': None,
        'help': 'Feature loss and its positive weight; requires --distiller.',
    }),
    CliOption(('--distill-weight',), {
        'type': float,
        'default': None,
    }),
    CliOption(('--distill-temperature',), {
        'type': float,
        'default': None,
        'help': 'CWD temperature or MGD masking fraction; requires --distiller.',
    }),
    CliOption(('--distill-mask-ratio',), {
        'type': float,
        'default': None,
    }),
    CliOption(('--incremental-adapter',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'dest': 'incremental_adapter',
        'help': 'Enable IncrementalAdapters for LibreYOLO YOLOX-Drax-MobileNetV3 training.',
    }),
    CliOption(('--incremental-adapter-train-only',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'dest': 'incremental_adapter_train_only',
        'help': 'Freeze the loaded foundation detector and train only its IncrementalAdapters.',
    }),
    CliOption(('--incremental-adapter-type',), {
        'default': None,
        'dest': 'incremental_adapter_type',
        'help': 'Select a registered YOLOX-Drax-MobileNetV3 IncrementalAdapter architecture.',
    }),
    CliOption(('--lr0',), {
        'type': float,
        'default': None,
        'help': 'Override initial learning rate.',
    }),
    CliOption(('--optimizer',), {
        'default': 'auto',
        'help': 'Provider optimizer selection; auto uses the provider default.',
    }),
    CliOption(('--nbs',), {
        'type': int,
        'default': 64,
        'help': 'Nominal batch size for LR scaling.',
    }),
    CliOption(('--warmup-epochs',), {
        'type': float,
        'default': 3.0,
        'dest': 'warmup_epochs',
        'help': 'Warmup epoch count.',
    }),
)
