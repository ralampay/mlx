import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rich.panel import Panel
from rich.table import Table

from mlx.cli_config import build_runtime_config, explicit_option_destinations
from mlx.cli_routing import (
    MODE_DESCRIPTORS,
    MODE_REGISTRY,
    ModeRunner,
    UnknownModeError,
    resolve_mode_runner,
    resolve_mode_descriptor,
)
from mlx.core.artifacts import json_safe
from mlx.core.exceptions import MLXAbort, MLXUserError
from mlx.core.random import apply_global_seed, seed_everything
from mlx.core.ui import console, print_error, print_startup, print_warning

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)


class CLIUsageError(Exception):
    """Raised when command-line arguments are invalid."""


class RichArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


def build_parser() -> RichArgumentParser:
    parser = RichArgumentParser(add_help=False, prog="python -m mlx")
    parser.add_argument("-h", "--help", action="store_true", dest="help")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--platform", choices=("local", "aws"), default="local")
    parser.add_argument("--config", default=None, dest="config_path")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--job-name", default=None, dest="job_name")
    parser.add_argument("--instance-type", default=None, dest="instance_type")
    parser.add_argument("--watch", action="store_true", default=False)
    parser.add_argument("--poll-interval", type=float, default=30.0, dest="poll_interval")
    parser.add_argument("--rebuild-image", action="store_true", default=False, dest="rebuild_image")
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        dest="output_format",
    )
    parser.add_argument("--provider", default="ultralytics")
    parser.add_argument("--model", default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action", default=None)
    parser.add_argument("--embedding-size", type=int, default=4096, dest="embedding_size")
    parser.add_argument(
        "--drax-fusion-mode",
        choices=("average", "sknet"),
        default="average",
        dest="drax_fusion_mode",
    )
    parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    parser.add_argument("--dataset", "--dataset-path", default="./tmp/dataset", dest="dataset_path")
    parser.add_argument("--dataset-s3-uri", default=None, dest="dataset_s3_uri")
    parser.add_argument(
        "--dataset-cache-dir",
        default="~/.cache/mlx/datasets",
        dest="dataset_cache_dir",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-pairs", type=int, default=100, dest="num_pairs")
    parser.add_argument("--output", default=None, dest="output_path")
    parser.add_argument("--train-count", type=int, default=None, dest="train_count")
    parser.add_argument("--val-count", type=int, default=None, dest="val_count")
    parser.add_argument("--test-count", type=int, default=None, dest="test_count")
    parser.add_argument("--train-ratio", type=float, default=None, dest="train_ratio")
    parser.add_argument("--val-ratio", type=float, default=None, dest="val_ratio")
    parser.add_argument("--test-ratio", type=float, default=None, dest="test_ratio")
    parser.add_argument("--split-mode", choices=("counts", "ratios"), default=None, dest="split_mode")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-path", default=None, dest="model_path")
    parser.add_argument("--model-file", default=None, dest="model_file")
    parser.add_argument("--input-file", default=None, dest="input_file")
    parser.add_argument("--output-file", default=None, dest="output_file")
    parser.add_argument("--column-name", "--column_name", default="content", dest="column_name")
    parser.add_argument("--file-path", default=None, dest="file_path")
    parser.add_argument("--input-img", default="/tmp/image.jpg", dest="input_img")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=300, dest="max_detections")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="save_predictions",
    )
    parser.add_argument(
        "--validate-after-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="validate_after_training",
    )
    parser.add_argument(
        "--validation-split",
        choices=("train", "val", "test"),
        default="val",
        dest="validation_split",
    )
    parser.add_argument(
        "--validation-confidence",
        type=float,
        default=0.001,
        dest="validation_confidence",
    )
    parser.add_argument(
        "--validation-iou",
        type=float,
        default=0.6,
        dest="validation_iou",
    )
    parser.add_argument(
        "--validation-max-detections",
        type=int,
        default=300,
        dest="validation_max_detections",
    )
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--tracker-config", default=None, dest="tracker_config")
    parser.add_argument("--tracking-jsonl", default=None, dest="tracking_jsonl")
    parser.add_argument(
        "--ground-truth",
        "--gt-file",
        default=None,
        dest="ground_truth",
    )
    parser.add_argument(
        "--track-class-id",
        action="append",
        type=int,
        default=None,
        dest="track_class_ids",
    )
    parser.add_argument(
        "--benchmark-iou",
        type=float,
        default=0.5,
        dest="benchmark_iou",
    )
    parser.add_argument("--camera-index", type=int, default=0, dest="camera_index")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-best", action=argparse.BooleanOptionalAction, default=True, dest="use_best")
    parser.add_argument(
        "--apply-transformations",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="apply_transformations",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--warmup-epochs", type=float, default=3.0, dest="warmup_epochs")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-clip", type=float, default=None, dest="loss_clip")
    parser.add_argument(
        "--ood-method",
        choices=("none", "deep-svdd"),
        default="none",
        dest="ood_method",
    )
    parser.add_argument("--svdd-weight", type=float, default=0.05, dest="svdd_weight")
    parser.add_argument("--svdd-dim", type=int, default=128, dest="svdd_dim")
    parser.add_argument(
        "--svdd-hidden-dim", type=int, default=256, dest="svdd_hidden_dim"
    )
    parser.add_argument(
        "--svdd-quantile", type=float, default=0.95, dest="svdd_quantile"
    )
    parser.add_argument(
        "--svdd-warmup-epochs", type=int, default=0, dest="svdd_warmup_epochs"
    )
    parser.add_argument("--clip-length", type=int, default=16, dest="clip_length")
    parser.add_argument("--frame-stride", type=int, default=1, dest="frame_stride")
    parser.add_argument(
        "--backbone-mode",
        choices=("3d", "frame-2d"),
        default="3d",
        dest="backbone_mode",
    )
    parser.add_argument(
        "--backbone-temporal-kernel-size",
        type=int,
        default=3,
        dest="backbone_temporal_kernel_size",
    )
    parser.add_argument("--temporal-model", default="tcn", dest="temporal_model")
    parser.add_argument("--temporal-hidden-dim", type=int, default=256, dest="temporal_hidden_dim")
    parser.add_argument("--temporal-embedding-dim", type=int, default=128, dest="temporal_embedding_dim")
    parser.add_argument("--temporal-kernel-size", type=int, default=3, dest="temporal_kernel_size")
    parser.add_argument("--temporal-dropout", type=float, default=0.0, dest="temporal_dropout")
    parser.add_argument("--frame-aggregation", choices=("mean", "max"), default="mean", dest="frame_aggregation")
    parser.add_argument("--seed", "--random-seed", type=int, default=None, dest="random_seed")
    parser.add_argument("--run-name", default=None, dest="run_name")
    parser.add_argument("--num-classes", type=int, default=2, dest="num_classes")
    parser.add_argument("--class-names", default=None, dest="class_names")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        dest="boundary_tolerance",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=15,
        dest="calibration_bins",
    )
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=101,
        dest="threshold_steps",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, dest="mask_threshold")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, dest="overlay_alpha")
    parser.add_argument("--cam-method", choices=("gradcam", "ablationcam", "scorecam"), default="gradcam", dest="cam_method")
    parser.add_argument("--target-layer", default=None, dest="target_layer")
    parser.add_argument("--target-index", type=int, default=None, dest="target_index")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True, dest="save_images")
    parser.add_argument("--window-delay", type=int, default=0, dest="window_delay")
    parser.add_argument("--aug-smooth", action=argparse.BooleanOptionalAction, default=False, dest="aug_smooth")
    parser.add_argument("--eigen-smooth", action=argparse.BooleanOptionalAction, default=False, dest="eigen_smooth")
    return parser


def _render_help() -> None:
    console.print(
        Panel.fit(
            "MLX\nA rich-powered CLI for machine-learning workflows.",
            border_style="cyan",
        )
    )

    usage = Table(title="Usage", show_header=False)
    usage.add_column("Command", style="bold cyan")
    usage.add_row("python -m mlx --mode object_detection --action ls-models")
    usage.add_row("python -m mlx --mode object_detection --provider libreyolo --action ls-models")
    usage.add_row("python -m mlx --mode object_detection --provider libreyolo --action train --dataset coco8 --model yolo9-t")
    usage.add_row("python -m mlx --mode object_detection --action train --dataset coco8 --model yolo26")
    usage.add_row("python -m mlx --mode object_detection --action benchmark --dataset ./dataset --model-path ./best.pt --split test --output ./benchmark")
    usage.add_row("python -m mlx --mode object_detection --platform aws --action train --config ./aws-training.yaml")
    usage.add_row("python -m mlx --mode object_detection --platform aws --action status --config ./aws-training.yaml --job-name JOB_NAME --watch")
    usage.add_row("python -m mlx --mode object_detection --platform aws --action resume --config ./aws-training.yaml --job-name JOB_NAME")
    usage.add_row("python -m mlx --mode object_detection --action train --dataset coco8 --model draxnet-yolo26 --output ./runs/draxnet")
    usage.add_row("python -m mlx --mode object_detection --action infer-camera --model draxnet-yolo26 --model-path ./runs/draxnet/exp/weights/best.pt")
    usage.add_row("python -m mlx --mode object_detection --action convert --model-path ./runs/draxnet/exp/weights/best.pt --output ./exports")
    usage.add_row("python -m mlx --mode track --tracker bytetrack --model yolo26 --model-path ./best.pt --file-path ./video.mp4 --output ./tracking-run")
    usage.add_row("python -m mlx --mode track --provider libreyolo --tracker sort --model-path ./best.onnx --file-path ./video.mp4 --ground-truth ./gt.txt --output ./tracking-run")
    usage.add_row("python -m mlx --mode track --action export-mot --tracking-jsonl ./tracking-run/tracks.jsonl --track-class-id 0 --output ./person-mot")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/resnet18 --dataset ./dataset --model resnet18")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/resnet18-s3 --dataset-s3-uri s3://my-datasets/classification.zip --model resnet18 --profile mlx-training")
    usage.add_row("python -m mlx --mode image_classification --platform aws --action train --config ./aws-classification.yaml")
    usage.add_row("python -m mlx --mode image_classification --platform aws --action status --config ./aws-classification.yaml --job-name JOB_NAME --watch")
    usage.add_row("python -m mlx --mode image_classification --action ls-models")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/siamese --dataset ./omniglot --model siamese-le-net")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/siamese-resnet --dataset ./omniglot --model siamese-resnet18 --pretrained")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --output ./dataset --train-count 100 --val-count 20 --test-count 20 --overwrite --seed 42")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --split-mode ratios --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --output ./dataset --overwrite --seed 42")
    usage.add_row("python -m mlx --mode segmentation --action train --dataset ./dataset --model unet --output unet-seg.pt")
    usage.add_row("python -m mlx --mode segmentation --action train --dataset ./dataset --model unet-resnet18 --pretrained --output ./unet-resnet18")
    usage.add_row("python -m mlx --mode segmentation --action train --dataset ./dataset --model unet-draxnet-sknet --output ./unet-draxnet-sknet")
    usage.add_row("python -m mlx --mode segmentation --action benchmark --dataset ./dataset --model-path ./unet-seg.pt --output ./benchmark-results")
    usage.add_row("python -m mlx --mode segmentation --action ls-models")
    usage.add_row("python -m mlx --mode segmentation --action infer-image --model-path ./unet-seg.pt --input-img ./sample.jpg")
    usage.add_row("python -m mlx --mode segmentation --action build-dataset --dataset ./raw-segmentation")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action train --model resnet18 --backbone-mode 3d --backbone-temporal-kernel-size 3 --clip-length 16 --dataset ./ped2-prepared --output ./artifacts/ped2")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action ls-models")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action benchmark --model-path ./model.pth --dataset ./ped2-prepared/test --output ./benchmark")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action infer-video --model-path ./model.pth --file-path ./sample.mp4 --output ./inference")
    usage.add_row("python -m mlx --mode nlp --action embed --model-file ./model.gguf --input-file ./input.csv")
    console.print(usage)

    options = Table(title="Options", show_lines=True)
    options.add_column("Flag", style="cyan", no_wrap=True)
    options.add_column("Default", style="magenta")
    options.add_column("Description", style="white")
    options.add_row(
        "--mode",
        "None",
        "Mode to run: " + ", ".join(item.name for item in MODE_DESCRIPTORS) + ".",
    )
    options.add_row("--platform", "local", "Execution platform. AWS supports object-detection and image-classification training lifecycle actions.")
    options.add_row("--config", "None", "AWS YAML job configuration. Local execution does not read this file.")
    options.add_row("--profile", "YAML/default", "AWS shared-credentials profile. An explicit value overrides aws.profile in YAML.")
    options.add_row("--job-name", "None", "SageMaker job name used by status, stop, and resume.")
    options.add_row("--instance-type", "YAML", "Explicit AWS instance-type override for train or resume.")
    options.add_row("--watch", "False", "Poll AWS job status until it reaches a terminal state.")
    options.add_row(
        "--format",
        "table",
        "Render interactive/Rich output or emit one structured JSON result.",
    )
    options.add_row("--rebuild-image", "False", "Force a new content-tagged SageMaker image build and ECR push.")
    options.add_row("--provider", "ultralytics", "Object-detection provider: ultralytics or libreyolo.")
    options.add_row("--model", "None", "Provider-specific model identifier, YAML path, or architecture name.")
    options.add_row("--action", "mode-specific", "Sub-action such as train, ls-models, infer-video, convert, benchmark, or build-dataset.")
    options.add_row("--dataset", "./tmp/dataset", "Local dataset source. Its layout and supported aliases are mode-specific.")
    options.add_row("--dataset-s3-uri", "None", "Training-only S3 URI of a ZIP dataset. Local training caches and extracts it before training; AWS training uses the managed input channel.")
    options.add_row("--dataset-cache-dir", "~/.cache/mlx/datasets", "Persistent content cache for locally staged S3 dataset ZIPs.")
    options.add_row("--output", "None", "Output directory written by training, benchmarks, or tracking. Detection uses it as the provider project directory, or as the ONNX export destination for convert.")
    options.add_row("--train-count", "None", "Images per label assigned to the train split when building classification datasets.")
    options.add_row("--val-count", "None", "Images per label assigned to the val split when building classification datasets.")
    options.add_row("--test-count", "None", "Images per label assigned to the test split when building classification datasets.")
    options.add_row("--train-ratio", "None", "Train split ratio applied within each label when building classification datasets.")
    options.add_row("--val-ratio", "None", "Validation split ratio applied within each label when building classification datasets.")
    options.add_row("--test-ratio", "None", "Test split ratio applied within each label when building classification datasets.")
    options.add_row("--split-mode", "None", "Build-dataset split mode: counts or ratios. Ratio mode splits each label independently using the provided ratios.")
    options.add_row("--overwrite / --no-overwrite", "False", "Allow supported workflows to replace existing output artifacts without prompting.")
    options.add_row("--model-path", "None", "Provider-compatible checkpoint path for inference, resume, warm starts, or ONNX conversion.")
    options.add_row("--model-file", "None", "GGUF embedding model used by NLP embed.")
    options.add_row("--input-file", "None", "Input CSV used by NLP embed.")
    options.add_row("--output-file", "derived", "Output CSV used by NLP embed; defaults beside the input file.")
    options.add_row("--column-name", "content", "CSV text column used by NLP embed.")
    options.add_row("--file-path", "None", "Video path for file-based inference.")
    options.add_row("--input-img", "/tmp/image.jpg", "Input image for classification inference.")
    options.add_row("--device", "cpu", "Execution device such as cpu or cuda:0.")
    options.add_row("--height / --width", "256 / 256", "Image size controls.")
    options.add_row("--batch-size", "1", "Training or evaluation batch size.")
    options.add_row("--epochs", "100", "Training epoch count.")
    options.add_row("--num-pairs", "100", "One-shot image-classification pairs per label for training or benchmarking.")
    options.add_row("--embedding-size", "4096", "Embedding width for any one-shot Siamese model.")
    options.add_row(
        "--drax-fusion-mode",
        "average",
        "Drax residual fusion: fixed average or adaptive SKNet channel weighting.",
    )
    options.add_row("--confidence", "0.25", "Detection confidence threshold.")
    options.add_row("--iou", "0.6", "NMS IoU threshold for object-detection benchmarks.")
    options.add_row("--max-detections", "300", "Maximum detections per image during object-detection benchmarking.")
    options.add_row("--workers", "4", "Evaluation dataloader worker count.")
    options.add_row("--save-predictions / --no-save-predictions", "True", "Write provider-native prediction JSON during object-detection benchmarking.")
    options.add_row("--validate-after-training", "False", "Run the provider-neutral benchmark after object-detection training.")
    options.add_row("--validation-split", "val", "Dataset split benchmarked after object-detection training.")
    options.add_row("--tracker", "bytetrack", "Tracking algorithm alias or external package.module:ClassName.")
    options.add_row("--tracker-config", "None", "Optional JSON object file passed as keyword arguments to the tracker constructor.")
    options.add_row("--tracking-jsonl", "None", "Class-aware tracks.jsonl input used by track --action export-mot.")
    options.add_row("--ground-truth / --gt-file", "None", "Optional 10-column MOTChallenge ground-truth file used for tracking benchmarks.")
    options.add_row("--track-class-id", "all", "Repeatable detector class ID included in tracking; all classes are used by default.")
    options.add_row("--benchmark-iou", "0.5", "Minimum box IoU used for MOT benchmark matching.")
    options.add_row("--camera-index", "0", "Camera index for webcam inference.")
    options.add_row("--pretrained / --no-pretrained", "False", "Toggle supported pretrained model initialization.")
    options.add_row("--plots / --no-plots", "True", "Write supported provider-native training or benchmark plots.")
    options.add_row("--verbose / --no-verbose", "False", "Show per-epoch live progress bars when supported.")
    options.add_row(
        "--use-best / --no-use-best",
        "True",
        "Use the best validation checkpoint when training supports it. For image classification this saves only the best validation-loss checkpoint; for object detection this selects weights/best.pt after training.",
    )
    options.add_row(
        "--apply-transformations / --no-apply-transformations",
        "False",
        "Apply image-classification training augmentations: RandomHorizontalFlip and RandomRotation(10).",
    )
    options.add_row("--lr", "None", "Learning rate for image-classification training.")
    options.add_row("--amp / --no-amp", "True", "Toggle mixed precision for supported training providers.")
    options.add_row("--lr0", "None", "Override initial learning rate.")
    options.add_row("--optimizer", "auto", "Provider optimizer selection; auto uses the provider default.")
    options.add_row("--nbs", "64", "Nominal batch size for LR scaling.")
    options.add_row("--warmup-epochs", "3.0", "Warmup epoch count.")
    options.add_row("--loss-clip", "None", "Optional gradient clipping value.")
    options.add_row("--ood-method", "none", "Optional standard-classifier OOD method: none or deep-svdd.")
    options.add_row("--svdd-weight", "0.05", "Weight applied to the joint Deep SVDD loss.")
    options.add_row("--svdd-dim", "128", "Deep SVDD embedding width.")
    options.add_row("--svdd-hidden-dim", "256", "Deep SVDD projection hidden width.")
    options.add_row("--svdd-quantile", "0.95", "Validation-score quantile used for rejection calibration.")
    options.add_row("--svdd-warmup-epochs", "0", "Classification-only epochs before applying the SVDD loss.")
    options.add_row("--clip-length", "16", "Frames per video-anomaly clip window.")
    options.add_row("--frame-stride", "1", "Sampling stride between frames in a video-anomaly clip.")
    options.add_row("--backbone-mode", "3d", "Clip-native 3D backbone, or legacy frame-2d plus TCN.")
    options.add_row("--backbone-temporal-kernel-size", "3", "Odd temporal kernel used to inflate spatial backbone convolutions.")
    options.add_row("--temporal-model", "tcn", "Legacy frame-2d temporal encoder alias.")
    options.add_row("--temporal-hidden-dim", "256", "Legacy frame-2d temporal CNN hidden channels.")
    options.add_row("--temporal-embedding-dim", "128", "Legacy frame-2d temporal CNN output width.")
    options.add_row("--temporal-kernel-size", "3", "Legacy frame-2d Conv1d kernel size.")
    options.add_row("--temporal-dropout", "0.0", "Legacy frame-2d temporal CNN dropout.")
    options.add_row("--frame-aggregation", "mean", "Aggregate overlapping window scores per frame using mean or max.")
    options.add_row("--seed / --random-seed", "None", "Global random seed applied across Python, NumPy, and PyTorch.")
    options.add_row("--run-name", "None", "Optional provider run folder name.")
    options.add_row("--num-classes", "2", "Number of image-classification or segmentation output classes.")
    options.add_row("--class-names", "generated", "Comma-separated segmentation class names stored in checkpoints and research artifacts.")
    options.add_row("--split", "test", "Object-detection or segmentation dataset split used by benchmark: train, val, or test.")
    options.add_row("--boundary-tolerance", "2", "Boundary-metric matching tolerance in resized-image pixels.")
    options.add_row("--calibration-bins", "15", "Confidence bins used for segmentation calibration metrics.")
    options.add_row("--threshold-steps", "101", "Number of binary segmentation thresholds evaluated by benchmark.")
    options.add_row("--mask-threshold", "0.5", "Threshold used when rendering binary segmentation masks.")
    options.add_row("--overlay-alpha", "0.45", "Blend strength for segmentation overlays.")
    options.add_row("--cam-method", "gradcam", "CAM method for image-classification cam: gradcam, ablationcam, or scorecam.")
    options.add_row("--target-layer", "None", "Optional dotted module path to explain, such as layer4.1 or features.-1.")
    options.add_row("--target-index", "None", "Optional class index or Siamese output index to explain. Defaults to model prediction.")
    options.add_row("--max-samples", "None", "Maximum number of test samples or one-shot pairs to render.")
    options.add_row(
        "--display / --no-display",
        "True",
        "Show live tracking overlays or rendered CAM images in OpenCV windows.",
    )
    options.add_row("--save-images / --no-save-images", "True", "Write rendered CAM images under --output when provided.")
    options.add_row("--window-delay", "0", "OpenCV waitKey delay in milliseconds between displayed CAM images.")
    options.add_row("--aug-smooth / --no-aug-smooth", "False", "Apply Grad-CAM test-time augmentation smoothing.")
    options.add_row("--eigen-smooth / --no-eigen-smooth", "False", "Apply Grad-CAM Eigen smoothing.")
    options.add_row("--help", "False", "Show this help screen.")
    console.print(options)

    available = Table(title="Available Modes", show_header=True)
    available.add_column("Mode", style="cyan", no_wrap=True)
    available.add_column("Actions", style="white")
    for descriptor in MODE_DESCRIPTORS:
        available.add_row(descriptor.name, ", ".join(descriptor.actions))
    console.print(available)


def _build_config(namespace: argparse.Namespace) -> Dict[str, Any]:
    return build_runtime_config(namespace)


def _explicit_option_destinations(
    parser: argparse.ArgumentParser,
    args: Sequence[str],
) -> set[str]:
    return explicit_option_destinations(parser, args)


def _resolve_mode_runner(mode: str) -> ModeRunner:
    return resolve_mode_runner(mode)


def _render_unknown_mode() -> None:
    table = Table(title="Available Modes", show_header=True)
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Purpose", style="white")
    for descriptor in MODE_DESCRIPTORS:
        table.add_row(descriptor.name, descriptor.purpose)
    console.print(table)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    try:
        namespace = parser.parse_args(args)
    except CLIUsageError as exc:
        print_error(str(exc))
        _render_help()
        return 2

    if namespace.help:
        _render_help()
        return 0
    if not namespace.mode:
        print_error("Missing required argument: --mode")
        _render_help()
        return 2

    config = _build_config(namespace)
    config["_explicit_options"] = _explicit_option_destinations(parser, args)

    try:
        descriptor = resolve_mode_descriptor(config["mode"])
        if config.get("action") is None:
            config["action"] = descriptor.default_action
        if config.get("output_format") == "json":
            seed_everything(config.get("random_seed"))
        else:
            apply_global_seed(config.get("random_seed"))
        if config.get("output_format") != "json":
            print_startup(
                config["mode"],
                config.get("action"),
                config.get("model") or config.get("model_file"),
            )
        runner = _resolve_mode_runner(config["mode"])
        local_json = (
            config.get("output_format") == "json"
            and config.get("platform", "local") == "local"
        )
        if local_json:
            with redirect_stdout(sys.stderr):
                result = runner(config)
        else:
            result = runner(config)
        if local_json:
            print(json.dumps(json_safe(result), sort_keys=True))
    except UnknownModeError as exc:
        if config.get("output_format") == "json":
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
        else:
            print_error(str(exc))
            _render_unknown_mode()
        return 1
    except MLXAbort:
        if config.get("output_format") == "json":
            print(json.dumps({"error": "Action cancelled."}), file=sys.stderr)
        else:
            print_warning("Action cancelled.")
        return 1
    except MLXUserError as exc:
        if config.get("output_format") == "json":
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
        else:
            print_error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
