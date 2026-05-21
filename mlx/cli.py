import argparse
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from rich.panel import Panel
from rich.table import Table

from mlx.core.exceptions import MLXAbort, MLXUserError
from mlx.core.random import apply_global_seed
from mlx.core.ui import console, print_error, print_startup, print_warning

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)


class CLIUsageError(Exception):
    """Raised when command-line arguments are invalid."""


class UnknownModeError(ValueError):
    """Raised when the selected mode is not registered."""


class RichArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


ModeRunner = Callable[[Dict[str, Any]], Any]

MODE_REGISTRY: Dict[str, str] = {
    "image-classification": "mlx.modes.image_classification.runner:run_image_classification",
    "image_classification": "mlx.modes.image_classification.runner:run_image_classification",
    "object-detection": "mlx.modes.object_detection.ultralytics.runner:run_object_detection",
    "object_detection": "mlx.modes.object_detection.ultralytics.runner:run_object_detection",
    "segmentation": "mlx.modes.segmentation.runner:run_segmentation",
}


def build_parser() -> RichArgumentParser:
    parser = RichArgumentParser(add_help=False, prog="python -m mlx")
    parser.add_argument("-h", "--help", action="store_true", dest="help")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action", default=None)
    parser.add_argument("--embedding-size", type=int, default=4096, dest="embedding_size")
    parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    parser.add_argument("--dataset", "--dataset-path", default="./tmp/dataset", dest="dataset_path")
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument("--file-path", default=None, dest="file_path")
    parser.add_argument("--input-img", default="/tmp/image.jpg", dest="input_img")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--camera-index", type=int, default=0, dest="camera_index")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--seed", "--random-seed", type=int, default=None, dest="random_seed")
    parser.add_argument("--run-name", default=None, dest="run_name")
    parser.add_argument("--num-classes", type=int, default=2, dest="num_classes")
    parser.add_argument("--mask-threshold", type=float, default=0.5, dest="mask_threshold")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, dest="overlay_alpha")
    return parser


def _render_help() -> None:
    console.print(
        Panel.fit(
            "MLX\nA rich-powered CLI for computer-vision workflows.",
            border_style="cyan",
        )
    )

    usage = Table(title="Usage", show_header=False)
    usage.add_column("Command", style="bold cyan")
    usage.add_row("python -m mlx --mode object_detection --action train --dataset coco8 --model yolo26")
    usage.add_row("python -m mlx --mode object_detection --action train --dataset coco8 --model draxnet-yolo26 --output ./runs/draxnet")
    usage.add_row("python -m mlx --mode object_detection --action infer-camera --model draxnet-yolo26 --model-path ./runs/draxnet/exp/weights/best.pt")
    usage.add_row("python -m mlx --mode object_detection --action convert --model-path ./runs/draxnet/exp/weights/best.pt --output ./exports")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/resnet18 --dataset ./dataset --model resnet18")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/siamese --dataset ./omniglot --model siamese-le-net")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --output ./dataset --train-count 100 --val-count 20 --test-count 20 --overwrite --seed 42")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --split-mode ratios --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --output ./dataset --overwrite --seed 42")
    usage.add_row("python -m mlx --mode segmentation --action train --dataset ./dataset --model unet --output unet-seg.pt")
    usage.add_row("python -m mlx --mode segmentation --action infer-image --model-path ./unet-seg.pt --input-img ./sample.jpg")
    usage.add_row("python -m mlx --mode segmentation --action build-dataset --dataset ./raw-segmentation")
    console.print(usage)

    options = Table(title="Options", show_lines=True)
    options.add_column("Flag", style="cyan", no_wrap=True)
    options.add_column("Default", style="magenta")
    options.add_column("Description", style="white")
    options.add_row("--mode", "None", "Mode to run: object_detection, image_classification, or segmentation.")
    options.add_row("--model", "None", "Model identifier, YAML path, or architecture name.")
    options.add_row("--action", "mode-specific", "Sub-action such as train, infer-video, convert, benchmark, or build-dataset.")
    options.add_row("--dataset", "./tmp/dataset", "Dataset source for training: local YOLO root, dataset YAML, or alias like coco8/coco128.")
    options.add_row("--output", "None", "Output directory written by train or benchmark. Detection uses it as the Ultralytics project directory, or as the ONNX export destination for convert.")
    options.add_row("--train-count", "None", "Images per label assigned to the train split when building classification datasets.")
    options.add_row("--val-count", "None", "Images per label assigned to the val split when building classification datasets.")
    options.add_row("--test-count", "None", "Images per label assigned to the test split when building classification datasets.")
    options.add_row("--train-ratio", "None", "Train split ratio applied within each label when building classification datasets.")
    options.add_row("--val-ratio", "None", "Validation split ratio applied within each label when building classification datasets.")
    options.add_row("--test-ratio", "None", "Test split ratio applied within each label when building classification datasets.")
    options.add_row("--split-mode", "None", "Build-dataset split mode: counts or ratios. Ratio mode splits each label independently using the provided ratios.")
    options.add_row("--overwrite / --no-overwrite", "False", "Allow build-dataset to replace an existing output directory without prompting.")
    options.add_row("--model-path", "None", "Weights checkpoint path for inference, warm starts, or ONNX conversion.")
    options.add_row("--file-path", "None", "Video path for file-based inference.")
    options.add_row("--input-img", "/tmp/image.jpg", "Input image for classification inference.")
    options.add_row("--device", "cpu", "Execution device such as cpu or cuda:0.")
    options.add_row("--height / --width", "256 / 256", "Image size controls.")
    options.add_row("--batch-size", "1", "Training or evaluation batch size.")
    options.add_row("--epochs", "100", "Training epoch count.")
    options.add_row("--embedding-size", "4096", "Siamese network embedding size.")
    options.add_row("--confidence", "0.25", "Detection confidence threshold.")
    options.add_row("--camera-index", "0", "Camera index for webcam inference.")
    options.add_row("--pretrained / --no-pretrained", "False", "Toggle Ultralytics pretrained initialization.")
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
    options.add_row("--amp / --no-amp", "True", "Toggle mixed precision for Ultralytics training.")
    options.add_row("--lr0", "None", "Override initial learning rate.")
    options.add_row("--optimizer", "auto", "Optimizer selection for Ultralytics.")
    options.add_row("--nbs", "64", "Nominal batch size for LR scaling.")
    options.add_row("--warmup-epochs", "3.0", "Warmup epoch count.")
    options.add_row("--loss-clip", "None", "Optional gradient clipping value.")
    options.add_row("--seed / --random-seed", "None", "Global random seed applied across Python, NumPy, and PyTorch.")
    options.add_row("--run-name", "None", "Optional Ultralytics run folder name.")
    options.add_row("--num-classes", "2", "Number of segmentation classes expected in the masks.")
    options.add_row("--mask-threshold", "0.5", "Threshold used when rendering binary segmentation masks.")
    options.add_row("--overlay-alpha", "0.45", "Blend strength for segmentation overlays.")
    options.add_row("--help", "False", "Show this help screen.")
    console.print(options)

    available = Table(title="Available Modes", show_header=True)
    available.add_column("Mode", style="cyan", no_wrap=True)
    available.add_column("Actions", style="white")
    available.add_row("object_detection", "train, infer-camera, infer-video, convert")
    available.add_row("image_classification", "train, test, benchmark, infer-image, build-dataset")
    available.add_row("segmentation", "train, test, infer-image, infer-camera, infer-video, build-dataset")
    console.print(available)


def _build_config(namespace: argparse.Namespace) -> Dict[str, Any]:
    config = vars(namespace).copy()
    config.pop("help", None)
    if config.get("mode"):
        config["mode"] = config["mode"].replace("-", "_")
    config["input_size"] = (config["width"], config["height"])
    return config


def _resolve_mode_runner(mode: str) -> ModeRunner:
    dotted_path = MODE_REGISTRY.get(mode)
    if dotted_path is None:
        raise UnknownModeError(f"Unknown mode '{mode}'.")

    module_path, func_name = dotted_path.split(":")
    module = import_module(module_path)
    return getattr(module, func_name)


def _render_unknown_mode() -> None:
    table = Table(title="Available Modes", show_header=True)
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Purpose", style="white")
    table.add_row("object_detection", "Ultralytics-backed detection training and inference")
    table.add_row("image_classification", "Image classification workflows for both one-shot and standard classifiers")
    table.add_row("segmentation", "Semantic segmentation workflows for U-Net style models")
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
    apply_global_seed(config.get("random_seed"))
    print_startup(config["mode"], config.get("action"), config["model"])

    try:
        runner = _resolve_mode_runner(config["mode"])
        runner(config)
    except UnknownModeError as exc:
        print_error(str(exc))
        _render_unknown_mode()
        return 1
    except MLXAbort:
        print_warning("Action cancelled.")
        return 1
    except MLXUserError as exc:
        print_error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
