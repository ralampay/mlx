import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rich.panel import Panel
from rich.table import Table

from mlx.core.exceptions import MLXAbort, MLXUserError
from mlx.platforms import UnknownModuleError, registered_modules, run_module
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
    parser = RichArgumentParser(add_help=False, prog="mlx")
    parser.add_argument("-h", "--help", action="store_true", dest="help")
    parser.add_argument("--module", default="system")
    parser.add_argument("--platform", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action", default="ls-env")
    parser.add_argument("--embedding-size", type=int, default=4096, dest="embedding_size")
    parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    parser.add_argument("--dataset-path", default="./tmp/dataset", dest="dataset_path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model-path", default=None, dest="model_path")
    parser.add_argument("--file-path", default=None, dest="file_path")
    parser.add_argument("--input-img", default="/tmp/image.jpg", dest="input_img")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--camera-index", type=int, default=0, dest="camera_index")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--warmup-epochs", type=float, default=3.0, dest="warmup_epochs")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-clip", type=float, default=None, dest="loss_clip")
    parser.add_argument("--run-name", default=None, dest="run_name")
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
    usage.add_row("mlx --module system --action ls-env")
    usage.add_row("mlx --module obj-detect --platform ultralytics --action train --dataset-path ./dataset --model ultralytics/cfg/models/ext/cad_yolo12.yaml")
    usage.add_row("mlx --module ic-one-shot --platform torch --action train --dataset-path ./omniglot")
    console.print(usage)

    options = Table(title="Options", show_lines=True)
    options.add_column("Flag", style="cyan", no_wrap=True)
    options.add_column("Default", style="magenta")
    options.add_column("Description", style="white")
    options.add_row("--module", "system", "Module to run: system, obj-detect, ic-one-shot.")
    options.add_row("--platform", "generic", "Platform backend: ultralytics or torch.")
    options.add_row("--model", "None", "Model identifier, YAML path, or architecture name.")
    options.add_row("--action", "ls-env", "Module action such as train, infer-video, or test.")
    options.add_row("--dataset-path", "./tmp/dataset", "Dataset root used by training and dataset utilities.")
    options.add_row("--model-path", "None", "Weights checkpoint path for inference or warm starts.")
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
    options.add_row("--amp / --no-amp", "True", "Toggle mixed precision for Ultralytics training.")
    options.add_row("--lr0", "None", "Override initial learning rate.")
    options.add_row("--optimizer", "auto", "Optimizer selection for Ultralytics.")
    options.add_row("--nbs", "64", "Nominal batch size for LR scaling.")
    options.add_row("--warmup-epochs", "3.0", "Warmup epoch count.")
    options.add_row("--loss-clip", "None", "Optional gradient clipping value.")
    options.add_row("--run-name", "None", "Optional Ultralytics run folder name.")
    options.add_row("--help", "False", "Show this help screen.")
    console.print(options)

    modules = registered_modules()
    available = Table(title="Available Modules", show_header=True)
    available.add_column("Scope", style="cyan", no_wrap=True)
    available.add_column("Modules", style="white")
    for scope, entries in sorted(modules.items()):
        available.add_row(str(scope), ", ".join(sorted(entries.keys())))
    console.print(available)


def _build_config(namespace: argparse.Namespace) -> Dict[str, Any]:
    config = vars(namespace).copy()
    config.pop("help", None)
    config["input_size"] = (config["width"], config["height"])
    return config


def _render_unknown_module(platform: Optional[str]) -> None:
    available = registered_modules()
    platform_modules = ", ".join(sorted(available.get(platform, {}).keys()))
    generic_modules = ", ".join(sorted(available.get("generic", {}).keys()))

    table = Table(title="Available Modules", show_header=True)
    table.add_column("Scope", style="cyan", no_wrap=True)
    table.add_column("Modules", style="white")

    if platform_modules:
        table.add_row(f"Platform '{platform}'", platform_modules)
    if generic_modules:
        table.add_row("Generic", generic_modules)

    if table.row_count:
        console.print(table)
    else:
        print_warning("No modules are registered for the requested platform.")


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

    config = _build_config(namespace)
    print_startup(config["module"], config["platform"], config["model"])

    try:
        run_module(config["platform"], config["module"], config)
    except UnknownModuleError as exc:
        print_error(str(exc))
        _render_unknown_module(config["platform"])
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
