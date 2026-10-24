import argparse
from contextlib import redirect_stdout
import json
import os
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
    validate_mode_options,
)
from mlx.core.artifacts import json_safe
from mlx.core.extensions import load_reference
from mlx.core.cli_options import boolean_value as _boolean_value
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
    from mlx.cli_options import OPTIONS
    for option in OPTIONS:
        option.add_to(parser)
    for descriptor in MODE_DESCRIPTORS:
        if descriptor.options:
            for option in load_reference(descriptor.options, kind="CLI options"):
                option.add_to(parser)
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
    usage.add_row("python -m mlx --doctor")
    usage.add_row("python -m mlx --mode object_detection --action ls-models")
    usage.add_row("python -m mlx --mode object_detection --provider libreyolo --action ls-models")
    usage.add_row("python -m mlx --mode object_detection --provider libreyolo --action train --dataset coco8 --model yolo9-t")
    usage.add_row("python -m mlx --mode object_detection --action train --dataset coco8 --model yolo26")
    usage.add_row("python -m mlx --mode object_detection --action benchmark --dataset ./dataset --model-path ./best.pt --split test --output ./benchmark")
    usage.add_row("python -m mlx --mode track --action benchmark --dataset ./benchmarks --model-path ./best.pt --track-class-id 0 --output ./tracking-results")
    usage.add_row("python -m mlx --mode object_detection --platform aws --action train --config ./aws-training.yaml")
    usage.add_row("python -m mlx --mode object_detection --provider libreyolo --action fine-tune --model yolo9-s-drax-b5 --model-path ./best.pt --dataset ./dataset")
    usage.add_row("python -m mlx --mode object_detection --platform aws --action fine-tune --config ./aws-fine-tune.yaml")
    usage.add_row("python -m mlx --mode object_detection --platform aws --action best-model --config ./aws-training.yaml")
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
    usage.add_row("python -m mlx --mode image_recognition_oc --action train --model deep-svdd --backbone resnet18 --dataset ./dataset --output ./artifacts/one-class")
    usage.add_row("python -m mlx --mode image_recognition_oc --action infer-image --model-path ./artifacts/one-class/resnet18-deep-svdd.pth --input-img ./sample.jpg")
    usage.add_row("python -m mlx --mode image_recognition_oc --action benchmark --model-path ./artifacts/one-class/resnet18-deep-svdd.pth --dataset ./dataset --output ./benchmark")
    usage.add_row("python -m mlx --mode image_recognition_oc --platform aws --action train-all --config ./aws-svdd.yaml")
    usage.add_row("python -m mlx --mode image_recognition_oc --platform aws --action benchmark --config ./aws-svdd.yaml")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/siamese --dataset ./omniglot --model siamese-le-net")
    usage.add_row("python -m mlx --mode image_classification --action train --output ./artifacts/siamese-resnet --dataset ./omniglot --model siamese-resnet18 --pretrained")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --output ./dataset --train-count 100 --val-count 20 --test-count 20 --overwrite --seed 42")
    usage.add_row("python -m mlx --mode image_classification --action build-dataset --dataset ./raw-dataset --split-mode ratios --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --output ./dataset --overwrite --seed 42")
    usage.add_row("python -m mlx --mode segmentation --action train --dataset ./dataset --model unet --output unet-seg.pt")
    usage.add_row("python -m mlx --mode segmentation --action train --dataset ./dataset --model unet-resnet18 --pretrained --output ./unet-resnet18")
    usage.add_row("python -m mlx --mode segmentation --action train --dataset ./dataset --model unet-draxnet-sknet --output ./unet-draxnet-sknet")
    usage.add_row("python -m mlx --mode segmentation --action train --model all --dataset-s3-uri s3://my-datasets/segmentation.zip --output ./all-segmentation-models")
    usage.add_row("python -m mlx --mode segmentation --action train --model all-small --dataset ./dataset --transform random-crop --width 512 --height 512 --output ./small-segmentation-models")
    usage.add_row("python -m mlx --mode segmentation --action benchmark --dataset ./dataset --model-path ./unet-seg.pt --output ./benchmark-results")
    usage.add_row("python -m mlx --mode segmentation --action ls-models")
    usage.add_row("python -m mlx --mode segmentation --action infer-image --model-path ./unet-seg.pt --input-img ./sample.jpg")
    usage.add_row("python -m mlx --mode segmentation --action build-dataset --dataset ./raw-segmentation")
    usage.add_row("python -m mlx --mode saliency-mapping --action ls-models")
    usage.add_row("python -m mlx --mode saliency-mapping --action train --model unet --dataset ./saliency-dataset --output ./saliency-run")
    usage.add_row("python -m mlx --mode saliency-mapping --action benchmark --model-path ./saliency-run/unet.pth --dataset ./saliency-dataset --output ./saliency-benchmark")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action train --model resnet18 --backbone-mode 3d --backbone-temporal-kernel-size 3 --clip-length 16 --dataset ./ped2-prepared --output ./artifacts/ped2")
    usage.add_row("python -m mlx --mode video_anomaly_detection --platform aws --action train-all --config ./aws-video-anomaly.yaml")
    usage.add_row("python -m mlx --mode video_anomaly_detection --platform aws --action status --config ./aws-video-anomaly.yaml --job-name JOB_NAME --watch")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action ls-models")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action benchmark --model-path ./model.pth --dataset ./ped2-prepared/test --output ./benchmark")
    usage.add_row("python -m mlx --mode video_anomaly_detection --action infer-video --model-path ./model.pth --file-path ./sample.mp4 --output ./inference")
    usage.add_row("python -m mlx --mode text-embedding --action embed --model ./model.gguf --input ./dataset --output ./artifacts")
    usage.add_row("python -m mlx --mode text-embedding --action benchmark --input ./artifacts --output ./results")
    usage.add_row("python -m mlx --mode autoencoder --action train --model simple --input ./corpus_embeddings.csv --output ./autoencoder --bottleneck-dim 128 --loss mse")
    usage.add_row("python -m mlx --mode autoencoder --action embed --model-path ./autoencoder/autoencoder.pth --input ./corpus_embeddings.csv --output ./corpus_embeddings_ae128.csv")
    usage.add_row("python -m mlx --mode nlp --action embed --model-file ./model.gguf --input-file ./input.csv")
    console.print(usage)

    options = Table(title="Options", show_lines=True)
    options.add_column("Flag", style="cyan", no_wrap=True)
    options.add_column("Default", style="magenta")
    options.add_column("Description", style="white")
    for action in build_parser()._actions:
        if action.option_strings and action.help != argparse.SUPPRESS:
            options.add_row(" / ".join(action.option_strings), str(action.default), action.help or "")
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
    if namespace.doctor:
        from mlx.core.doctor import InspectEnvironment
        from mlx.core.ui import print_environment_report

        report = InspectEnvironment(os.environ).execute()
        if namespace.output_format == "json":
            print(json.dumps(report, sort_keys=True))
        else:
            print_environment_report(report)
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
        validate_mode_options(config)
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
        if config.get("verbose"):
            import traceback

            traceback.print_exc(file=sys.stderr)
        if config.get("output_format") == "json":
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
        else:
            print_error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
