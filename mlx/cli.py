from pathlib import Path
from typing import Dict, Any, Optional

import typer
from dotenv import load_dotenv

from mlx.platforms import run_module, registered_modules, UnknownModuleError

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

app = typer.Typer(
    help="MLX - Machine Learning eXecutor"
)

@app.command()
def main(
    module: str = typer.Option("chat", help="Module to run (default: chat)"),
    platform: str = typer.Option(None, help="Platform to use (openai, torch, ultralytics)"),
    model: str = typer.Option("gpt-4o-mini", help="Model to use"),
    temperature: float = typer.Option(0.7, help="Creativity / randomness level (0.0-2.0)"),
    top_p: float = typer.Option(0.7, help="Nucleus sampling threshold (0-1)."),
    top_k: int = typer.Option(50, help="Top-k sampling cutoff (>=1)"),
    height: int = typer.Option(256, help="Height of image"),
    width: int = typer.Option(256, help="Width of image"),
    device: str = typer.Option("cpu", help="Device to load model and data on"),
    action: str = typer.Option("test", help="Action for model to take (default: test)"),
    embedding_size: int = typer.Option(4096, help="Embedding size (default: 4096"),
    batch_size: int = typer.Option(1, help="Batch size"),
    dataset_path: str = typer.Option("./tmp/dataset", help="Path for dataset"),
    epochs: int = typer.Option(100, help="Number of epochs"),
    model_path: Optional[str] = typer.Option(
        None,
        help="Path to .pt model. Optional for training; required for camera inference.",
    ),
    file_path: Optional[str] = typer.Option(
        None,
        help="Path to a video file for infer-video action.",
    ),
    input_img: str = typer.Option("/tmp/image.jpg", help="Input image for inference"),
    confidence: float = typer.Option(0.25, help="Confidence threshold for detection"),
    camera_index: int = typer.Option(0, help="Camera index for infer-camera action"),
    pretrained: bool = typer.Option(
        False,
        "--pretrained/--no-pretrained",
        help="Allow Ultralytics to load pretrained weights when only a YAML is provided.",
        show_default=True,
    ),
    lr0: Optional[float] = typer.Option(
        None,
        help="Initial learning rate for Ultralytics training (overrides default when provided).",
    ),
    optimizer: str = typer.Option(
        "auto",
        help="Optimizer to use (auto, sgd, adam, adamw, rmsprop).",
    ),
    nbs: int = typer.Option(
        64,
        help="Nominal batch size used for learning-rate scaling inside Ultralytics.",
    ),
    warmup_epochs: float = typer.Option(
        3.0,
        help="Number of warmup epochs before switching to the main schedule.",
    ),
    amp: bool = typer.Option(
        True,
        "--amp/--no-amp",
        help="Enable mixed-precision (AMP) training inside Ultralytics.",
        show_default=True,
    ),
    loss_clip: Optional[float] = typer.Option(
        None,
        help="Optional gradient clipping value for Ultralytics training (None disables).",
    ),
    local: bool = typer.Option(
        False,
        "--local/--no-local",
        help="Use the local LLM model defined in LOCAL_LLM_MODEL for embedding workloads.",
        show_default=True,
    ),
    chunk_size: int = typer.Option(
        800,
        help="Character count for each chunk when generating embeddings.",
    ),
    chunk_overlap: int = typer.Option(
        100,
        help="Overlap between chunks when splitting documents for embeddings.",
    ),
    table_name: Optional[str] = typer.Option(
        None,
        help="Destination table/collection for RAG vector storage.",
    ),
    file_limit: Optional[int] = typer.Option(
        None,
        help="Maximum number of files to process for RAG utilities.",
    ),

):
    typer.echo(f"MLX starting [module={module}] [platform={platform}] [model={model}]")

    config: Dict[str, Any] = {
        "module": module,
        "platform": platform,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "height": height,
        "width": width,
        "device": device,
        "action": action,
        "embedding_size": embedding_size,
        "batch_size": batch_size,
        "dataset_path": dataset_path,
        "epochs": epochs,
        "model_path": model_path,
        "file_path": file_path,
        "input_img": input_img,
        "input_size": (width, height),
        "confidence": confidence,
        "camera_index": camera_index,
        "pretrained": pretrained,
        "lr0": lr0,
        "optimizer": optimizer,
        "nbs": nbs,
        "warmup_epochs": warmup_epochs,
        "amp": amp,
        "loss_clip": loss_clip,
        "local": local,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "table_name": table_name,
        "file_limit": file_limit,
    }

    try:
        run_module(platform, module, config)
    except UnknownModuleError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        available = registered_modules()
        platform_modules = ", ".join(sorted(available.get(platform, {}).keys()))
        generic_modules = ", ".join(sorted(available.get("generic", {}).keys()))

        if platform_modules:
            typer.secho(f"Available modules for '{platform}': {platform_modules}", fg=typer.colors.YELLOW, err=True)
        if generic_modules:
            typer.secho(f"Platform-agnostic modules: {generic_modules}", fg=typer.colors.YELLOW, err=True)

        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
