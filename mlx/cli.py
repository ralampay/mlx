from typing import Dict, Any

import typer

from mlx.platforms import run_module, registered_modules, UnknownModuleError

app = typer.Typer(
    help="MLX - Machine Learning eXecutor"
)

@app.command()
def main(
    module: str = typer.Option("chat", help="Module to run (default: chat)"),
    platform: str = typer.Option("openai", help="Platform to use (openai, torch, ultralytics)"),
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
    model_path: str = typer.Option("/tmp/sample.pt", help="Path to .pt model"),
    input_img: str = typer.Option("/tmp/image.jpg", help="Input image for inference"),
    confidence: float = typer.Option(0.25, help="Confidence threshold for detection"),
    camera_index: int = typer.Option(0, help="Camera index for infer-camera action"),

):
    typer.echo(f"MLX starting [module={module}] [platform={platform}] [model={model}]")

    config: Dict[str, Any] = {
        "module": module,
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
        "input_img": input_img,
        "input_size": (width, height),
        "confidence": confidence,
        "camera_index": camera_index,
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
