import typer
from mlx.modules.chat import run_chat
from mlx.modules.ic_one_shot import run_ic_one_shot

app = typer.Typer(
    help="MLX - Machine Learning eXecutor"
)

@app.command()
def main(
    module: str = typer.Option("chat", help="Module to run (default: chat)"),
    platform: str = typer.Option("openai", help="Platform to use (openai, aws)"),
    model: str = typer.Option("gpt-4o-mini", help="Model to use"),
    temperature: float = typer.Option(0.7, help="Creativity / randomness level (0.0-2.0)"),
    top_p: float = typer.Option(0.7, help="Nucleus sampling threshold (0-1)."),
    top_k: int = typer.Option(50, help="Top-k sampling cutoff (>=1)"),
    height: int = typer.Option(105, help="Height of image (default: 105"),
    width: int = typer.Option(105, help="Width of image (default: 105"),
    device: str = typer.Option("cpu", help="Device to load model and data on"),
    action: str = typer.Option("test", help="Action for model to take (default: test)"),
    embedding_size: int = typer.Option(4096, help="Embedding size (default: 4096"),
    batch_size: int = typer.Option(1, help="Batch size"),
    dataset_path: str = typer.Option("./tmp/dataset", help="Path for dataset"),
    epochs: int = typer.Option(100, help="Number of epochs"),
    model_path: str = typer.Option("/tmp/sample.pt", help="Path to .pt model"),
    input_img: str = typer.Option("/tmp/image.jpg", help="Input image for inference"),

):
    typer.echo(f"MLX starting [module={module}] [platform={platform}] [model={model}]")

    if module == "chat":
        run_chat(platform, model, temperature, top_p, top_k)
    elif module == "ic-one-shot":
        run_ic_one_shot(model, 
            action=action,
            device=device,
            input_size=(width, height),
            embedding_size=embedding_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            epochs=epochs,
            model_path=model_path,
            input_img=input_img
        )
    else:
        typer.echo(f"Unkown module: {module}")

if __name__ == "__main__":
    app()
