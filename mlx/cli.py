import typer
from mlx.modules.chat import run_chat

app = typer.Typer(
    help="MLX - Machine Learning eXecutor"
)

@app.command()
def main(
    module: str = typer.Option("chat", help="Module to run (chat)"),
    platform: str = typer.Option("openai", help="Platform to use (openai, aws)"),
    model: str = typer.Option("gpt-4o-mini", help="Model to use"),
    temperature: float = typer.Option(0.7, help="Creativity / randomness level (0.0-2.0)"),
    top_p: float = typer.Option(0.7, help="Nucleus sampling threshold (0-1)."),
    top_k: int = typer.Option(50, help="Top-k sampling cutoff (>=1)")
):
    typer.echo(f"MLX starting [module={module}] [platform={platform}] [model={model}]")

    if module == "chat":
        run_chat(platform, model, temperature, top_p, top_k)
    else:
        typer.echo(f"Unkown module: {module}")

if __name__ == "__main__":
    app()
