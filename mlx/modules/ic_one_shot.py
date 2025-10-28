import torch
import typer
from rich.table import Table
from rich.console import Console
from mlx.modules.ic.siamese_le_net import SiameseLeNet
from mlx.modules.data_builder import build_ic_one_shot

console = Console()

def run_ic_one_shot(model, **kwargs):
    defaults = {
        "action": "test",
        "device": "cpu",
        "embedding_size": 4096,
        "input_size": (105, 105),
        "batch_size": 1,
        "dataset_path": ""
    }

     # Merge defaults with kwargs (user overrides)
    config = {**defaults, **kwargs}

    # Display configuration summary
    _print_config_summary(model, config)

    if model == "siamese-le-net":
        net = SiameseLeNet(colored=True, embedding_size=config["embedding_size"])
    else:
        raise ValueError(f"Invalid model {model}")

    if config["action"] == "test":
        _test_model(net, config)
    elif config["action"] == "build-dataset":
        build_ic_one_shot(config["dataset_path"])
    else:
        raise ValueError(f"Unsupported action {config['action']}")

def _print_config_summary(model: str, config: dict):
    """Print a nicely formatted configuration summary."""
    table = Table(title=f"Configuration for {model}", show_lines=True)
    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)

def _test_model(net, config):
    """Run test with random tensors using resolved config."""
    batch   = config["batch_size"]
    h, w    = config["input_size"]
    device  = config["device"]

    typer.secho(
        f"Running test on device={device} | input={h}x{w} | batch={batch}",
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Assume colored
    x1 = torch.randn(batch, 3, h, w).to(device)
    x2 = torch.randn(batch, 3, h, w).to(device)

    out = net(x1, x2)

    typer.secho("\nTest completed successfully!", fg=typer.colors.BRIGHT_GREEN, bold=True)
    typer.secho(f"Output tensor shape: {list(out.shape)}\n", fg=typer.colors.BRIGHT_GREEN)

    # Display output in compact table
    table = Table(title="Model Output", show_header=True)
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="magenta")

    for i, val in enumerate(out.flatten().tolist()):
        table.add_row(str(i), f"{val:.6f}")

    console.print(table)
