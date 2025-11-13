# Helper: render matplotlib loss curve
import matplotlib.pyplot as plt
from rich.panel import Panel
import numpy as np
import cv2

def render_loss_plot(train_losses, val_losses, epochs, width=60, height=15):
    """Render ASCII graph of train/val losses."""
    if len(train_losses) == 0:
        return Panel("No data yet...", border_style="red")

    # Normalize and scale values
    max_loss = max(max(train_losses), max(val_losses))
    min_loss = min(min(train_losses), min(val_losses))
    rng = max_loss - min_loss if max_loss != 0 else 1.0

    def scale(values):
        return [int((v - min_loss) / rng * (height - 1)) for v in values]

    t_scaled = scale(train_losses)
    v_scaled = scale(val_losses)

    canvas = np.full((height, width), " ", dtype=str)
    for i, y in enumerate(t_scaled[-width:]):
        canvas[height - 1 - y, i] = "•"  # train
    for i, y in enumerate(v_scaled[-width:]):
        if canvas[height - 1 - y, i] == "•":
            canvas[height - 1 - y, i] = "@"  # overlap
        else:
            canvas[height - 1 - y, i] = "+"

    graph = "\n".join("".join(row) for row in canvas)
    return Panel(graph, title=f"Loss Curve (Epoch {len(train_losses)}/{epochs})", border_style="cyan")

def _resolve_model_paths(
    config: Dict[str, Any],
    require_yaml: bool,
    require_weights: bool,
) -> Tuple[Optional[Path], Optional[Path]]:
    model_cfg = config.get("model")
    resolved_cfg = Path(_resolve_weights_source(model_cfg)) if model_cfg else None
    if require_yaml and resolved_cfg is None:
        raise typer.BadParameter("This action requires --model pointing to the model YAML.")
    if resolved_cfg and not resolved_cfg.exists():
        raise typer.BadParameter(f"Model YAML not found: {resolved_cfg}")

    weights_path = config.get("model_path")
    resolved_weights = Path(_resolve_weights_source(weights_path)) if weights_path else None
    if require_weights and resolved_weights is None:
        raise typer.BadParameter("This action requires --model-path pointing to trained weights (.pt).")
    if resolved_weights and not resolved_weights.exists():
        raise typer.BadParameter(f"Model weights not found: {resolved_weights}")

    return resolved_cfg, resolved_weights
