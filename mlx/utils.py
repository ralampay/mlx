# Helper: render matplotlib loss curve
import matplotlib.pyplot as plt
from rich.panel import Panel
import numpy as np

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
