"""Terminal and standalone report rendering, separate from metric computation."""

import json
from html import escape

import numpy as np

from .metrics import acf, spectrum


class RichTemporalReporter:
    def emit(self, event):
        from mlx.core.ui import console

        console.print(event.message, markup=False)


def display_result(result):
    from mlx.core.ui import console

    console.print_json(data=result)


class TemporalReportWriter:
    def write(self, directory, metrics, real, synthetic, metadata):
        # Agg canvas avoids changing the process's global pyplot backend.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        features = metadata["feature_columns"]
        plot_names = []
        for index, name in enumerate(features):
            figure = Figure(figsize=(11, 8))
            FigureCanvasAgg(figure)
            axes = figure.subplots(2, 2)
            mean, scale = metadata["mean"][index], metadata["scale"][index]
            r, s = (
                real[:, :, index] * scale + mean,
                synthetic[:, :, index] * scale + mean,
            )
            for values, label, color in (
                (r, "held-out real", "tab:blue"),
                (s, "synthetic", "tab:orange"),
            ):
                axes[0, 0].hist(
                    values.ravel(),
                    bins=30,
                    density=True,
                    alpha=0.5,
                    label=label,
                    color=color,
                )
                for row in values[:3]:
                    axes[1, 1].plot(row, color=color, alpha=0.5)
            axes[0, 0].set_title(f"{name}: distribution (original units)")
            axes[0, 0].legend()
            for values, label in ((real, "held-out real"), (synthetic, "synthetic")):
                curve = acf(values)[:, index]
                axes[0, 1].plot(np.arange(1, len(curve) + 1), curve, label=label)
                axes[1, 0].plot(spectrum(values)[:, index], label=label)
            axes[0, 1].set_title("Autocorrelation by lag")
            axes[1, 0].set_title("Normalized power spectrum by frequency bin")
            axes[1, 1].set_title("Independent sample trajectories (original units)")
            figure.tight_layout()
            filename = f"feature-{index + 1}.png"
            figure.savefig(directory / filename)
            plot_names.append((name, filename))
            figure.clear()
        rows = "".join(
            f"<tr><td>{escape(key)}</td><td>{'undefined' if value is None else f'{value:.6g}'}</td></tr>"
            for key, value in metrics["fidelity"]["aggregate"].items()
        )
        plots = "".join(
            f'<h2>{escape(name)}</h2><img src="{filename}" alt="Diagnostics for {escape(name, quote=True)}">'
            for name, filename in plot_names
        )
        html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Temporal synthetic quality</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}}td{{padding:8px;border-bottom:1px solid #ddd}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Temporal synthetic-data quality</h1>
<p>Distances use training-standardized values. Smaller fidelity discrepancies are better; variation ratios should be near one.
Compare with the real-versus-real reference. Copying diagnostics do not establish privacy.</p>
<table><tr><th>Aggregate metric</th><th>Value</th></tr>{rows}</table>
{plots}<h2>Full metrics and evaluation protocol</h2><pre>{escape(json.dumps(metrics, indent=2, allow_nan=False))}</pre></html>"""
        (directory / "report.html").write_text(html, encoding="utf-8")
