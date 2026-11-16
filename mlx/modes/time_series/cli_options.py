"""Lightweight mode-specific options; no model imports during CLI discovery."""

from mlx.core.cli_options import CliOption

OPTIONS = tuple(
    CliOption((flag,), {"default": default, "type": kind, "help": help_text})
    for flag, default, kind, help_text in (
        (
            "--time-column",
            None,
            str,
            "Temporal CSV numeric or ISO-8601 timestamp column.",
        ),
        (
            "--feature-columns",
            None,
            str,
            "Comma-separated numeric series columns, in model feature order.",
        ),
        (
            "--series-id-column",
            None,
            str,
            "Optional CSV column identifying independent temporal series.",
        ),
        (
            "--window-size",
            128,
            int,
            "Temporal sequence length in timesteps (minimum 4).",
        ),
        (
            "--stride",
            32,
            int,
            "Training-window stride; evaluation uses non-overlapping windows.",
        ),
        ("--latent-dim", 16, int, "Temporal VAE latent dimension."),
        (
            "--hidden-dims",
            "64,128",
            str,
            "Comma-separated dense widths or temporal convolutional channels.",
        ),
        ("--kl-weight", 1.0, float, "Temporal VAE KL-divergence objective weight."),
        ("--model-config", None, str, "Path to temporal model-specific JSON options."),
        (
            "--num-sequences",
            1000,
            int,
            "Number of independent synthetic sequences to sample.",
        ),
        ("--synthetic-data", None, str, "Synthetic sequence CSV to benchmark."),
        (
            "--max-metric-windows",
            512,
            int,
            "Maximum windows per partition in quality metrics.",
        ),
    )
)
