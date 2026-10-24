"""Lightweight CLI option declarations; shared flags retain historical defaults."""
import argparse
from mlx.core.cli_options import CliOption

OPTIONS = (
    CliOption(('--input-dim',), {
        'type': int,
        'default': None,
        'dest': 'input_dim',
        'help': 'Optional expected vector width for autoencoder training.',
    }),
    CliOption(('--hidden-dim',), {
        'type': int,
        'default': 256,
        'dest': 'hidden_dim',
        'help': 'Hidden layer width for the simple autoencoder.',
    }),
    CliOption(('--bottleneck-dim',), {
        'type': int,
        'default': 128,
        'dest': 'bottleneck_dim',
        'help': 'Autoencoder bottleneck representation width.',
    }),
    CliOption(('--autoencoder-config',), {
        'default': None,
        'dest': 'autoencoder_config',
        'help': 'Optional JSON object with custom autoencoder-definition options.',
    }),
    CliOption(('--normalize-inputs',), {
        'action': argparse.BooleanOptionalAction,
        'default': None,
        'dest': 'normalize_inputs',
        'help': 'Control autoencoder training input L2 normalization; defaults to source-manifest provenance.',
    }),
    CliOption(('--output-dim',), {
        'type': int,
        'default': None,
        'help': 'Supported ordered checkpoint prefix for autoencoder export.',
    }),
)
