"""Lightweight CLI option declarations; shared flags retain historical defaults."""
from mlx.core.cli_options import CliOption

OPTIONS = (
    CliOption(('--bce-weight',), {
        'type': float,
        'default': 1.0,
        'dest': 'bce_weight',
        'help': 'Weights for the saliency BASNet-style hybrid objective.',
    }),
    CliOption(('--ssim-weight',), {
        'type': float,
        'default': 1.0,
        'dest': 'ssim_weight',
    }),
    CliOption(('--iou-weight',), {
        'type': float,
        'default': 1.0,
        'dest': 'iou_weight',
    }),
)
