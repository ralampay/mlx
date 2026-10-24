"""Lightweight CLI option declarations; shared flags retain historical defaults."""
from mlx.core.cli_options import CliOption

OPTIONS = (
    CliOption(('--clip-length',), {
        'type': int,
        'default': 16,
        'dest': 'clip_length',
        'help': 'Frames per video-anomaly clip window.',
    }),
    CliOption(('--frame-stride',), {
        'type': int,
        'default': 1,
        'dest': 'frame_stride',
        'help': 'Sampling stride between frames in a video-anomaly clip.',
    }),
    CliOption(('--backbone-mode',), {
        'choices': ('3d', 'frame-2d'),
        'default': '3d',
        'dest': 'backbone_mode',
        'help': 'Clip-native 3D backbone, or legacy frame-2d plus TCN.',
    }),
    CliOption(('--backbone-temporal-kernel-size',), {
        'type': int,
        'default': 3,
        'dest': 'backbone_temporal_kernel_size',
        'help': 'Odd temporal kernel used to inflate spatial backbone convolutions.',
    }),
    CliOption(('--temporal-model',), {
        'default': 'tcn',
        'dest': 'temporal_model',
        'help': 'Legacy frame-2d temporal encoder alias.',
    }),
    CliOption(('--temporal-hidden-dim',), {
        'type': int,
        'default': 256,
        'dest': 'temporal_hidden_dim',
        'help': 'Legacy frame-2d temporal CNN hidden channels.',
    }),
    CliOption(('--temporal-embedding-dim',), {
        'type': int,
        'default': 128,
        'dest': 'temporal_embedding_dim',
        'help': 'Legacy frame-2d temporal CNN output width.',
    }),
    CliOption(('--temporal-kernel-size',), {
        'type': int,
        'default': 3,
        'dest': 'temporal_kernel_size',
        'help': 'Legacy frame-2d Conv1d kernel size.',
    }),
    CliOption(('--temporal-dropout',), {
        'type': float,
        'default': 0.0,
        'dest': 'temporal_dropout',
        'help': 'Legacy frame-2d temporal CNN dropout.',
    }),
    CliOption(('--frame-aggregation',), {
        'choices': ('mean', 'max'),
        'default': 'mean',
        'dest': 'frame_aggregation',
        'help': 'Aggregate overlapping window scores per frame using mean or max.',
    }),
)
