"""Lightweight CLI option declarations; shared flags retain historical defaults."""
from mlx.core.cli_options import CliOption, boolean_value

OPTIONS = (
    CliOption(('--tracker',), {
        'default': 'bytetrack',
        'help': 'Tracking algorithm alias or external package.module:ClassName.',
    }),
    CliOption(('--real-time-results',), {
        'type': boolean_value,
        'nargs': '?',
        'const': True,
        'default': True,
    }),
    CliOption(('--no-real-time-results',), {
        'action': 'store_false',
        'dest': 'real_time_results',
    }),
    CliOption(('--tracker-config',), {
        'default': None,
        'dest': 'tracker_config',
        'help': 'Optional JSON object file passed as keyword arguments to the tracker constructor.',
    }),
    CliOption(('--tracking-jsonl',), {
        'default': None,
        'dest': 'tracking_jsonl',
        'help': 'Class-aware tracks.jsonl input used by track --action export-mot.',
    }),
    CliOption(('--ground-truth', '--gt-file'), {
        'default': None,
        'dest': 'ground_truth',
        'help': 'Optional 10-column MOTChallenge ground-truth file used for tracking benchmarks.',
    }),
    CliOption(('--track-class-id',), {
        'action': 'append',
        'type': int,
        'default': None,
        'dest': 'track_class_ids',
        'help': 'Repeatable detector class ID included in tracking; all classes are used by default.',
    }),
    CliOption(('--benchmark-iou',), {
        'type': float,
        'default': 0.5,
        'dest': 'benchmark_iou',
        'help': 'Minimum box IoU used for MOT benchmark matching.',
    }),
)
