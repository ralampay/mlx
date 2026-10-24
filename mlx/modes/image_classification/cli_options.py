"""Lightweight CLI option declarations; shared flags retain historical defaults."""
import argparse
from mlx.core.cli_options import CliOption

OPTIONS = (
    CliOption(('--embedding-size',), {
        'type': int,
        'default': 4096,
        'dest': 'embedding_size',
        'help': 'Embedding width for any one-shot Siamese model.',
    }),
    CliOption(('--num-pairs',), {
        'type': int,
        'default': 100,
        'dest': 'num_pairs',
        'help': 'One-shot image-classification pairs per label for training or benchmarking.',
    }),
    CliOption(('--ood-method',), {
        'choices': ('none', 'deep-svdd'),
        'default': 'none',
        'dest': 'ood_method',
        'help': 'Optional standard-classifier OOD method: none or deep-svdd.',
    }),
    CliOption(('--cam-method',), {
        'default': 'gradcam',
        'dest': 'cam_method',
        'help': 'Registered CAM method or an explicit package.module:Class reference.',
    }),
    CliOption(('--target-layer',), {
        'default': None,
        'dest': 'target_layer',
        'help': 'Optional dotted module path to explain, such as layer4.1 or features.-1.',
    }),
    CliOption(('--target-index',), {
        'type': int,
        'default': None,
        'dest': 'target_index',
        'help': 'Optional class index or Siamese output index to explain. Defaults to model prediction.',
    }),
    CliOption(('--max-samples',), {
        'type': int,
        'default': None,
        'dest': 'max_samples',
        'help': 'Maximum number of test samples or one-shot pairs to render.',
    }),
    CliOption(('--save-images',), {
        'action': argparse.BooleanOptionalAction,
        'default': True,
        'dest': 'save_images',
        'help': 'Write rendered CAM images under --output when provided.',
    }),
    CliOption(('--window-delay',), {
        'type': int,
        'default': 0,
        'dest': 'window_delay',
        'help': 'OpenCV waitKey delay in milliseconds between displayed CAM images.',
    }),
    CliOption(('--aug-smooth',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'dest': 'aug_smooth',
        'help': 'Apply Grad-CAM test-time augmentation smoothing.',
    }),
    CliOption(('--eigen-smooth',), {
        'action': argparse.BooleanOptionalAction,
        'default': False,
        'dest': 'eigen_smooth',
        'help': 'Apply Grad-CAM Eigen smoothing.',
    }),
)
