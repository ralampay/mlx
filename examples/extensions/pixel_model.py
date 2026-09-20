"""A 1x1 convolution demonstrates dense-prediction construction, not accuracy."""
from torch import nn


def build_segmenter(name, config, *, num_classes):
    return nn.Conv2d(3 if config.get("colored", True) else 1, num_classes, 1)


def build_saliency(name, config):
    return build_segmenter(name, config, num_classes=1)
