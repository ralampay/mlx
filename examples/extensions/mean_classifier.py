from torch import nn


def build_classifier(*, num_classes, colored, pretrained, config=None):
    if pretrained:
        raise ValueError("This example has no pretrained weights.")
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(3 if colored else 1, num_classes),
    )
