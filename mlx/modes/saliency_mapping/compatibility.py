"""The deliberate gateway to segmentation-owned construction and image policies."""


def segmentation_inventory():
    from mlx.modes.segmentation.models import SMALL_MODEL_NAMES, supported_model_names
    return tuple(supported_model_names()), frozenset(SMALL_MODEL_NAMES)


def build_segmentation_saliency(model_name, config):
    from mlx.modes.segmentation.models import build_segmentation_model
    return build_segmentation_model(model_name, config, num_classes=1)


def segmentation_metadata(model_name):
    from mlx.modes.segmentation.models.backbones import BACKBONE_SPECS
    spec = BACKBONE_SPECS.get(model_name)
    return (
        "native" if spec is None else spec.classification_model,
        spec is not None and spec.classification_model != "draxnet",
    )


def normalize_segmentation_transform(value):
    from mlx.modes.segmentation.data import normalize_segmentation_transform as normalize
    return normalize(value)


def evaluation_segmentation_transform(value):
    from mlx.modes.segmentation.data import evaluation_segmentation_transform as evaluate
    return evaluate(value)


def image_extensions():
    from mlx.modes.segmentation.data import IMAGE_EXTENSIONS
    return IMAGE_EXTENSIONS


def evenly_spaced_sample_indices(count, limit):
    from mlx.modes.segmentation.samples import evenly_spaced_sample_indices as indices
    return indices(count, limit)
