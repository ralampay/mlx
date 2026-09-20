import csv
import json

import numpy as np
from PIL import Image

from examples.extensions.pixel_model import build_segmenter, build_saliency


def paired_dataset(root):
    for split in ("train", "val"):
        for kind in ("images", "masks"):
            (root / split / kind).mkdir(parents=True)
        Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(root / split / "images/a.png")
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / split / "masks/a.png")
    return root / "val/images/a.png"


def test_custom_segmenter_trains_reloads_and_infers(tmp_path):
    from mlx.modes.segmentation.models.registry import SegmentationModelRegistry
    from mlx.modes.segmentation.train import TrainSegmentationModel
    from mlx.modes.segmentation.inference import InferSegmentationImage
    from mlx.modes.segmentation.requests import TrainSegmentationRequest, InferSegmentationRequest
    from mlx.modes.segmentation.list_models import ListSegmentationModels
    image = paired_dataset(tmp_path / "data")
    registry = SegmentationModelRegistry({}).register("pixel", build_segmenter)
    assert [item.model_name for item in ListSegmentationModels({}, model_registry=registry).execute()] == ["pixel"]
    TrainSegmentationModel(TrainSegmentationRequest(
        model="pixel", dataset_path=str(tmp_path / "data"), output_path=str(tmp_path / "run"),
        epochs=1, batch_size=1, input_size=(8, 8),
    ), model_registry=registry).execute()
    result = InferSegmentationImage(InferSegmentationRequest(
        model_path=str(tmp_path / "run/pixel.pth"), input_img=str(image),
    ), model_registry=registry).execute()
    assert result["predicted_mask"].shape == (8, 8)


def test_custom_saliency_trains_reloads_and_infers(tmp_path):
    from mlx.modes.saliency_mapping.models import SaliencyModelRegistry
    from mlx.modes.saliency_mapping.train import TrainSaliencyModel
    from mlx.modes.saliency_mapping.inference import InferSaliencyImage
    from mlx.modes.saliency_mapping.requests import TrainSaliencyRequest, SaliencyRequest
    from mlx.modes.saliency_mapping.list_models import ListSaliencyModels
    image = paired_dataset(tmp_path / "data")
    registry = SaliencyModelRegistry({}).register("pixel", build_saliency)
    assert [item.model_name for item in ListSaliencyModels({}, model_registry=registry).execute()] == ["pixel"]
    TrainSaliencyModel(TrainSaliencyRequest(
        model="pixel", dataset_path=str(tmp_path / "data"), output_path=str(tmp_path / "run"),
        epochs=1, batch_size=1, input_size=(8, 8), plots=False,
    ), model_registry=registry).execute()
    result = InferSaliencyImage(SaliencyRequest(
        model_path=str(tmp_path / "run/pixel.pth"), input_img=str(image),
        output_path=str(tmp_path / "inference"),
    ), model_registry=registry).execute()
    assert result["probability_map"].shape == (8, 8)


def test_autoencoder_with_locally_registered_loss(tmp_path):
    from mlx.modes.autoencoder.commands import TrainAutoencoder
    from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
    from mlx.modes.autoencoder.losses import ReconstructionLossRegistry
    from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
    loss_registry = ReconstructionLossRegistry({}).register(
        "absolute", "examples.extensions.absolute_error:AbsoluteErrorDefinition",
    )
    path = tmp_path / "vectors.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "embedding"])
        writer.writerows((str(i), json.dumps([i, 1., 2.])) for i in range(4))
    result = TrainAutoencoder(AutoencoderTrainRequest(
        model="tiny", loss="absolute", input_path=str(path), output_path=str(tmp_path / "run"),
        bottleneck_dim=1, epochs=1, batch_size=2, plots=False,
    ), loss_registry=loss_registry).execute()
    adapter = AutoencoderRepresentationTransformer(result.checkpoint_path)
    assert len(adapter.transform([[0., 1., 2.]])[0]) == 1
