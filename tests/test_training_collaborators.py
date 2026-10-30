"""Native commands accept synthetic data without a filesystem dataset layout."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from mlx.core.training_data import TrainingDatasets
from examples.extensions.pixel_model import build_segmenter, build_saliency


@pytest.mark.parametrize("mode", ["segmentation", "saliency"])
def test_train_with_injected_partitions_optimizer_and_samples(tmp_path, mode):
    calls = []
    source = object()
    images = torch.rand(2, 3, 8, 8)
    targets = (torch.zeros(2, 8, 8, dtype=torch.long) if mode == "segmentation"
               else torch.rand(2, 1, 8, 8))
    dataset = TensorDataset(images, targets)

    def optimizer(parameters, **kwargs):
        calls.append("optimizer")
        return torch.optim.SGD(parameters, **kwargs)

    def samples(config, **kwargs):
        assert kwargs.get("test_split_path", kwargs.get("split_path")) is source
        assert kwargs["checkpoint_path"].is_file()
        return SimpleNamespace(execute=lambda: calls.append("samples"))

    common = dict(model="pixel", dataset_path=str(tmp_path / "absent"),
                  output_path=str(tmp_path / "out"), epochs=1, workers=0,
                  input_size=(8, 8), batch_size=2)
    injected = dict(dataset_loader=lambda request: TrainingDatasets(dataset, dataset, source),
                    optimizer_factory=optimizer, sample_factory=samples)
    if mode == "segmentation":
        from mlx.modes.segmentation.train import TrainSegmentationModel
        from mlx.modes.segmentation.requests import TrainSegmentationRequest
        from mlx.modes.segmentation.models import SegmentationModelRegistry
        result = TrainSegmentationModel(TrainSegmentationRequest(**common),
            model_registry=SegmentationModelRegistry({}).register("pixel", build_segmenter),
            **injected).execute()
        assert result.checkpoint_path.is_file()
        assert result.best_validation_loss < float("inf")
    else:
        from mlx.modes.saliency_mapping.train import TrainSaliencyModel
        from mlx.modes.saliency_mapping.requests import TrainSaliencyRequest
        from mlx.modes.saliency_mapping.models import SaliencyModelRegistry
        result = TrainSaliencyModel(TrainSaliencyRequest(**common, threshold_steps=3),
            model_registry=SaliencyModelRegistry({}).register("pixel", build_saliency),
            **injected).execute()
        assert result["checkpoint_path"].is_file()
    assert calls == ["optimizer", "samples"]
    assert not (tmp_path / "absent").exists()
