"""Shared loader configuration validates explicit concurrency without spawning processes."""
import pytest

from mlx.core.configuration import resolve_data_loader_workers
from mlx.core.exceptions import MLXUserError
from mlx.modes.image_classification.requests import TrainImageClassificationRequest
from mlx.modes.segmentation.requests import BenchmarkSegmentationRequest


@pytest.mark.parametrize('value', [0, 1, 3, '2'])
def test_explicit_concurrency_survives_request_roundtrip(value):
    for request_type in (TrainImageClassificationRequest, BenchmarkSegmentationRequest):
        request = request_type.from_config({'workers': value})
        assert resolve_data_loader_workers(request.to_config()) == int(value)


@pytest.mark.parametrize('value', [-1, 'invalid', None, True, 1.5])
def test_invalid_worker_counts_are_actionable(value):
    with pytest.raises(MLXUserError, match='workers'):
        resolve_data_loader_workers({'workers': value})
