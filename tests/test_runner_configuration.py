"""Runner configuration must preserve Python inputs and CLI explicit-option semantics."""
from importlib import import_module

import pytest

from mlx.core.configuration import with_explicit_options


@pytest.mark.parametrize("metadata, expected", [(None, {"batch_size"}), (set(), set()), ({"epochs"}, {"epochs"})])
def test_explicit_options_are_copied_and_cli_metadata_is_authoritative(metadata, expected):
    original = {"batch_size": 3, "_internal": True}
    if metadata is not None:
        original["_explicit_options"] = metadata
    result = with_explicit_options(original)
    assert result["_explicit_options"] == expected
    result["_explicit_options"].add("changed")
    assert "changed" not in (metadata or ())
    assert original["batch_size"] == 3
    assert ("_explicit_options" in original) is (metadata is not None)


@pytest.mark.parametrize("mode", ["autoencoder", "text_embedding"])
@pytest.mark.parametrize("cli", [False, True])
def test_embedding_runners_preserve_python_batch_size_and_cli_defaults(monkeypatch, mode, cli):
    runner = import_module(f"mlx.modes.{mode}.runner")
    command_name = "TrainAutoencoder" if mode == "autoencoder" else "EmbedTextCommand"
    captured = []

    class Capture:
        def __init__(self, request, **kwargs):
            captured.append(request)

        def execute(self):
            return captured[-1]

    monkeypatch.setattr(runner, command_name, Capture)
    config = {"action": "train" if mode == "autoencoder" else "embed", "batch_size": 3,
              "output_format": "json"}
    if cli:
        config["_explicit_options"] = {"action"}
    original = dict(config)
    result = getattr(runner, f"run_{mode}")(config)
    assert result.batch_size == ((64 if mode == "autoencoder" else 16) if cli else 3)
    assert "_explicit_options" not in result.to_config()
    assert config == original


@pytest.mark.parametrize("mode, action", [
    ("image_recognition_oc", "infer-image"),
    ("video_anomaly_detection", "infer-video"),
])
def test_anomaly_runners_preserve_python_action_and_model(monkeypatch, mode, action):
    runner = import_module(f"mlx.modes.{mode}.runner")
    class Capture:
        def __init__(self, request, **kwargs):
            self.request = request

        def execute(self):
            return self.request

    name = "InferImageOneClass" if mode == "image_recognition_oc" else "InferVideoAnomaly"
    monkeypatch.setattr(runner, name, Capture)
    result = getattr(runner, f"run_{mode}")({
        "action": action, "model": "custom", "batch_size": 3, "output_format": "json",
        "display": False,
    })
    assert result.model == "custom"
    assert result.batch_size == 3
    assert "_explicit_options" not in result.to_config()


def test_autoencoder_python_inference_model_is_retained(monkeypatch):
    from mlx.modes.autoencoder import runner

    class Capture:
        def __init__(self, request, **kwargs):
            self.request = request

        def execute(self):
            return self.request

    monkeypatch.setattr(runner, "EmbedAutoencoder", Capture)
    result = runner.run_autoencoder({"action": "embed", "model": "custom", "output_format": "json"})
    assert result.model == "custom"


@pytest.mark.parametrize("mode", ["segmentation", "saliency_mapping"])
def test_python_training_rejects_two_explicit_dataset_sources(mode):
    from mlx.core.exceptions import MLXUserError

    runner = import_module(f"mlx.modes.{mode}.runner")
    with pytest.raises(MLXUserError, match="both|either|together"):
        getattr(runner, f"run_{mode}")({
            "action": "train", "dataset_path": "local-dataset",
            "dataset_s3_uri": "s3://bucket/data.zip", "output_format": "json",
        })


def test_detection_python_benchmark_preserves_values(monkeypatch):
    from mlx.modes.object_detection import runner

    class Capture:
        def __init__(self, request, **kwargs):
            self.request = request

        def execute(self):
            return self.request

    monkeypatch.setattr(runner, "BenchmarkObjectDetectionModel", Capture)
    result = runner.run_object_detection({
        "action": "benchmark", "confidence": 0.3, "batch_size": 2,
        "height": 128, "width": 256, "verbose": False, "output_format": "json",
    })
    assert (result.confidence, result.batch_size, result.height, result.width, result.verbose) == (
        0.3, 2, 128, 256, False,
    )
