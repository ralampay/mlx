from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import RandomSampler, SequentialSampler

from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.commands import TrainAutoencoder
from mlx.modes.autoencoder.data import MergeSingletonBatchSampler
from mlx.modes.autoencoder.losses import (
    DEFAULT_LOSS_REGISTRY, MSESimilarityLoss, SimilarityPreservingLoss,
    ReconstructionLossRegistry,
)
from mlx.modes.autoencoder.requests import AutoencoderTrainRequest
from test_autoencoder import write_embeddings


def test_similarity_formula_and_different_dimensions():
    x = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    z = torch.tensor([[1., 0.], [0.6, 0.8]])
    assert SimilarityPreservingLoss()(x, z).item() == pytest.approx(0.36)
    assert MSESimilarityLoss(2)(x + 1, x, latent=z).item() == pytest.approx(1.72)


def test_diagonal_is_excluded_even_for_zero_vectors():
    # Off-diagonal similarities are zero in both spaces, despite differing diagonals.
    x = torch.zeros(2, 3)
    z = torch.eye(2)
    assert SimilarityPreservingLoss()(x, z).item() == 0


def test_rotation_and_scale_preserve_similarity():
    x = torch.tensor([[1., 2.], [3., -1.], [2., 4.]], dtype=torch.float64)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    assert SimilarityPreservingLoss()(x, 3 * x @ rotation).item() < 1e-28


def test_similarity_gradients_reach_encoder_but_not_targets():
    x = torch.tensor([[1., 0., 0.], [0., 1., 0.]], requires_grad=True)
    encoder = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[1., 0.6, 0.], [0., 0.8, 1.]]))
    # Separate the target from the encoder inputs to isolate its gradient path.
    loss = SimilarityPreservingLoss()(x, encoder(x.detach()))
    loss.backward()
    assert x.grad is None
    assert encoder.weight.grad.abs().sum() > 0


def test_combined_objective_trains_decoder():
    x = torch.eye(3)
    encoder, decoder = nn.Linear(3, 2), nn.Linear(2, 3)
    z = encoder(x)
    MSESimilarityLoss()(decoder(z), x, latent=z).backward()
    assert encoder.weight.grad.abs().sum() > 0
    assert decoder.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_similarity_half_precision_and_autocast(dtype):
    x = torch.tensor([[1., 0.], [0., 1.]], dtype=dtype)
    z = torch.tensor([[1., 0.], [0.6, 0.8]], dtype=dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = SimilarityPreservingLoss()(x, z)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.36, abs=0.003)
    loss.backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1", None, 10**400])
def test_invalid_weight(value):
    with pytest.raises(MLXUserError, match="finite, nonnegative"):
        DEFAULT_LOSS_REGISTRY.resolve("mse-similarity")[0].build({"similarity_weight": value})


def test_unknown_option_and_missing_latent():
    with pytest.raises(MLXUserError, match="Unsupported mse-similarity"):
        DEFAULT_LOSS_REGISTRY.resolve("mse-similarity")[0].build({"weight": 1})
    with pytest.raises(ValueError, match="requires latent"):
        MSESimilarityLoss()(torch.eye(2), torch.eye(2))


@pytest.mark.parametrize("x,z", [
    (torch.ones(3), torch.ones(3)),
    (torch.ones(2, 3), torch.ones(3, 2)),
    (torch.ones(1, 3), torch.ones(1, 2)),
    (torch.ones(2, 0), torch.ones(2, 2)),
    (torch.ones(2, 3, dtype=torch.long), torch.ones(2, 2)),
])
def test_invalid_similarity_shapes_and_types(x, z):
    with pytest.raises(ValueError):
        SimilarityPreservingLoss()(x, z)


@pytest.mark.parametrize("count,batch_size", [(2, 2), (3, 2), (5, 2), (7, 3), (8, 3), (3, 8)])
def test_batch_sampler_preserves_every_sample(count, batch_size):
    sampler = MergeSingletonBatchSampler(SequentialSampler(range(count)), batch_size)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert [i for batch in batches for i in batch] == list(range(count))
    assert all(2 <= len(batch) <= batch_size + 1 for batch in batches)


def test_batch_sampler_seed_and_epoch_shuffling():
    def epochs():
        source = RandomSampler(range(13), generator=torch.Generator().manual_seed(17))
        sampler = MergeSingletonBatchSampler(source, 4)
        return list(sampler), list(sampler)
    first, second = epochs()
    assert (first, second) == epochs()
    assert first != second
    assert sorted(i for batch in second for i in batch) == list(range(13))


def request(tmp_path, **kwargs):
    values = dict(
        input_path=str(write_embeddings(tmp_path / "vectors.csv")),
        output_path=str(tmp_path / "run"), model="simple", loss="mse-similarity",
        hidden_dim=3, bottleneck_dim=2, epochs=1, batch_size=2,
        val_ratio=0.5, random_seed=17, plots=False,
    )
    return AutoencoderTrainRequest(**(values | kwargs))


@pytest.mark.parametrize("options", [{"batch_size": 1}, {"val_ratio": 0.1}, {"val_ratio": 0.9}])
def test_invalid_partitions_fail_before_output_creation(tmp_path, options):
    with pytest.raises(MLXUserError, match="at least"):
        TrainAutoencoder(request(tmp_path, **options)).execute()
    assert not (tmp_path / "run").exists()


def test_zero_weight_matches_mse_with_singletons(tmp_path):
    base = request(tmp_path, batch_size=1, val_ratio=0.1, loss="mse")
    paths = []
    for name, config in [("mse", None), ("mse-similarity", {"similarity_weight": 0})]:
        result = TrainAutoencoder(replace(
            base, loss=name, loss_config=config, output_path=str(tmp_path / name),
        )).execute()
        paths.append(result.output_dir)
    assert (paths[0] / "training.csv").read_text() == (paths[1] / "training.csv").read_text()
    first, second = [torch.load(path / "autoencoder.pth", weights_only=True) for path in paths]
    assert all(torch.equal(value, second["state_dict"][key]) for key, value in first["state_dict"].items())


def test_similarity_training_with_merged_batches_is_deterministic(tmp_path):
    base = request(tmp_path)
    outputs = []
    for name in ("first", "second"):
        result = TrainAutoencoder(replace(base, output_path=str(tmp_path / name))).execute()
        outputs.append((result.output_dir / "training.csv").read_text())
    assert outputs[0] == outputs[1]


def test_external_latent_loss_uses_single_encoder_pass(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from mlx.modes.autoencoder.models import AutoencoderRegistry, SimpleAutoencoder

    calls = []
    class Model(SimpleAutoencoder):
        def encode(self, x):
            calls.append("encode")
            return super().encode(x)
        def decode(self, z):
            calls.append("decode")
            return super().decode(z)

    class ModelDefinition:
        name, description = "external", "test"
        def build(self, config):
            return Model(4, 3, 2)

    class Loss(nn.Module):
        requires_latent = True
        minimum_batch_size = 2
        def forward(self, reconstruction, target, *, latent):
            calls.append(("loss", len(latent), torch.is_grad_enabled()))
            return F.mse_loss(reconstruction, target) + latent.square().mean()

    class Definition:
        name, description = "external", "test"
        def build(self, config):
            return Loss()

    module = ModuleType("external_similarity_test")
    module.Definition, module.ModelDefinition = Definition, ModelDefinition
    monkeypatch.setitem(sys.modules, module.__name__, module)
    TrainAutoencoder(
        request(tmp_path, model="external", loss="external"),
        model_registry=AutoencoderRegistry({"external": module.__name__ + ":ModelDefinition"}),
        loss_registry=ReconstructionLossRegistry({"external": module.__name__ + ":Definition"}),
    ).execute()
    # Each partition has three rows: its singleton merges into one batch of three.
    assert calls[-6:] == ["encode", "decode", ("loss", 3, True), "encode", "decode", ("loss", 3, False)]


def test_explicit_weight_from_json_is_saved_in_checkpoint_and_manifest(tmp_path):
    import json
    config = tmp_path / "loss.json"
    config.write_text(json.dumps({"similarity_weight": 0.25}))
    result = TrainAutoencoder(request(tmp_path, loss_config=str(config))).execute()
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    manifest = json.loads((result.output_dir / "autoencoder_manifest.json").read_text())
    assert checkpoint["loss_config"] == {"similarity_weight": 0.25}
    assert manifest["loss_config"] == {"similarity_weight": 0.25}
