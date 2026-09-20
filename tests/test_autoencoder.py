from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

from mlx.cli import _build_config, build_parser
from mlx.cli_routing import resolve_mode_descriptor
from mlx.core.exceptions import MLXUserError
from mlx.core.vector_transforms import VectorRepresentationTransformer
from mlx.modes.autoencoder.adapter import AutoencoderRepresentationTransformer
from mlx.modes.autoencoder.commands import (
    EmbedAutoencoder,
    ListAutoencoderLosses,
    ListAutoencoderModels,
    TrainAutoencoder,
)
from mlx.modes.autoencoder.data import EmbeddingCsvLoader
from mlx.modes.autoencoder.losses import (
    DEFAULT_LOSS_REGISTRY,
    ReconstructionLossRegistry,
)
from mlx.modes.autoencoder.models import (
    AutoencoderRegistry,
    SimpleAutoencoder,
)
from mlx.modes.autoencoder.requests import AutoencoderEmbedRequest, AutoencoderTrainRequest
from mlx.modes.text_embedding.commands import EmbedTextCommand
from mlx.modes.text_embedding.requests import EmbedTextRequest


def write_embeddings(path: Path, vectors=None) -> Path:
    values = vectors or [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("id", "text", "embedding"))
        writer.writeheader()
        for index, vector in enumerate(values):
            writer.writerow(
                {"id": f"v{index}", "text": f"row {index}", "embedding": json.dumps(vector)}
            )
    return path


def test_autoencoder_mode_and_cli_options_parse() -> None:
    descriptor = resolve_mode_descriptor("autoencoder")
    assert descriptor.actions == ("train", "embed", "ls-models", "ls-loss-functions", "ls-losses")
    config = _build_config(
        build_parser().parse_args(
            [
                "--mode", "autoencoder", "--action", "train", "--model", "simple",
                "--input", "vectors.csv", "--output", "run", "--input-dim", "384",
                "--hidden-dim", "256", "--bottleneck-dim", "128", "--loss", "mae",
                "--normalize-inputs", "--autoencoder-config", "model.json",
                "--loss-config", "loss.json",
            ]
        )
    )
    assert config["input_dim"] == 384
    assert config["bottleneck_dim"] == 128
    assert config["normalize_inputs"] is True
    assert config["loss"] == "mae"

    embed = _build_config(
        build_parser().parse_args(
            [
                "--mode", "autoencoder", "--action", "embed", "--model", "simple",
                "--model-path", "autoencoder.pth", "--input", "vectors.csv",
                "--output", "vectors-ae.csv",
            ]
        )
    )
    assert embed["model_path"] == "autoencoder.pth"
    assert embed["input_path"] == "vectors.csv"
    assert embed["output_path"] == "vectors-ae.csv"

    adapted_text = _build_config(
        build_parser().parse_args(
            [
                "--mode", "text-embedding", "--action", "embed", "--model", "model.gguf",
                "--adapter", "autoencoder.pth", "--input", "dataset", "--output", "artifacts",
            ]
        )
    )
    assert adapted_text["adapter"] == "autoencoder.pth"


def test_embedding_csv_loader_preserves_metadata_and_detects_manifest(tmp_path: Path) -> None:
    path = write_embeddings(tmp_path / "corpus_embeddings.csv")
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps({"embedding": {"dimensions": 4, "normalized": True}}), encoding="utf-8"
    )
    table = EmbeddingCsvLoader().load(path)
    assert table.dimensions == 4
    assert table.source_normalized is True
    assert table.rows[0]["text"] == "row 0"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("id,text\na,hello\n", "embedding.*column"),
        ('id,embedding\na,"[1, true]"\n', "non-numeric"),
        ('id,embedding\na,"[1, 2]"\nb,"[1]"\n', "dimension changed"),
        ('id,embedding\na,not-json\n', "Malformed embedding JSON"),
        ("id,embedding\n", "no vector rows"),
    ],
)
def test_embedding_csv_loader_rejects_invalid_data(tmp_path, content, message) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(MLXUserError, match=message):
        EmbeddingCsvLoader().load(path)


def test_simple_autoencoder_contract_and_shapes() -> None:
    model = SimpleAutoencoder(4, 3, 2)
    values = torch.randn(5, 4)
    encoded = model.encode(values)
    assert encoded.shape == (5, 2)
    assert model.decode(encoded).shape == (5, 4)
    assert model(values).shape == values.shape
    with pytest.raises(ValueError, match="smaller"):
        SimpleAutoencoder(4, 4, 4)


def test_builtin_losses_are_correct_and_listed() -> None:
    prediction = torch.tensor([[2.0, 0.0]])
    target = torch.tensor([[0.0, 0.0]])
    mse = DEFAULT_LOSS_REGISTRY.resolve("mse")[0].build({})
    mae = DEFAULT_LOSS_REGISTRY.resolve("mae")[0].build({})
    smooth = DEFAULT_LOSS_REGISTRY.resolve("smooth-l1")[0].build({"beta": 1.0})
    assert mse(prediction, target).item() == pytest.approx(2.0)
    assert mae(prediction, target).item() == pytest.approx(1.0)
    assert smooth(prediction, target).item() == pytest.approx(0.75)
    assert DEFAULT_LOSS_REGISTRY.names() == ("mae", "mse", "smooth-l1")
    assert [item["name"] for item in ListAutoencoderLosses().execute()] == [
        "mae", "mse", "smooth-l1"
    ]
    assert [item["name"] for item in ListAutoencoderModels().execute()] == ["simple", "tiny"]


def test_external_model_and_loss_definition_import_paths(monkeypatch) -> None:
    module = ModuleType("fake_autoencoder_extension")

    class CustomModelDefinition:
        name = "custom"
        description = "test model"

        def build(self, config):
            return SimpleAutoencoder(
                int(config["input_dimensions"]),
                int(config["hidden_dimensions"]),
                int(config["bottleneck_dimensions"]),
            )

    class CustomLossDefinition:
        name = "custom-loss"
        description = "test loss"

        def build(self, config):
            return nn.MSELoss(reduction=config.get("reduction", "mean"))

    module.CustomModelDefinition = CustomModelDefinition
    module.CustomLossDefinition = CustomLossDefinition
    monkeypatch.setitem(sys.modules, module.__name__, module)
    models = AutoencoderRegistry({}).register(
        "custom", "fake_autoencoder_extension:CustomModelDefinition"
    )
    losses = ReconstructionLossRegistry({}).register(
        "custom-loss", "fake_autoencoder_extension:CustomLossDefinition"
    )
    assert models.resolve("custom")[0].name == "custom"
    assert losses.resolve("custom-loss")[0].build({})(prediction := torch.ones(1), prediction).item() == 0


def test_train_checkpoint_and_standalone_embed_workflow(tmp_path: Path) -> None:
    source = write_embeddings(tmp_path / "vectors.csv")
    output = tmp_path / "training"
    result = TrainAutoencoder(
        AutoencoderTrainRequest(
            model="simple",
            input_path=str(source),
            output_path=str(output),
            input_dim=4,
            hidden_dim=3,
            bottleneck_dim=2,
            epochs=2,
            batch_size=2,
            val_ratio=0.34,
            random_seed=7,
            plots=False,
        )
    ).execute()
    assert result.checkpoint_path.is_file()
    assert {path.name for path in output.iterdir()} == {
        "autoencoder.pth", "training.csv", "autoencoder_manifest.json", "run_metadata.json"
    }
    transformer = AutoencoderRepresentationTransformer(result.checkpoint_path)
    assert isinstance(transformer, VectorRepresentationTransformer)
    assert transformer.input_dimensions == 4
    assert transformer.output_dimensions == 2
    assert len(transformer.transform([[1.0, 0.0, 0.0, 0.0]])[0]) == 2

    transformed_path = tmp_path / "vectors-ae.csv"
    embedded = EmbedAutoencoder(
        AutoencoderEmbedRequest(
            model="simple",
            model_path=str(result.checkpoint_path),
            input_path=str(source),
            output_path=str(transformed_path),
            batch_size=2,
        )
    ).execute()
    assert embedded.rows == 6
    assert embedded.output_dimensions == 2
    transformed = EmbeddingCsvLoader().load(transformed_path)
    assert transformed.dimensions == 2
    assert transformed.rows[0]["text"] == "row 0"
    assert transformed_path.with_suffix(".manifest.json").is_file()


def test_training_normalization_contract_and_dimension_failures(tmp_path: Path) -> None:
    source = write_embeddings(tmp_path / "corpus_embeddings.csv")
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps({"embedding": {"dimensions": 4, "normalized": True}}), encoding="utf-8"
    )
    request = AutoencoderTrainRequest(
        input_path=str(source), output_path=str(tmp_path / "run"), hidden_dim=3,
        bottleneck_dim=2, epochs=1, batch_size=2, val_ratio=0.3,
        normalize_inputs=False, plots=False,
    )
    with pytest.raises(MLXUserError, match="conflicts"):
        TrainAutoencoder(request).execute()

    mismatch = AutoencoderTrainRequest(
        input_path=str(source), output_path=str(tmp_path / "other"), input_dim=3,
        hidden_dim=3, bottleneck_dim=2, epochs=1, plots=False,
    )
    with pytest.raises(MLXUserError, match="CSV contains 4"):
        TrainAutoencoder(mismatch).execute()


def test_training_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    source = write_embeddings(tmp_path / "vectors.csv")

    def run(name: str):
        result = TrainAutoencoder(
            AutoencoderTrainRequest(
                input_path=str(source), output_path=str(tmp_path / name),
                hidden_dim=3, bottleneck_dim=2, epochs=1, batch_size=2,
                val_ratio=0.34, random_seed=19, plots=False,
            )
        ).execute()
        checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
        return checkpoint, (tmp_path / name / "training.csv").read_text(encoding="utf-8")

    first, first_history = run("first")
    second, second_history = run("second")
    assert first_history == second_history
    assert all(
        torch.equal(first["state_dict"][name], second["state_dict"][name])
        for name in first["state_dict"]
    )


class FakeEmbeddingProvider:
    dimensions = 2

    def embed(self, texts):
        return [[3.0, 4.0] for _ in texts]


class FakeTransformer:
    input_dimensions = 2
    output_dimensions = 1
    provenance = {"type": "autoencoder", "sha256": "abc", "architecture": "fake"}

    def transform(self, vectors):
        return [[vector[0] + vector[1]] for vector in vectors]


class FakeStore:
    def __init__(self):
        self.records = []

    def add(self, records):
        self.records.extend(records)

    def query(self, vector, *, k):
        return ()

    def close(self):
        return None


def write_beir(root: Path) -> Path:
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "Title", "text": "body"}) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "query"}) + "\n", encoding="utf-8"
    )
    (root / "qrels" / "test.tsv").write_text("q1\td1\t1\n", encoding="utf-8")
    return root


def test_text_embedding_adapter_transforms_before_export_and_indexing(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    dataset = write_beir(tmp_path / "dataset")
    store = FakeStore()
    output = tmp_path / "artifacts"
    result = EmbedTextCommand(
        EmbedTextRequest(
            model=str(model), input_path=str(dataset), output_path=str(output),
            representation="autoencoder-1", normalize_embeddings=True,
        ),
        provider_factory=lambda _path: FakeEmbeddingProvider(),
        vector_store_factory=lambda *_args, **_kwargs: store,
        transformer=FakeTransformer(),
    ).execute()
    assert result.dimensions == 1
    assert store.records[0].vector == pytest.approx((1.0,))
    manifest = json.loads((output / "embedding_manifest.json").read_text())
    assert manifest["embedding"]["source_dimensions"] == 2
    assert manifest["embedding_configuration"]["embedding_dimension"] == 2
    assert manifest["embedding_configuration"]["pooling_effective"] == "unknown"
    assert manifest["embedding"]["dimensions"] == 1
    assert manifest["adapter"]["architecture"] == "fake"
    with (output / "query_embeddings.csv").open(newline="") as source:
        row = next(csv.DictReader(source))
    assert json.loads(row["embedding"]) == [1.0]


@pytest.mark.parametrize("registry", [AutoencoderRegistry({}), ReconstructionLossRegistry({})])
@pytest.mark.parametrize("invalid", ["not-class", "constructor", "protocol"])
def test_definition_factories_preserve_contextual_failures(monkeypatch, registry, invalid):
    module = ModuleType("test_invalid_definition")

    class BrokenDefinition:
        def __init__(self):
            raise ValueError("invalid settings")

    module.Definition = {
        "not-class": lambda: None,
        "constructor": BrokenDefinition,
        "protocol": type("MissingProtocol", (), {}),
    }[invalid]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    expected = {
        "not-class": "does not reference a class",
        "constructor": "Unable to construct",
        "protocol": "must provide name, description",
    }[invalid]
    with pytest.raises(MLXUserError, match=expected) as error:
        registry.resolve("test_invalid_definition:Definition")
    if invalid == "constructor":
        assert isinstance(error.value.__cause__, ValueError)
