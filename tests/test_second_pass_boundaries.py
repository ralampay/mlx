from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mlx.core.exceptions import MLXUserError
from mlx.modes.autoencoder.artifacts import build_checkpoint_model, load_checkpoint
from mlx.modes.autoencoder.model_registry import AutoencoderRegistry
from mlx.modes.text_embedding.artifacts import EmbeddingArtifactReader
from mlx.modes.text_embedding.commands import BenchmarkTextEmbeddingCommand
from mlx.modes.text_embedding.models import RetrievalQuery, RelevanceJudgment
from mlx.modes.text_embedding.requests import BenchmarkTextEmbeddingRequest
from mlx.modes.text_embedding.vector_store.protocol import VectorSearchResult


def benchmark_fixture(tmp_path, results):
    artifacts = {
        "root": tmp_path,
        "embedding_manifest": {
            "model": {"path": "fake.gguf", "sha256": "fake"},
            "embedding": {"dimensions": 2},
            "vector_store": {"provider": "fake"},
        },
        "dataset_manifest": {"name": "tiny", "corpus_documents": 2},
        "queries": ((RetrievalQuery("judged", "yes"), (1., 0.)),
                    (RetrievalQuery("unjudged", "no"), (1., 0.))),
        "qrels": (RelevanceJudgment("judged", "doc", 1),),
        "corpus_ids": frozenset({"doc", "other"}),
    }
    captured = {}
    store = SimpleNamespace(query=lambda *a, **k: results, close=lambda: captured.update(closed=True))
    command = BenchmarkTextEmbeddingCommand(
        BenchmarkTextEmbeddingRequest(input_path=".", output_path=str(tmp_path / "out"),
                                      top_k=2, k_values=(1, 2)),
        artifact_reader=SimpleNamespace(load=lambda _: artifacts),
        vector_store_factory=lambda *a, **k: store,
        artifact_writer=SimpleNamespace(write=lambda *a, **k: captured.update(k)),
    )
    return command, captured


def test_unjudged_queries_do_not_depress_retrieval_scores(tmp_path):
    command, captured = benchmark_fixture(tmp_path, (VectorSearchResult("doc", 1.0),))
    result = command.execute()
    assert result.metrics["recall@1"] == 1
    assert result.query_count == 1 and result.failures == 0
    assert captured["summary"]["excluded_queries"] == 1
    assert captured["manifest"]["excluded_queries"] == 1
    assert captured["closed"]


@pytest.mark.parametrize("results", [
    (VectorSearchResult("doc", 1), VectorSearchResult("doc", 0)),
    (VectorSearchResult("missing", 1),),
    (VectorSearchResult("doc", float("nan")),),
    (VectorSearchResult("doc", 0), VectorSearchResult("other", 1)),
    tuple(VectorSearchResult("doc", 1) for _ in range(3)),
])
def test_invalid_rankings_fail_and_close_store(tmp_path, results):
    command, captured = benchmark_fixture(tmp_path, results)
    with pytest.raises(MLXUserError):
        command.execute()
    assert captured["closed"]
    assert "summary" not in captured


@pytest.mark.parametrize("section", ["embedding", "model", "vector_store"])
def test_corrupt_manifest_sections_are_user_errors(section):
    dataset = dict(name="tiny", corpus_documents=1, queries=1, qrels=1)
    manifest = dict(model=dict(path="m", sha256="hash"),
                    embedding=dict(dimensions=2, normalized=False),
                    vector_store=dict(provider="chroma", similarity="cosine"))
    manifest[section] = None
    with pytest.raises(MLXUserError, match=section):
        EmbeddingArtifactReader._validate_manifests(dataset, manifest)


def checkpoint(reference):
    from mlx.modes.autoencoder.architectures.tiny import TinyAutoencoder
    model = TinyAutoencoder(3, 1)
    return dict(checkpoint_version=1, adapter_type="autoencoder", architecture="tiny",
                architecture_path=reference, model_config={}, input_dimensions=3,
                bottleneck_dimensions=1, expects_l2_normalized_input=False,
                state_dict=model.state_dict())


def test_external_checkpoint_code_is_checked_before_import():
    value = checkpoint("external:Definition")
    with patch("mlx.modes.autoencoder.artifacts.AutoencoderRegistry.resolve") as resolve:
        with pytest.raises(MLXUserError, match="trust-checkpoint-code"):
            build_checkpoint_model(value)
        resolve.assert_not_called()


@pytest.mark.parametrize("explicit_registry", [False, True])
def test_external_checkpoint_code_can_be_explicitly_trusted(explicit_registry):
    from mlx.modes.autoencoder.architectures.tiny import TinyAutoencoderDefinition
    registry = AutoencoderRegistry({"external": "external:Definition"}) if explicit_registry else AutoencoderRegistry()
    with patch.object(AutoencoderRegistry, "resolve", return_value=(TinyAutoencoderDefinition(), "external:Definition")):
        model = build_checkpoint_model(checkpoint("external:Definition"), registry=registry,
                                       trust_checkpoint_code=not explicit_registry)
    assert model(torch.zeros(2, 3)).shape == (2, 3)


def test_builtin_checkpoint_loads_with_restricted_pickle(tmp_path):
    path = tmp_path / "ae.pth"
    torch.save(checkpoint("mlx.modes.autoencoder.architectures.tiny:TinyAutoencoderDefinition"), path)
    _, value = load_checkpoint(path)
    assert build_checkpoint_model(value).encode(torch.zeros(2, 3)).shape == (2, 1)


def test_checkpoint_loading_never_falls_back_to_unsafe_pickle(tmp_path):
    import pickle
    path = tmp_path / "bad.pth"
    path.touch()
    with patch("torch.load", side_effect=pickle.UnpicklingError("blocked")) as load:
        with pytest.raises(MLXUserError):
            load_checkpoint(path)
    assert load.call_count == 1
    assert load.call_args.kwargs["weights_only"] is True


@pytest.mark.parametrize("loss", [torch.ones(2), torch.tensor(float("nan")), torch.tensor(1.)])
def test_scalar_loss_contract_rejects_invalid_training_outputs(loss):
    from mlx.core.losses import validate_scalar_loss
    with pytest.raises(MLXUserError):
        validate_scalar_loss(loss, training=True)


def test_libreyolo_rejects_unproven_adapter_kwargs():
    from mlx.modes.object_detection.libreyolo.model_factory import validate_incremental_adapter_support
    with pytest.raises(MLXUserError, match="does not expose"):
        validate_incremental_adapter_support(SimpleNamespace(train=lambda **kwargs: None))


def test_libreyolo_accepts_adapter_options_in_trainer_config():
    from dataclasses import make_dataclass
    from mlx.modes.object_detection.libreyolo.model_factory import validate_incremental_adapter_support
    config = make_dataclass("Config", [("incremental_adapter", bool),
                                     ("incremental_adapter_train_only", bool),
                                     ("incremental_adapter_type", str)])
    trainer = SimpleNamespace(_config_class=lambda: config)
    validate_incremental_adapter_support(SimpleNamespace(train=lambda **kw: None, _trainer_class=lambda: trainer))


def test_chroma_reload_rejects_wrong_similarity(tmp_path):
    from mlx.modes.text_embedding.vector_store.chroma import ChromaVectorStore
    collection = SimpleNamespace(metadata={"hnsw:space": "l2"})
    factory = lambda **kw: SimpleNamespace(get_collection=lambda **kw: collection)
    with pytest.raises(MLXUserError, match="cosine"):
        ChromaVectorStore(tmp_path, create=False, client_factory=factory)


@pytest.mark.parametrize("module", [
    "mlx.cli", "mlx.modes.segmentation.models.registry",
    "mlx.modes.saliency_mapping.models", "mlx.modes.image_classification.models.standard",
    "mlx.modes.object_detection.tracking.registry", "mlx.modes.text_embedding.embedding.registry",
])
def test_import_boundaries_in_fresh_process(module):
    import subprocess
    import sys
    source = '''
import importlib, importlib.abc, sys
blocked = {"cv2", "llama_cpp", "ultralytics", "libreyolo", "onnxruntime", "boto3", "chromadb", "matplotlib.pyplot"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + ".") for item in blocked):
            raise AssertionError("Eager dependency: " + fullname)
sys.meta_path.insert(0, Block())
importlib.import_module(sys.argv[1])
'''
    result = subprocess.run([sys.executable, "-c", source, module], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_core_and_saliency_dependency_direction():
    import ast
    root = Path(__file__).resolve().parents[1]
    for package, forbidden, permitted in [
        (root / "mlx/core", "mlx.modes", set()),
        (root / "mlx/modes/saliency_mapping", "mlx.modes.segmentation", {"compatibility.py"}),
    ]:
        for path in package.rglob("*.py"):
            if path.name in permitted:
                continue
            tree = ast.parse(path.read_text())
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
            imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
            assert not any(name.startswith(forbidden) for name in imports), path

@pytest.mark.parametrize("mode", ["autoencoder", "text-embedding"])
def test_checkpoint_trust_flag_reaches_typed_request(mode):
    from mlx.cli import build_parser, _build_config
    from mlx.modes.autoencoder.requests import AutoencoderEmbedRequest
    from mlx.modes.text_embedding.requests import EmbedTextRequest
    config = _build_config(build_parser().parse_args([
        "--mode", mode, "--action", "embed", "--trust-checkpoint-code",
        "--model", "model.gguf", "--input", "data", "--output", "out",
    ]))
    request_type = AutoencoderEmbedRequest if mode == "autoencoder" else EmbedTextRequest
    assert request_type.from_config(config).trust_checkpoint_code is True


def test_saliency_registry_is_local_immutable_and_owns_groups():
    from examples.extensions.pixel_model import build_saliency
    from mlx.modes.saliency_mapping.models import SaliencyModelRegistry, grouped_model_names
    original = SaliencyModelRegistry({})
    selected = original.register("pixel", build_saliency, small=True)
    assert not original.entries
    assert grouped_model_names("all", registry=selected) == ["pixel"]
    assert grouped_model_names("all-small", registry=selected) == ["pixel"]
    with pytest.raises(TypeError):
        selected.entries["other"] = build_saliency


@pytest.mark.parametrize("method", ["GRADCAM", "custom.cam:TinyCAM"])
def test_cam_preserves_custom_reference_and_normalizes_alias(method, monkeypatch):
    from mlx.modes.image_classification import cam
    model = SimpleNamespace(to=lambda _: model, eval=lambda: None)
    sentinel_registry = object()
    seen = {}
    def load(config, *, model_registry):
        assert model_registry is sentinel_registry
        return model, {"family": "standard"}
    def generate(*args, **kwargs):
        seen.update(kwargs)
        return []
    monkeypatch.setattr(cam, "load_checkpoint_bundle", load)
    monkeypatch.setattr(cam, "generate_standard_cams", generate)
    cam._generate_cams({"cam_method": method}, model_registry=sentinel_registry)
    assert seen["method"] == ("gradcam" if method == "GRADCAM" else method)
