"""Portable contracts for CLI composition and replaceable workflow boundaries."""
from importlib import import_module
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mlx.core.exceptions import MLXUserError


@pytest.mark.parametrize("mode, default", [
    ("image_classification", (224, 224)), ("segmentation", (256, 256)),
])
@pytest.mark.parametrize("values, expected", [
    ({}, None),
    ({"input_size": (40, 24)}, (40, 24)),
    ({"width": 256, "height": 256}, (256, 256)),
    ({"input_size": (40, 24), "width": 32}, (32, 24)),
])
def test_runners_normalize_python_dimensions_without_parser(mode, default, values, expected, monkeypatch):
    runner = import_module(f"mlx.modes.{mode}.runner")
    monkeypatch.setitem(runner.ACTION_HANDLERS, "test", lambda config: config)
    config = {"action": "test", "output_format": "json", **values}
    original = dict(config)
    result = getattr(runner, f"run_{mode}")(config)
    assert result["input_size"] == (expected or default)
    assert (result["width"], result["height"]) == result["input_size"]
    assert config == original


def test_classification_explicit_cli_default_dimensions_are_preserved(monkeypatch):
    from mlx.cli import build_parser, _build_config, _explicit_option_destinations
    from mlx.modes.image_classification import runner
    parser = build_parser()
    args = ["--mode", "image-classification", "--width", "256", "--height", "256", "--format", "json"]
    config = _build_config(parser.parse_args(args))
    config["_explicit_options"] = _explicit_option_destinations(parser, args)
    monkeypatch.setitem(runner.ACTION_HANDLERS, "test", lambda config: config)
    assert runner.run_image_classification(config)["input_size"] == (256, 256)


@pytest.mark.parametrize("size", [(0, 4), (3.5, 4), (True, 4), (4,), None])
def test_invalid_dimensions_are_user_errors(size):
    from mlx.core.configuration import normalize_image_dimensions
    if size is None:
        assert normalize_image_dimensions({"input_size": None}, default=(8, 8))["input_size"] == (8, 8)
    else:
        with pytest.raises(MLXUserError, match="positive integer"):
            normalize_image_dimensions({"input_size": size}, default=(8, 8))


def isolated(source):
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parser_build_does_not_import_workflows_or_frameworks():
    isolated('''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'cv2', 'matplotlib', 'ultralytics', 'libreyolo'}:
            raise AssertionError(fullname)
        if fullname.startswith('mlx.modes.') and fullname.rsplit('.', 1)[-1] in {'commands', 'runner', 'train', 'evaluation'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from mlx.cli import build_parser
assert build_parser().parse_args(['--mode', 'track']).mode == 'track'
''')


@pytest.mark.parametrize("mode", ["image_classification", "segmentation"])
def test_name_listing_does_not_load_unrelated_execution_dependencies(mode):
    isolated('''
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cv2' or fullname.startswith('matplotlib'):
            raise AssertionError(fullname)
        if fullname.startswith('mlx.modes.') and fullname.rsplit('.', 1)[-1] in {'train', 'evaluation', 'inference', 'cam'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
mode = ''' + repr(mode) + '''
runner = importlib.import_module('mlx.modes.' + mode + '.runner')
assert getattr(runner, 'run_' + mode)({'action': 'ls-models', 'names_only': True, 'output_format': 'json'})
''')


def test_help_uses_registered_option_metadata(monkeypatch):
    from mlx import cli
    from mlx.core.cli_options import CliOption
    from mlx import cli_options
    from rich.console import Console
    from io import StringIO
    monkeypatch.setattr(cli_options, "OPTIONS", (*cli_options.OPTIONS,
        CliOption(("--synthetic-option",), {"default": "contract-default", "help": "contract description"})))
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=200))
    assert cli.build_parser().parse_args([]).synthetic_option == "contract-default"
    cli._render_help()
    assert all(value in output.getvalue() for value in ("--synthetic-option", "contract-default", "contract description"))


def test_detection_registry_isolation_reaches_runner(monkeypatch):
    from mlx.modes.object_detection import providers, runner
    from types import ModuleType
    module = ModuleType("synthetic_providers")
    module.first = lambda: SimpleNamespace(benchmark=lambda request, reporter: "first")
    module.second = lambda: SimpleNamespace(benchmark=lambda request, reporter: "second")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = providers.BUILTIN_PROVIDER_REGISTRY
    first = original.register("custom", "synthetic_providers:first")
    second = original.register("custom", "synthetic_providers:second")
    monkeypatch.setattr(providers, "_COMPAT_PROVIDER_ENTRIES", dict(providers._COMPAT_PROVIDER_ENTRIES))
    monkeypatch.setattr(providers, "DEFAULT_PROVIDER_REGISTRY", original)
    providers.register_provider("custom", "synthetic_providers:second")
    config = {"action": "benchmark", "provider": "custom", "output_format": "json"}
    assert runner.run_object_detection(config, provider_registry=first) == "first"
    assert runner.run_object_detection(config, provider_registry=second) == "second"
    assert "custom" not in original.entries
    assert providers.get_provider("custom").benchmark(None, None) == "second"
    with pytest.raises(TypeError):
        first.entries["custom"] = "synthetic_providers:second"


def test_classification_training_uses_injected_data_and_storage(tmp_path):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    from mlx.modes.image_classification.train import TrainImageClassificationModel
    from mlx.modes.image_classification.requests import TrainImageClassificationRequest
    from mlx.modes.image_classification.models.standard import StandardModelRegistry
    calls = []
    dataset = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
    def load(*args, **kwargs):
        calls.append("data")
        return dataset, dataset, ["a", "b"]
    artifacts = SimpleNamespace(
        restore=lambda *a, **k: (0, float("inf"), []),
        write_history=lambda path, history: calls.append(("history", len(history))),
        save_model=lambda *a, **k: calls.append("model"),
        save_training=lambda *a, **k: calls.append(("training", k["completed_epoch"])),
    )
    registry = StandardModelRegistry({}).register("tiny", lambda **kwargs: nn.Linear(2, 2))
    output = tmp_path / "no-files"
    TrainImageClassificationModel(
        TrainImageClassificationRequest(model="tiny", epochs=1, workers=0, batch_size=2,
                                        output_path=str(output)),
        model_registry=registry, standard_dataset_loader=load, artifacts=artifacts,
    ).execute()
    assert calls == ["data", ("history", 1), "model", ("training", 1)]
    assert not output.exists()


@pytest.mark.parametrize("family", ["standard", "one-shot"])
def test_benchmark_uses_injected_data_and_writer_for_every_output(tmp_path, family):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    from mlx.modes.image_classification.evaluation import BenchmarkImageClassification
    from mlx.modes.image_classification.evaluation_records import OneShotPairRecord
    from mlx.modes.image_classification.requests import BenchmarkImageClassificationRequest
    paths = {"a": [Path("a1"), Path("a2")], "b": [Path("b1"), Path("b2")]}
    pairs = [OneShotPairRecord(paths["a"][0], "a", paths[label][1], label, int(label == "a")) for label in paths]
    data = SimpleNamespace(
        resolve_root=lambda _: Path("unused"),
        load_standard=lambda *a, **k: TensorDataset(torch.eye(2), torch.tensor([0, 1])),
        build_pairs=lambda *a, **k: pairs,
        index_images=lambda *a, **k: paths,
        load_image=lambda path, **kw: torch.tensor([float(str(path).startswith("b"))]),
    )
    class PairModel(nn.Module):
        def forward(self, first, second):
            return 1 - (first - second).abs()
    model = nn.Identity() if family == "standard" else PairModel()
    metadata = dict(family=family, classes=["a", "b"], input_size=(1, 1), colored=False)
    writes = []
    writer = SimpleNamespace(
        write_metrics=lambda path, **kw: writes.append(("metrics", path, kw)),
        write_one_shot=lambda path, **kw: writes.append(("pairs", path, kw)),
        write_n_way=lambda path, rows: writes.append(("n-way", path, rows)),
    )
    output = tmp_path / "no-files"
    result = BenchmarkImageClassification(
        BenchmarkImageClassificationRequest(workers=0, output_path=str(output)),
        checkpoint_loader=lambda config: (model, metadata), dataset_loader=data, artifact_writer=writer,
    ).execute()
    assert result["accuracy"] == 1
    assert [row[0] for row in writes] == (["metrics"] if family == "standard" else ["metrics", "n-way", "metrics", "pairs"])
    assert not output.exists()


def test_default_benchmark_writer_preserves_artifacts(tmp_path):
    import numpy as np
    from mlx.modes.image_classification.benchmark_artifacts import ClassificationBenchmarkArtifacts
    output = tmp_path / "benchmark"
    ClassificationBenchmarkArtifacts().write_metrics(
        output, results={"accuracy": 1.}, targets=[0, 1], preds=[0, 1],
        probabilities=np.eye(2), class_names=["a", "b"],
    )
    assert {path.name for path in output.iterdir()} == {
        "metrics.csv", "confusion_matrix.csv", "confusion_matrix.png", "roc_curve.png",
    }
    assert "accuracy,1.000000" in (output / "metrics.csv").read_text()


def test_one_shot_training_propagates_loader_storage_and_optimizer(tmp_path, monkeypatch):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    from mlx.modes.image_classification import train
    from mlx.modes.image_classification.requests import TrainImageClassificationRequest

    class PairModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)

        def forward(self, first, second):
            return self.linear(torch.cat((first, second), dim=1)).sigmoid()

    monkeypatch.setattr(train, "build_image_classification_model", lambda *a, **kw: PairModel())
    dataset = TensorDataset(torch.eye(2), torch.eye(2), torch.tensor([0., 1.]))
    calls = []

    def optimizer(parameters, **kwargs):
        calls.append("optimizer")
        return torch.optim.SGD(parameters, **kwargs)

    def loader(*args, **kwargs):
        calls.append("dataset")
        return dataset, dataset

    artifacts = SimpleNamespace(
        restore=lambda *a, **k: (0, float("inf"), []),
        write_history=lambda *a: calls.append("history"),
        save_model=lambda *a, **k: calls.append("model"),
        save_training=lambda *a, **k: calls.append("checkpoint"),
    )
    train.TrainImageClassificationModel(
        TrainImageClassificationRequest(model="siamese-le-net", epochs=1, workers=0,
                                        batch_size=2, output_path=str(tmp_path / "unused")),
        one_shot_dataset_loader=loader, artifacts=artifacts, optimizer_factory=optimizer,
    ).execute()
    assert calls == ["optimizer", "dataset", "history", "model", "checkpoint"]
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("config", [
    {"platform": "aws", "action": "train"},
    {"action": "adapter-experiment"},
])
def test_provider_override_rejects_unrelated_composition_boundaries(config):
    from mlx.modes.object_detection.runner import run_object_detection
    from mlx.modes.object_detection.providers import ProviderRegistry
    with pytest.raises(MLXUserError, match="ordinary local detection"):
        run_object_detection(config, provider_registry=ProviderRegistry({}))


def test_benchmark_dependency_failure_prevents_output(tmp_path):
    from mlx.modes.image_classification.evaluation import BenchmarkImageClassification
    from mlx.modes.image_classification.requests import BenchmarkImageClassificationRequest
    writes = []

    def missing_checkpoint(config):
        raise MLXUserError("Synthetic checkpoint is unavailable")

    with pytest.raises(MLXUserError, match="Synthetic checkpoint"):
        BenchmarkImageClassification(
            BenchmarkImageClassificationRequest(output_path=str(tmp_path / "unused")),
            checkpoint_loader=missing_checkpoint,
            artifact_writer=SimpleNamespace(write_metrics=lambda *a, **k: writes.append(k)),
        ).execute()
    assert not writes
    assert not (tmp_path / "unused").exists()
