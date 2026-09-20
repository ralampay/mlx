"""Public package exports must not pull workflows into lower-level imports."""
from importlib import import_module
import subprocess
import sys

import pytest


PACKAGES = (
    "autoencoder", "text_embedding", "image_recognition_oc", "video_anomaly_detection",
)


def run_isolated(source, *arguments):
    result = subprocess.run(
        [sys.executable, "-c", source, *arguments], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", PACKAGES)
def test_package_and_requests_do_not_load_workflows_or_frameworks(mode):
    run_isolated('''
import importlib, importlib.abc, sys
prefix = "mlx.modes." + sys.argv[1]
blocked = {"torch", "cv2", "matplotlib", "rich", "llama_cpp", "chromadb",
           prefix + ".commands", prefix + ".runner", prefix + ".presentation"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + ".") for item in blocked):
            raise AssertionError("Unexpected dependency: " + fullname)
sys.meta_path.insert(0, Block())
package = importlib.import_module(prefix)
assert set(package.__all__).issubset(dir(package))
requests = importlib.import_module(prefix + ".requests")
for name in package.__all__:
    if name.endswith("Request"):
        assert getattr(package, name) is getattr(requests, name)
''', mode)


@pytest.mark.parametrize("module", [
    "autoencoder.model_registry",
    "text_embedding.metric_registry",
    "image_recognition_oc.algorithms",
    "video_anomaly_detection.models.temporal",
])
def test_extension_imports_do_not_load_workflows(module):
    run_isolated('''
import importlib, importlib.abc, sys
blocked = {"commands", "runner", "presentation", "training", "evaluation", "inference"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("mlx.modes.") and fullname.rsplit(".", 1)[-1] in blocked:
            raise AssertionError("Unexpected workflow dependency: " + fullname)
sys.meta_path.insert(0, Block())
importlib.import_module("mlx.modes." + sys.argv[1])
''', module)


@pytest.mark.parametrize("mode, expected", [
    ("autoencoder", {
        "AutoencoderEmbedRequest": "requests", "AutoencoderTrainRequest": "requests",
        "EmbedAutoencoder": "commands", "TrainAutoencoder": "commands",
    }),
    ("text_embedding", {
        "BenchmarkTextEmbeddingCommand": "commands", "BenchmarkTextEmbeddingRequest": "requests",
        "EmbedTextCommand": "commands", "EmbedTextRequest": "requests",
    }),
    ("image_recognition_oc", {
        "BenchmarkImageOneClass": "commands", "ImageOneClassInferenceResult": "commands",
        "InferImageOneClass": "commands", "ListImageOneClassModels": "commands",
        "TrainImageOneClassModel": "commands", "run_image_recognition_oc": "runner",
    }),
    ("video_anomaly_detection", {
        "BenchmarkVideoAnomalyModel": "commands", "BenchmarkVideoAnomalyRequest": "requests",
        "InferVideoAnomaly": "commands", "InferVideoAnomalyRequest": "requests",
        "VideoAnomalyInferenceResult": "inference", "ListVideoAnomalyModels": "commands",
        "ListVideoAnomalyModelsRequest": "requests", "TrainVideoAnomalyModel": "commands",
        "TrainVideoAnomalyRequest": "requests", "VideoAnomalyModel": "models",
        "VideoAnomaly3DModel": "models",
    }),
])
def test_public_exports_preserve_identity_and_star_imports(mode, expected):
    prefix = f"mlx.modes.{mode}"
    package = import_module(prefix)
    namespace = {}
    exec(f"from {prefix} import *", namespace)
    assert package.__all__ == list(expected)
    for name, module in expected.items():
        original = getattr(import_module(f"{prefix}.{module}"), name)
        assert getattr(package, name) is original
        assert namespace[name] is original
    with pytest.raises(AttributeError):
        getattr(package, "unknown_public_export")
