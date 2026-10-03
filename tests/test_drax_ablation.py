import sys
from types import SimpleNamespace

import pytest
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.libreyolo.model_factory import build_scratch_model
from mlx.modes.object_detection.libreyolo.utils import resolve_model_spec
from mlx.modes.object_detection.libreyolo.training import TrainLibreYOLOObjectDetection
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest


@pytest.mark.parametrize(
    "preset", ["refine-p3p4", "spp-p5", "balanced-drax", "pyramid-drax"]
)
def test_alias_injects_variant(monkeypatch, preset):
    def constructor(architecture_variant=None, **kwargs):
        return dict(kwargs, architecture_variant=architecture_variant)

    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLOXDraxMobileNetV3Large=constructor),
    )
    size = "m" if preset == "pyramid-drax" else "l"
    spec = resolve_model_spec(
        f"yolox-drax-mobilenet-v3-large-{size}-" + preset
    )
    assert build_scratch_model(spec, device="cpu")["architecture_variant"] == preset


def test_variant_rejects_provider_that_would_ignore_option(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLOXDraxMobileNetV3Large=lambda **kw: None),
    )
    with pytest.raises(MLXUserError, match="does not support backbone variants"):
        build_scratch_model(
            resolve_model_spec("yolox-drax-mobilenet-v3-large-l-spp-p5"), device="cpu"
        )


def test_optional_training_controls():
    request = TrainObjectDetectionRequest(
        workers=2, eval_interval=1, no_aug_epochs=15, patience=0, pretrained=False
    )
    command = TrainLibreYOLOObjectDetection(request)
    kwargs = command._build_train_kwargs(
        data="data.yaml",
        project_dir="out",
        run_name="test",
        resume=False,
        allow_pretrained=True,
    )
    assert {
        k: kwargs[k] for k in ("workers", "eval_interval", "no_aug_epochs", "patience")
    } == dict(workers=2, eval_interval=1, no_aug_epochs=15, patience=0)
    assert kwargs["pretrained"] is False
    with pytest.raises(MLXUserError):
        TrainLibreYOLOObjectDetection({"workers": -1})._build_train_kwargs(
            data="data.yaml",
            project_dir="out",
            run_name="test",
            resume=False,
            allow_pretrained=True,
        )
