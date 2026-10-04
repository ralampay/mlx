"""CLI construction boundary; workflows also expose direct Python execute APIs."""

from pathlib import Path

from mlx.core.exceptions import MLXUserError
from .data import read_json
from .report import GenerateTransferReport
from .gallery import GenerateTransferGallery


def run_transfer_action(config):
    if not config.get("output_path"):
        raise MLXUserError(
            "Zero-shot studies require --output pointing to a dedicated experiment directory."
        )
    output = Path(config["output_path"]).expanduser()
    if config["action"] == "adapter-zero-shot-report":
        return GenerateTransferReport(output, GenerateTransferGallery(output)).execute()
    if not config.get("study_config"):
        raise MLXUserError(
            "Provide --study-config with foundation, studies, seeds, datasets and settings."
        )
    from ..libreyolo.zero_shot_backend import LibreYOLOTransferEvaluator
    from ..libreyolo.adapter_backend import VerifyFoundationCheckpoint
    from .workflow import RunTransferStudy

    spec = read_json(Path(config["study_config"]).expanduser())
    required = {"foundation", "studies", "seeds", "datasets", "settings"}
    if not required <= spec.keys():
        raise MLXUserError(f"Study config missing keys: {sorted(required-spec.keys())}")
    if spec["settings"] != {
        "image_size": 640,
        "precision": "float32",
        "confidence": 0.001,
        "nms_iou": 0.6,
        "max_det": 300,
        "coco_max_det": 100,
        "fixed_confidence": 0.25,
        "fixed_iou": 0.5,
    }:
        raise MLXUserError(
            "This protocol requires the documented fixed zero-shot evaluation settings."
        )
    evaluator = LibreYOLOTransferEvaluator(
        config.get("device", "cpu"), config.get("workers", 4)
    )
    result = RunTransferStudy(
        spec,
        output,
        evaluator,
        VerifyFoundationCheckpoint("yolox-l", Path(spec["foundation"]).expanduser()),
        config.get("study_phase", "all"),
    ).execute()
    if config.get("study_phase", "all") == "all":
        return GenerateTransferReport(output, GenerateTransferGallery(output)).execute()
    return result
