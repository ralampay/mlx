from __future__ import annotations

from dataclasses import asdict, dataclass

from mlx.modes.video_anomaly_detection.models.backbone3d import DEFAULT_BACKBONE_3D_REGISTRY


@dataclass(frozen=True)
class VideoAnomalyModelVariant:
    model_name: str
    variant_id: str
    drax_fusion_mode: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def video_anomaly_model_variants(registry=None) -> tuple[VideoAnomalyModelVariant, ...]:
    variants: list[VideoAnomalyModelVariant] = []
    for model_name in sorted((registry or DEFAULT_BACKBONE_3D_REGISTRY).entries):
        fusion_modes = ("average", "sknet") if model_name.startswith("drax") else (None,)
        for fusion_mode in fusion_modes:
            variant_id = (
                f"{model_name}-{fusion_mode}" if fusion_mode is not None else model_name
            )
            variants.append(
                VideoAnomalyModelVariant(
                    model_name=model_name,
                    variant_id=variant_id,
                    drax_fusion_mode=fusion_mode,
                )
            )
    return tuple(variants)


__all__ = ["VideoAnomalyModelVariant", "video_anomaly_model_variants"]
