from __future__ import annotations

from dataclasses import asdict, dataclass

from mlx.modes.image_recognition_oc.backbones import image_backbone_names


@dataclass(frozen=True)
class ImageOneClassVariant:
    model_name: str
    backbone_name: str
    variant_id: str
    drax_fusion_mode: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def image_one_class_svdd_variants() -> tuple[ImageOneClassVariant, ...]:
    variants: list[ImageOneClassVariant] = []
    for backbone_name in image_backbone_names():
        fusion_modes = ("average", "sknet") if backbone_name.startswith("drax") else (None,)
        for fusion_mode in fusion_modes:
            suffix = f"-{fusion_mode}" if fusion_mode else ""
            variants.append(
                ImageOneClassVariant(
                    model_name="deep-svdd",
                    backbone_name=backbone_name,
                    variant_id=f"{backbone_name}{suffix}-deep-svdd",
                    drax_fusion_mode=fusion_mode,
                )
            )
    return tuple(variants)


__all__ = ["ImageOneClassVariant", "image_one_class_svdd_variants"]
