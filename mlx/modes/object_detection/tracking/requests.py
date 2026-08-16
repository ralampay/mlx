from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrackingRequest:
    provider: str = "ultralytics"
    model: str | None = None
    model_path: str | None = None
    device: str = "cpu"
    height: int = 640
    width: int = 640
    confidence: float = 0.25
    file_path: str | None = None
    output_path: str | None = None
    tracker: str = "bytetrack"
    tracker_config: str | None = None
    ground_truth: str | None = None
    benchmark_iou: float = 0.5
    track_class_ids: tuple[int, ...] = ()
    overwrite: bool = False
    display: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TrackingRequest:
        field_names = set(cls.__dataclass_fields__) - {"extras", "track_class_ids"}
        values = {name: config[name] for name in field_names if name in config}
        raw_class_ids = config.get("track_class_ids") or ()
        values["track_class_ids"] = tuple(int(value) for value in raw_class_ids)
        values["extras"] = {
            name: value
            for name, value in config.items()
            if name not in field_names and name != "track_class_ids"
        }
        return cls(**values)

    def detector_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_path": self.model_path,
            "device": self.device,
            "height": self.height,
            "width": self.width,
            "confidence": self.confidence,
        }
