"""Route detector checkpoints at the MLX/LibreYOLO inference boundary."""

from collections.abc import Mapping
from pathlib import Path
from pickle import UnpicklingError

from mlx.core.exceptions import MLXUserError


class LoadLibreYOLODetector:
    def __init__(self, model_path, *, device="cpu", adapter_path=None, registry=None):
        self.model_path = Path(model_path)
        self.device = device
        self.adapter_path = adapter_path
        self.registry = registry

    def execute(self):
        from .dependencies import quiet_unused_attention_imports

        with quiet_unused_attention_imports():
            return self._load_checkpoint()

    def _load_checkpoint(self):
        if self.model_path.suffix.lower() == ".pt":
            import torch

            try:
                payload = torch.load(self.model_path, map_location="cpu", weights_only=True)
            except (OSError, ValueError, RuntimeError, EOFError, UnpicklingError):
                # LibreYOLO owns legacy conversion and its checkpoint-read errors.
                # Delegate the original file rather than relaxing safe loading.
                return self._load_provider_checkpoint()
            if isinstance(payload, Mapping):
                checkpoint_format = payload.get("format")
                if (isinstance(checkpoint_format, str)
                        and checkpoint_format.startswith("mlx-yolox-adapter-detector-")):
                    from .adapter_checkpoint import FORMAT, LoadAdapterDetector

                    if checkpoint_format != FORMAT:
                        raise MLXUserError(
                            f"Unsupported adapter detector format {checkpoint_format!r}; expected {FORMAT}."
                        )
                    if self.adapter_path:
                        raise MLXUserError(
                            "--model-path already contains an adapter detector. "
                            "Omit --adapter, or supply its original foundation checkpoint instead."
                        )
                    return LoadAdapterDetector(
                        self.model_path, device=self.device, registry=self.registry, payload=payload,
                    ).execute()
            del payload
        return self._load_provider_checkpoint()

    def _load_provider_checkpoint(self):
        if self.adapter_path:
            from .adapter_loading import LoadAdaptedYOLOX

            return LoadAdaptedYOLOX(
                self.model_path, self.adapter_path, device=self.device,
                **({"registry": self.registry} if self.registry is not None else {}),
            ).execute()
        from libreyolo import LibreYOLO

        return LibreYOLO(str(self.model_path), device=self.device, task="detect")
