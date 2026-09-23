from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.artifacts import (
    detect_existing_training_artifacts,
    find_existing_checkpoint,
    find_latest_checkpoint,
)
from mlx.modes.object_detection.libreyolo.model_factory import (
    build_scratch_model, validate_incremental_adapter_support,
)
from mlx.modes.object_detection.libreyolo.utils import (
    dependency_error,
    resolve_dataset_source,
    resolve_imgsz,
    resolve_model_path,
    resolve_model_spec,
)
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest
from mlx.modes.object_detection.distillation import validate_distillation_options
from mlx.modes.object_detection.libreyolo.distillation import PrepareDistillationRun


class TrainLibreYOLOObjectDetection:
    def __init__(
        self,
        config: dict[str, Any] | TrainObjectDetectionRequest,
        *,
        reporter: WorkflowReporter | None = None,
    ) -> None:
        self.config = (
            config.to_config()
            if isinstance(config, TrainObjectDetectionRequest)
            else dict(config)
        )
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> dict[str, Any]:
        validate_distillation_options({"provider": "libreyolo", **self.config})
        if self.config.get("loss_clip") is not None:
            raise MLXUserError(
                "--loss-clip is not supported by LibreYOLO training. "
                "Remove the option or use --provider ultralytics."
            )
        if self.config.get("incremental_adapter_train_only") and not self.config.get(
            "incremental_adapter"
        ):
            raise MLXUserError(
                "--incremental-adapter-train-only requires --incremental-adapter."
            )
        if self.config.get("incremental_adapter_type") and not self.config.get(
            "incremental_adapter"
        ):
            raise MLXUserError(
                "--incremental-adapter-type requires --incremental-adapter."
            )

        try:
            from libreyolo import LibreYOLO
        except ImportError as exc:
            raise dependency_error("training an object-detection model") from exc

        model_spec = resolve_model_spec(self.config.get("model"))
        if (
            self.config.get("incremental_adapter")
            and model_spec.constructor_name != "LibreYOLOXDraxMobileNetV3Large"
        ):
            raise MLXUserError(
                "--incremental-adapter is supported only for "
                "yolox-drax-mobilenet-v3-large models."
            )
        explicit_weights = resolve_model_path(self.config.get("model_path"), required=False)
        if explicit_weights is not None and explicit_weights.suffix.lower() != ".pt":
            raise MLXUserError(
                "LibreYOLO training requires --model-path to point to a PyTorch "
                "checkpoint (.pt)."
            )
        dataset = resolve_dataset_source(self.config)
        project_dir = dataset.project_dir
        run_name = str(self.config.get("run_name") or "mlx-libreyolo")
        auto_resume, auto_warm_start = detect_existing_training_artifacts(
            project_dir=project_dir,
            run_name=run_name,
            explicit_weights=explicit_weights,
        )
        initialization_weights = explicit_weights or auto_warm_start
        if self.config.get("distiller") and auto_warm_start is not None:
            raise MLXUserError("Distillation requires explicit fine-tuning weights or a full resume checkpoint; choose a new output directory for scratch training.")

        if auto_resume is not None:
            emit(self.reporter, "info", f"Continuing LibreYOLO training from checkpoint: {auto_resume}")
        elif initialization_weights is not None:
            emit(self.reporter, "info", f"Warm-starting LibreYOLO from checkpoint: {initialization_weights}")

        try:
            if auto_resume is not None:
                model = LibreYOLO(
                    str(auto_resume),
                    device=self.config.get("device", "cpu"),
                    task="detect",
                )
            elif initialization_weights is not None:
                model = LibreYOLO(
                    str(initialization_weights),
                    device=self.config.get("device", "cpu"),
                    task="detect",
                )
            else:
                model = build_scratch_model(
                    model_spec, device=self.config.get("device", "cpu")
                )

            if self.config.get("incremental_adapter"):
                validate_incremental_adapter_support(model)
            train_kwargs = self._build_train_kwargs(
                data=dataset.data,
                project_dir=project_dir,
                run_name=run_name,
                resume=auto_resume is not None,
                allow_pretrained=auto_resume is None and initialization_weights is None,
            )
            train_kwargs.update(PrepareDistillationRun(
                self.config, project_dir / run_name, auto_resume
            ).execute())
            emit(self.reporter, "info", "Starting LibreYOLO training loop...")
            raw_results = model.train(**train_kwargs)
        except (
            AttributeError,
            OSError,
            ImportError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise MLXUserError(
                f"LibreYOLO training failed: {exc}. Check the model, dataset YAML, device, "
                "and training options."
            ) from exc

        results = dict(raw_results) if isinstance(raw_results, dict) else {"results": raw_results}
        selected_checkpoint = self._select_checkpoint(
            results,
            project_dir=project_dir,
            run_name=run_name,
            use_best=bool(self.config.get("use_best", True)),
        )
        if selected_checkpoint is not None:
            checkpoint_text = str(selected_checkpoint)
            results["model_path"] = checkpoint_text
            results["checkpoint_path"] = checkpoint_text
            emit(
                self.reporter,
                "success",
                f"Selected LibreYOLO checkpoint for downstream use: {selected_checkpoint}"
            )
        else:
            emit(
                self.reporter,
                "warning",
                "LibreYOLO training completed, but MLX could not find a .pt checkpoint.",
            )

        emit(self.reporter, "success", "LibreYOLO training complete!")
        return results

    def _build_train_kwargs(
        self,
        *,
        data: str,
        project_dir: Path,
        run_name: str,
        resume: bool,
        allow_pretrained: bool,
    ) -> dict[str, Any]:
        optimizer = str(self.config.get("optimizer") or "auto").lower()
        kwargs: dict[str, Any] = {
            "data": data,
            "epochs": int(self.config.get("epochs", 100)),
            "batch": int(self.config.get("batch_size", 16)),
            "imgsz": resolve_imgsz(self.config),
            "device": self.config.get("device", "cpu"),
            "project": str(project_dir),
            "name": run_name,
            "exist_ok": True,
            "resume": resume,
            "amp": bool(self.config.get("amp", True)),
            "nbs": int(self.config.get("nbs", 64)),
            "warmup_epochs": float(self.config.get("warmup_epochs", 3.0)),
            "save_plots": bool(self.config.get("plots", True)),
            "save_period": int(self.config.get("save_period", -1)),
        }
        if self.config.get("incremental_adapter"):
            kwargs["incremental_adapter"] = True
            kwargs["incremental_adapter_train_only"] = bool(
                self.config.get("incremental_adapter_train_only", False)
            )
            if self.config.get("incremental_adapter_type"):
                kwargs["incremental_adapter_type"] = str(
                    self.config["incremental_adapter_type"]
                )
        if allow_pretrained:
            kwargs["pretrained"] = bool(self.config.get("pretrained", False))
        if optimizer != "auto":
            kwargs["optimizer"] = optimizer
        if self.config.get("lr0") is not None:
            kwargs["lr0"] = float(self.config["lr0"])
        if self.config.get("random_seed") is not None:
            kwargs["seed"] = int(self.config["random_seed"])
        return kwargs

    def _select_checkpoint(
        self,
        results: dict[str, Any],
        *,
        project_dir: Path,
        run_name: str,
        use_best: bool,
    ) -> Optional[Path]:
        preferred_key = "best_checkpoint" if use_best else "last_checkpoint"
        fallback_key = "last_checkpoint" if use_best else "best_checkpoint"
        for key in (preferred_key, fallback_key):
            raw_path = results.get(key)
            if raw_path and Path(raw_path).is_file():
                if key == fallback_key:
                    emit(
                        self.reporter,
                        "warning",
                        f"Preferred LibreYOLO {preferred_key} was unavailable; "
                        f"using {fallback_key}."
                    )
                return Path(raw_path).resolve()

        save_dir = Path(results.get("save_dir") or (project_dir / run_name))
        preferred_name = "best.pt" if use_best else "last.pt"
        fallback_name = "last.pt" if use_best else "best.pt"
        preferred = find_existing_checkpoint(
            project_dir=save_dir,
            run_dir=save_dir,
            file_name=preferred_name,
        )
        if preferred is not None:
            return preferred
        fallback = find_existing_checkpoint(
            project_dir=save_dir,
            run_dir=save_dir,
            file_name=fallback_name,
        )
        if fallback is not None:
            emit(
                self.reporter,
                "warning",
                f"Preferred checkpoint {preferred_name} was unavailable; using {fallback_name}."
            )
            return fallback
        return find_latest_checkpoint(save_dir, pattern="*.pt")


def train_object_detection(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility function for direct LibreYOLO-provider callers."""

    from mlx.modes.object_detection.presentation import RichWorkflowReporter

    return TrainLibreYOLOObjectDetection(
        config,
        reporter=RichWorkflowReporter(),
    ).execute()
