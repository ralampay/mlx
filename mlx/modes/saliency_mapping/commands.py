from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.saliency_mapping.benchmark import BenchmarkSaliencyMapping
from mlx.modes.saliency_mapping.checkpoints import training_paths
from mlx.modes.saliency_mapping.models import grouped_model_names
from mlx.modes.saliency_mapping.requests import BenchmarkSaliencyRequest, TrainSaliencyRequest
from mlx.modes.saliency_mapping.train import TrainSaliencyModel

TrainerFactory = Callable[[TrainSaliencyRequest, WorkflowReporter], TrainSaliencyModel]


class TrainSaliencyModelGroup:
    def __init__(
        self,
        request: TrainSaliencyRequest,
        *,
        reporter: WorkflowReporter | None = None,
        trainer_factory: TrainerFactory | None = None,
    ) -> None:
        self.request = request
        self.reporter = reporter or NullWorkflowReporter()
        self.trainer_factory = trainer_factory or (
            lambda request, reporter: TrainSaliencyModel(request, reporter=reporter)
        )

    def execute(self) -> dict[str, Any]:
        if self.request.pretrained:
            raise MLXUserError("Grouped saliency training requires scratch initialization; omit --pretrained.")
        if self.request.model_path:
            raise MLXUserError("Grouped saliency training cannot resume multiple models from one --model-path.")
        if not self.request.output_path:
            raise MLXUserError("Grouped saliency training requires --output.")
        root = Path(self.request.output_path).expanduser()
        if root.exists():
            if not root.is_dir():
                raise MLXUserError(f"Grouped saliency output must be a directory: {root}")
            entries = list(root.iterdir())
            if self.request.dataset_s3_uri:
                entries = [entry for entry in entries if entry.name != "dataset_source.json"]
            if entries:
                raise MLXUserError(f"Grouped saliency output must be a new or empty directory: {root}")
        test_path = Path(self.request.dataset_path).expanduser() / "test"
        if not (test_path / "images").is_dir() or not (test_path / "masks").is_dir():
            raise MLXUserError("Grouped saliency training requires paired test/images and test/masks.")
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        names = grouped_model_names(str(self.request.model))
        for index, model_name in enumerate(names, start=1):
            emit(
                self.reporter,
                "progress",
                f"Training saliency model {model_name} ({index}/{len(names)}).",
                current=index - 1,
                total=len(names),
            )
            model_request = replace(
                self.request,
                model=model_name,
                output_path=str(root / model_name),
            )
            try:
                training = self.trainer_factory(model_request, self.reporter).execute()
                checkpoint = training_paths(model_request.to_config(), model_name=model_name)["checkpoint_path"]
                benchmark = BenchmarkSaliencyMapping(
                    BenchmarkSaliencyRequest.from_config(
                        {
                            **model_request.to_config(),
                            "model_path": str(checkpoint),
                            "output_path": str(root / model_name / "benchmark"),
                            "split": "test",
                        }
                    ),
                    reporter=self.reporter,
                ).execute()
                row = {
                    "model": model_name,
                    "best_validation_mae": training["best_validation_mae"],
                    **benchmark,
                }
                rows.append(row)
                self._write(root, names, rows, "running")
            except Exception as exc:
                self._write(root, names, rows, "failed", error=str(exc))
                raise MLXUserError(f"Grouped saliency training failed for '{model_name}': {exc}") from exc
        self._write(root, names, rows, "completed")
        return {"status": "completed", "model_group": self.request.model, "models": rows}

    @staticmethod
    def _write(root, names, rows, status, error=None):
        write_json_atomic(
            root / "all-models.json",
            {"status": status, "ranking_metric": "mae", "models": names, "results": rows, "error": error},
        )
        ranked = sorted(rows, key=lambda row: (float(row.get("mae", float("inf"))), row["model"]))
        write_csv(root / "leaderboard.csv", ({"rank": index, **row} for index, row in enumerate(ranked, 1)))


__all__ = ["TrainSaliencyModelGroup"]
