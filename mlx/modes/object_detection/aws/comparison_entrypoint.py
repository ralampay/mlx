"""Evaluation-only SageMaker entrypoint for a completed detector comparison."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.aws.dataset import extract_sagemaker_dataset
from mlx.modes.object_detection.commands import BenchmarkObjectDetectionModel
from mlx.modes.object_detection.requests import BenchmarkObjectDetectionRequest


SAGEMAKER_INPUT = Path("/opt/ml/input/data/training")
SAGEMAKER_MODEL = Path("/opt/ml/model")
DATASET_DIR = Path("/tmp/mlx-dataset")


class RunSageMakerDetectionComparisonTest:
    def __init__(
        self,
        *,
        hyperparameters: Mapping[str, Any],
        input_dir: Path = SAGEMAKER_INPUT,
        dataset_dir: Path = DATASET_DIR,
        model_dir: Path = SAGEMAKER_MODEL,
        s3: Any = None,
        benchmark_command=BenchmarkObjectDetectionModel,
    ) -> None:
        self.hyperparameters = hyperparameters
        self.input_dir = input_dir
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.s3 = s3
        self.benchmark_command = benchmark_command

    def execute(self) -> dict[str, Any]:
        try:
            settings = json.loads(self.hyperparameters["mlx_comparison_test"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MLXUserError("Comparison test settings are missing or invalid.") from exc
        models = settings.get("models")
        if not isinstance(models, dict) or len(models) < 2:
            raise MLXUserError("Comparison test requires at least two model input channels.")
        volume = int(self.hyperparameters.get("mlx_volume_size_gb", 100))
        dataset = extract_sagemaker_dataset(
            self.input_dir, self.dataset_dir,
            max_uncompressed_bytes=volume * 700_000_000,
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {"dataset": str(dataset), "split": "test", "models": {}}
        for name, channel in models.items():
            if (not isinstance(name, str) or not name or "/" in name or ".." in name
                    or not isinstance(channel, str) or "/" in channel or ".." in channel):
                raise MLXUserError("Comparison model name or channel is invalid.")
            candidates = sorted((self.input_dir.parent / channel).rglob("*.pt"))
            if len(candidates) != 1:
                raise MLXUserError(
                    f"Expected one .pt checkpoint in the {channel} channel; found {len(candidates)}."
                )
            request = BenchmarkObjectDetectionRequest(
                provider=str(settings["provider"]),
                model_path=str(candidates[0]),
                dataset_path=str(dataset),
                output_path=str(self.model_dir / "benchmarks" / name),
                device="0",
                height=int(settings["height"]),
                width=int(settings["width"]),
                batch_size=int(settings["batch_size"]),
                confidence=float(settings["confidence"]),
                iou=float(settings["iou"]),
                max_detections=int(settings["max_detections"]),
                split="test",
                plots=False,
                verbose=False,
            )
            benchmark = self.benchmark_command(request).execute()
            validation_uri = settings.get("validation_s3_uris", {}).get(name)
            results["models"][name] = {
                "test_metrics": dict(benchmark.metrics),
                "validation_metrics": self._read_validation(validation_uri) if validation_uri else None,
                "checkpoint_file": candidates[0].name,
                "checkpoint_bytes": candidates[0].stat().st_size,
            }
        json_body = json.dumps(results, indent=2, sort_keys=True).encode()
        (self.model_dir / "results.json").write_bytes(json_body)
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["model", "split", "metric", "value"])
        for model, values in results["models"].items():
            for metric, value in sorted((values["validation_metrics"] or {}).items()):
                writer.writerow([model, "val", metric, value])
            for metric, value in sorted(values["test_metrics"].items()):
                writer.writerow([model, "test", metric, value])
        (self.model_dir / "results.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")
        self._upload(settings["results_s3_uri"], json_body, "application/json")
        self._upload(settings["results_csv_s3_uri"], csv_buffer.getvalue().encode(), "text/csv")
        return results

    def _upload(self, uri: str, body: bytes, content_type: str) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise MLXUserError(f"Invalid comparison result S3 URI: {uri}")
        try:
            self._s3_client().put_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/"),
                Body=body, ContentType=content_type,
            )
        except Exception as exc:
            raise MLXUserError(f"Unable to upload comparison result to {uri}: {exc}") from exc

    def _read_validation(self, uri: str) -> Mapping[str, Any]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise MLXUserError(f"Invalid validation S3 URI: {uri}")
        try:
            body = self._s3_client().get_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
            )["Body"].read()
            return json.loads(body)["metrics"]
        except Exception as exc:
            raise MLXUserError(f"Unable to read validation metrics at {uri}: {exc}") from exc

    def _s3_client(self):
        if self.s3 is None:
            import boto3
            self.s3 = boto3.client("s3")
        return self.s3
