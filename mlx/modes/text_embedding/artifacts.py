from __future__ import annotations

import csv
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mlx.core.artifacts import sha256_file, write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.models import EmbeddedText, RelevanceJudgment, RetrievalQuery


SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_new_output_directory(path: str | Path, *, purpose: str) -> Path:
    output = Path(path).expanduser()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise MLXUserError(
            f"{purpose} output must be a new or empty directory: {output}"
        )
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MLXUserError(f"Unable to create output directory '{output}': {exc}") from exc
    return output


class EmbeddingArtifactWriter:
    def initialize_csv(self, path: Path, *, kind: str) -> None:
        fields = ["id", "title", "text", "embedding"] if kind == "corpus" else ["id", "text", "embedding"]
        self._write_rows(path, (), fields, append=False)

    def append_embeddings(
        self, path: Path, records: Sequence[EmbeddedText], *, kind: str
    ) -> None:
        fields = ["id", "title", "text", "embedding"] if kind == "corpus" else ["id", "text", "embedding"]
        rows = []
        for record in records:
            row = {
                "id": record.id,
                "text": record.text,
                "embedding": json.dumps(list(record.vector), separators=(",", ":")),
            }
            if kind == "corpus":
                row["title"] = str(record.metadata.get("title", ""))
            rows.append(row)
        self._write_rows(path, rows, fields, append=True)

    @staticmethod
    def _write_rows(
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        fields: Sequence[str],
        *,
        append: bool,
    ) -> None:
        try:
            with path.open("a" if append else "w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                if not append:
                    writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            raise MLXUserError(f"Unable to write embedding CSV '{path}': {exc}") from exc

    def write_manifests(
        self,
        output_dir: Path,
        *,
        dataset,
        model_path: Path,
        dimensions: int,
        normalized: bool,
        vector_store: str,
        query_prefix: str,
        document_prefix: str,
        representation: str,
        source_dimensions: int,
        adapter: Mapping[str, Any] | None,
        started_at: str,
        backend: str = "llama-cpp-python",
        embedding_configuration: Mapping[str, Any] | None = None,
    ) -> None:
        qrels_name = "qrels.tsv"
        self._write_qrels(output_dir / qrels_name, dataset.qrels)
        source_reference = os.path.relpath(dataset.source_path.resolve(), output_dir.resolve())
        dataset_manifest = {
            "schema_version": SCHEMA_VERSION,
            "name": dataset.name,
            "source_path": source_reference,
            "qrels_path": qrels_name,
            "corpus_documents": len(dataset.corpus),
            "queries": len(dataset.queries),
            "qrels": len(dataset.qrels),
        }
        embedding_manifest = {
            "embedding_configuration": dict(embedding_configuration or {}),
            "schema_version": SCHEMA_VERSION,
            "model": {
                "path": model_path.name,
                "sha256": sha256_file(model_path),
                "backend": backend,
            },
            "dataset": dataset_manifest,
            "embedding": {
                "dimensions": dimensions,
                "source_dimensions": source_dimensions,
                "normalized": normalized,
                "query_prefix": query_prefix,
                "document_prefix": document_prefix,
                "representation": representation,
            },
            "vector_store": {"provider": vector_store, "similarity": "cosine"},
        }
        if adapter is not None:
            embedding_manifest["adapter"] = dict(adapter)
        write_json_atomic(output_dir / "dataset_manifest.json", dataset_manifest)
        write_json_atomic(output_dir / "embedding_manifest.json", embedding_manifest)
        write_json_atomic(
            output_dir / "run_metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "mode": "text_embedding",
                "action": "embed",
                "embedding_configuration": dict(embedding_configuration or {}),
                "started_at": started_at,
                "completed_at": utc_timestamp(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
        )

    @staticmethod
    def _write_qrels(path: Path, qrels: Sequence[RelevanceJudgment]) -> None:
        try:
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(("query-id", "corpus-id", "score"))
                writer.writerows(
                    (item.query_id, item.document_id, item.relevance) for item in qrels
                )
        except OSError as exc:
            raise MLXUserError(f"Unable to write qrels artifact '{path}': {exc}") from exc


class EmbeddingArtifactReader:
    def load(self, root: str | Path) -> dict[str, Any]:
        directory = Path(root).expanduser()
        if not directory.is_dir():
            raise MLXUserError(f"Embedding artifact directory not found: {directory}")
        dataset_manifest = self._read_json(directory / "dataset_manifest.json")
        embedding_manifest = self._read_json(directory / "embedding_manifest.json")
        if dataset_manifest.get("schema_version") != SCHEMA_VERSION or embedding_manifest.get("schema_version") != SCHEMA_VERSION:
            raise MLXUserError("Unsupported or missing embedding artifact schema version.")
        self._validate_manifests(dataset_manifest, embedding_manifest)
        corpus_scan = self._scan_embedding_csv(
            directory / "corpus_embeddings.csv", query=False, collect=False
        )
        query_scan = self._scan_embedding_csv(
            directory / "query_embeddings.csv", query=True, collect=True
        )
        query_rows = query_scan["rows"]
        qrels_path = directory / str(dataset_manifest.get("qrels_path", "qrels.tsv"))
        qrels = self._read_qrels(qrels_path)
        expected_dimensions = embedding_manifest.get("embedding", {}).get("dimensions")
        if not isinstance(expected_dimensions, int) or expected_dimensions < 1:
            raise MLXUserError("Embedding manifest has an invalid dimensions value.")
        if query_scan["dimensions"] != {expected_dimensions}:
            raise MLXUserError("Query embedding dimensionality does not match embedding_manifest.json.")
        if corpus_scan["dimensions"] != {expected_dimensions}:
            raise MLXUserError("Corpus embedding dimensionality does not match embedding_manifest.json.")
        if corpus_scan["count"] != dataset_manifest.get("corpus_documents"):
            raise MLXUserError("Corpus embedding count does not match dataset_manifest.json.")
        if query_scan["count"] != dataset_manifest.get("queries"):
            raise MLXUserError("Query embedding count does not match dataset_manifest.json.")
        if len(qrels) != dataset_manifest["qrels"]:
            raise MLXUserError("Qrels count does not match dataset_manifest.json.")
        for judgment in qrels:
            if (
                judgment.query_id not in query_scan["ids"]
                or judgment.document_id not in corpus_scan["ids"]
            ):
                raise MLXUserError(
                    "Embedding artifact qrels reference IDs absent from the exported embeddings."
                )
        return {
            "root": directory,
            "dataset_manifest": dataset_manifest,
            "embedding_manifest": embedding_manifest,
            "queries": tuple(
                (RetrievalQuery(identifier, text), vector)
                for identifier, text, vector in query_rows
            ),
            "qrels": tuple(qrels),
            "corpus_ids": frozenset(corpus_scan["ids"]),
        }

    @staticmethod
    def _validate_manifests(dataset, manifest) -> None:
        if not isinstance(manifest.get("embedding_configuration", {}), dict):
            raise MLXUserError("Embedding manifest has an invalid embedding_configuration object.")
        for field in ("corpus_documents", "queries", "qrels"):
            if type(dataset.get(field)) is not int or dataset[field] < 1:
                raise MLXUserError(f"Dataset manifest has an invalid {field} value.")
        if not isinstance(dataset.get("name"), str) or not dataset["name"].strip():
            raise MLXUserError("Dataset manifest requires a non-empty name.")
        for section, fields in {
            "model": ("path", "sha256"),
            "embedding": ("dimensions", "normalized"),
            "vector_store": ("provider", "similarity"),
        }.items():
            value = manifest.get(section)
            if not isinstance(value, dict) or any(field not in value for field in fields):
                raise MLXUserError(f"Embedding manifest has an invalid {section} section.")
        embedding = manifest["embedding"]
        if type(embedding["dimensions"]) is not int or embedding["dimensions"] < 1:
            raise MLXUserError("Embedding manifest has invalid dimensions.")
        if type(embedding["normalized"]) is not bool:
            raise MLXUserError("Embedding manifest normalization must be Boolean.")
        for section, field in (("model", "path"), ("model", "sha256"),
                               ("vector_store", "provider")):
            if not isinstance(manifest[section][field], str) or not manifest[section][field].strip():
                raise MLXUserError(f"Embedding manifest requires {section}.{field}.")
        if manifest["vector_store"]["similarity"] != "cosine":
            raise MLXUserError("Embedding manifest similarity must be cosine.")
        embedded_dataset = manifest.get("dataset")
        if not isinstance(embedded_dataset, dict) or any(
            embedded_dataset.get(key) != dataset.get(key)
            for key in ("name", "corpus_documents", "queries", "qrels")
        ):
            raise MLXUserError("Embedding and dataset manifests describe different datasets.")
        if not isinstance(dataset.get("qrels_path", "qrels.tsv"), str):
            raise MLXUserError("Dataset manifest qrels_path must be a string.")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise MLXUserError(f"Embedding artifact is missing required file: {path.name}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MLXUserError(f"Corrupt JSON embedding artifact '{path}': {exc}") from exc
        if not isinstance(value, dict):
            raise MLXUserError(f"Embedding artifact must contain a JSON object: {path}")
        return value

    @staticmethod
    def _scan_embedding_csv(
        path: Path, *, query: bool, collect: bool
    ) -> dict[str, Any]:
        if not path.is_file():
            raise MLXUserError(f"Embedding artifact is missing required file: {path.name}")
        rows: list[tuple[str, str, tuple[float, ...]]] = []
        seen: set[str] = set()
        dimensions: set[int] = set()
        count = 0
        try:
            with path.open(newline="", encoding="utf-8") as source:
                for line, row in enumerate(csv.DictReader(source), start=2):
                    identifier = row.get("id", "")
                    text = row.get("text", "")
                    if not identifier or identifier in seen or (query and not text):
                        raise MLXUserError(f"Invalid or duplicate embedding row in {path} at line {line}.")
                    try:
                        raw = json.loads(row.get("embedding", ""))
                        if not isinstance(raw, list) or any(
                            isinstance(value, bool) or not isinstance(value, (int, float))
                            for value in raw
                        ):
                            raise ValueError("Expected a numeric JSON array.")
                        vector = tuple(float(value) for value in raw)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise MLXUserError(f"Invalid embedding vector in {path} at line {line}.") from exc
                    if not vector:
                        raise MLXUserError(f"Empty embedding vector in {path} at line {line}.")
                    if any(not math.isfinite(value) for value in vector):
                        raise MLXUserError(f"Non-finite embedding vector in {path} at line {line}.")
                    seen.add(identifier)
                    dimensions.add(len(vector))
                    count += 1
                    if collect:
                        rows.append((identifier, text, vector))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise MLXUserError(f"Unable to read embedding CSV '{path}': {exc}") from exc
        if not count:
            raise MLXUserError(f"Embedding CSV contains no records: {path}")
        return {
            "rows": rows,
            "ids": seen,
            "dimensions": dimensions,
            "count": count,
        }

    @staticmethod
    def _read_qrels(path: Path) -> list[RelevanceJudgment]:
        if not path.is_file():
            raise MLXUserError(f"Embedding artifact qrels file not found: {path}")
        judgments: list[RelevanceJudgment] = []
        try:
            with path.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source, delimiter="\t"):
                    judgments.append(
                        RelevanceJudgment(
                            str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
                        )
                    )
        except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
            raise MLXUserError(f"Corrupt qrels embedding artifact '{path}': {exc}") from exc
        if not judgments:
            raise MLXUserError(f"Embedding artifact qrels contains no judgments: {path}")
        return judgments


class BenchmarkArtifactWriter:
    def write(
        self,
        output_dir: Path,
        *,
        summary: Mapping[str, Any],
        query_rows: Sequence[Mapping[str, Any]],
        rankings: Sequence[Mapping[str, Any]],
        failures: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        started_at: str,
    ) -> None:
        write_json_atomic(output_dir / "metrics.json", summary)
        metrics = summary["metrics"]
        write_csv(
            output_dir / "metrics.csv",
            [{
                "dataset": summary["dataset"],
                "model": summary["model"],
                "representation": summary["representation"],
                "dimensions": summary["dimensions"],
                "similarity": summary["similarity"],
                "queries": summary["query_count"],
                "corpus_size": summary["corpus_size"],
                **{
                    key: summary.get("embedding_configuration", {}).get(key, "unknown")
                    for key in ("pooling_requested", "pooling_effective",
                                "prompt_format_requested", "prompt_format_effective")
                },
                **metrics,
            }],
        )
        write_csv(output_dir / "query_metrics.csv", query_rows)
        self._write_jsonl(output_dir / "rankings.jsonl", rankings)
        write_csv(
            output_dir / "failures.csv",
            failures,
            fieldnames=(
                "query_id",
                "query",
                "relevant_documents",
                "best_relevant_rank",
                "reciprocal_rank",
            ),
        )
        write_json_atomic(output_dir / "benchmark_manifest.json", manifest)
        write_json_atomic(
            output_dir / "run_metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "mode": "text_embedding",
                "action": "benchmark",
                "embedding_configuration": summary.get("embedding_configuration", {}),
                "started_at": started_at,
                "completed_at": utc_timestamp(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
        )
        self._write_report(output_dir / "report.md", summary, failures)

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        try:
            with path.open("w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        except OSError as exc:
            raise MLXUserError(f"Unable to write rankings artifact '{path}': {exc}") from exc

    @staticmethod
    def _write_report(path: Path, summary: Mapping[str, Any], failures) -> None:
        metric_rows = "\n".join(
            f"| {name} | {value:.6f} |" for name, value in sorted(summary["metrics"].items())
        )
        configuration = summary.get("embedding_configuration", {})
        configuration_rows = "\n".join(
            f"- {key}: `{configuration.get(key, 'unknown')}`"
            for key in ("pooling_requested", "pooling_effective",
                        "prompt_format_requested", "prompt_format_effective")
        )
        artifacts = (
            "metrics.json", "metrics.csv", "query_metrics.csv", "rankings.jsonl",
            "failures.csv", "benchmark_manifest.json", "run_metadata.json",
        )
        text = f"""# Text Embedding Retrieval Benchmark

## Configuration

- Representation: `{summary['representation']}`
- Similarity: `{summary['similarity']}`
- Embedding dimensions: {summary['dimensions']}

## Dataset

- Dataset: `{summary['dataset']}`
- Corpus documents: {summary['corpus_size']}
- Evaluated queries: {summary['query_count']}
- Excluded unjudged queries: {summary.get('excluded_queries', 0)}

## Embedding Model

- Model: `{summary['model']}`
- Model SHA-256: `{summary['model_sha256']}`
{configuration_rows}

## Retrieval Configuration

- Vector store: `{summary['vector_store']}`
- Maximum retrieval depth: {summary['top_k']}

## Metrics

| Metric | Value |
| --- | ---: |
{metric_rows}

## Query-Level Analysis

Per-query values are available in `query_metrics.csv`.

## Retrieval Failures

Queries with no relevant result in the retrieval depth: {len(failures)}.

## Artifact Inventory

""" + "\n".join(f"- [{name}]({name})" for name in artifacts) + "\n"
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise MLXUserError(f"Unable to write benchmark report '{path}': {exc}") from exc


__all__ = [
    "BenchmarkArtifactWriter",
    "EmbeddingArtifactReader",
    "EmbeddingArtifactWriter",
    "SCHEMA_VERSION",
    "prepare_new_output_directory",
    "utc_timestamp",
]
