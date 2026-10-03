"""Pinned laptop retrieval suite and atomic conversion to the existing BEIR layout."""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, emit
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.data import BeirDatasetLoader


@dataclass(frozen=True)
class RetrievalDatasetSource:
    name: str
    repository: str
    revision: str
    documents: int
    queries: int


# Immutable revisions; a change in selection or conversion requires a new suite version.
NANO_SOURCES = (
    RetrievalDatasetSource("nano-arguana", "zeta-alpha-ai/NanoArguAna", "8f4a982d470a32c45817738b9d29042ca55d75ad", 3635, 50),
    RetrievalDatasetSource("nano-fiqa2018", "zeta-alpha-ai/NanoFiQA2018", "4163ba032953d5044a7a6244261413f609c14342", 4598, 50),
    RetrievalDatasetSource("nano-scidocs", "zeta-alpha-ai/NanoSCIDOCS", "484eb90549fc3f0b9c42b3551e80ceb999515537", 2210, 50),
    RetrievalDatasetSource("nano-touche2020", "zeta-alpha-ai/NanoTouche2020", "0d2f26ed8c5ad309f95c7f9499c70a40e140fccd", 5745, 49),
    RetrievalDatasetSource("nano-quora", "zeta-alpha-ai/NanoQuoraRetrieval", "2ab2d73e6c862026282808b913a34f4136928545", 5046, 50),
    RetrievalDatasetSource("nano-dbpedia", "zeta-alpha-ai/NanoDBPedia", "438f1c25129f05db6238699b5afdc9c6b58d2096", 6045, 50),
    RetrievalDatasetSource("nano-hotpotqa", "zeta-alpha-ai/NanoHotpotQA", "d79c0cdda980aba54842756770928035e1b61a51", 5090, 50),
    RetrievalDatasetSource("nano-msmarco", "zeta-alpha-ai/NanoMSMARCO", "7b8ff22f2771dc65ac5b439f222eb19a1f56abda", 5043, 50),
)
HELD_OUT_NANO_SOURCES = (
    RetrievalDatasetSource("nano-climatefever", "zeta-alpha-ai/NanoClimateFEVER", "96741bfa30b9f56db8c9eb7d08e775ed6474f206", 3408, 50),
    RetrievalDatasetSource("nano-fever", "zeta-alpha-ai/NanoFEVER", "a8bfdf1bf15181167a7e22e69cf8754bdea9b4c8", 4996, 50),
    RetrievalDatasetSource("nano-nq", "zeta-alpha-ai/NanoNQ", "77540146379abf95df8326a3c5bb9eb21c7146c3", 5035, 50),
)
SUITE_DATASETS = ("nfcorpus", "scifact") + tuple(source.name for source in NANO_SOURCES)
DATASET_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")


def validate_suite(suite):
    if suite not in ("laptop-ae-v1", "held-out-nano-v1"):
        raise MLXUserError(f"Unknown retrieval suite {suite!r}; choose laptop-ae-v1 or held-out-nano-v1.")


def dataset_hashes(root):
    return {name: sha256_file(Path(root) / name) for name in DATASET_FILES}


class HuggingFaceRetrievalSource:
    """Provider boundary: download pinned Parquet shards; no remote Python execution."""

    def fetch(self, source, destination):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(source.repository, repo_type="dataset", revision=source.revision,
                              allow_patterns=["corpus/*.parquet", "queries/*.parquet", "qrels/*.parquet", "README.md"],
                              local_dir=destination)
        except Exception as exc:
            raise MLXUserError(f"Unable to download {source.repository}@{source.revision}: {exc}") from exc

    def rows(self, directory, kind):
        try:
            import pyarrow.parquet as pq
            files = sorted((Path(directory) / kind).glob("*.parquet"))
            if not files:
                raise MLXUserError(f"Downloaded dataset has no {kind} Parquet shards.")
            for path in files:
                yield from pq.read_table(path).to_pylist()
        except MLXUserError:
            raise
        except Exception as exc:
            raise MLXUserError(f"Unable to read retrieval {kind} Parquet data: {exc}") from exc


class PrepareRetrievalDatasets:
    def __init__(self, dataset_path, *, suite="laptop-ae-v1", source=None, reporter=None,
                 sources=None):
        self.root = Path(dataset_path).expanduser()
        self.suite = suite
        self.source = source or HuggingFaceRetrievalSource()
        self.reporter = reporter or NullWorkflowReporter()
        self.sources = sources if sources is not None else (
            HELD_OUT_NANO_SOURCES if suite == "held-out-nano-v1" else NANO_SOURCES)

    def execute(self):
        validate_suite(self.suite)
        self.root.mkdir(parents=True, exist_ok=True)
        completed = []
        for source in self.sources:
            target = self.root / source.name
            if target.exists():
                self._verify_existing(target, source)
            else:
                self._prepare(source, target)
            completed.append(str(target))
            emit(self.reporter, "info", f"Dataset ready: {source.name}", payload={"event": "retrieval_stage"})
        return {"suite": self.suite, "datasets": completed}

    def _verify_existing(self, target, source):
        try:
            manifest = json.loads((target / "source_manifest.json").read_text())
            if (manifest["revision"] != source.revision or manifest["repository"] != source.repository
                    or manifest["conversion_version"] != 1 or manifest["hashes"] != dataset_hashes(target)):
                raise ValueError("source identity or contents differ")
            BeirDatasetLoader(allow_empty_documents=True).load(target)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MLXUserError(f"Existing dataset conflicts with the pinned suite: {target}. Use a different root or inspect its manifest.") from exc

    def _prepare(self, source, target):
        temporary = Path(tempfile.mkdtemp(prefix=f".{source.name}-", dir=self.root))
        try:
            raw = temporary / "source"
            self.source.fetch(source, raw)
            converted = temporary / "converted"
            (converted / "qrels").mkdir(parents=True)
            for kind in ("corpus", "queries"):
                with (converted / f"{kind}.jsonl").open("w", encoding="utf-8") as output:
                    for row in self.source.rows(raw, kind):
                        record = {"_id": row["_id"], "text": row["text"]}
                        if kind == "corpus":
                            record["title"] = row.get("title", "")
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
            pairs = set()
            with (converted / "qrels/test.tsv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(("query-id", "corpus-id", "score"))
                for row in self.source.rows(raw, "qrels"):
                    pair = (row["query-id"], row["corpus-id"])
                    if pair in pairs:
                        raise MLXUserError(f"Duplicate source judgment: {pair}")
                    pairs.add(pair)
                    writer.writerow((*pair, 1))
            data = BeirDatasetLoader(allow_empty_documents=True).load(converted)
            if len(data.corpus) != source.documents or len(data.queries) != source.queries:
                raise MLXUserError(f"Unexpected row counts in pinned dataset {source.name}.")
            write_json_atomic(converted / "source_manifest.json", {
                "conversion_version": 1, "suite": self.suite, "repository": source.repository,
                "revision": source.revision, "source_split": "train", "evaluation_split": "test",
                "relevance": "positive pairs mapped to binary score 1", "title_default": "", "empty_document_policy": "preserve; embed formatted empty text",
                "documents": len(data.corpus), "queries": len(data.queries), "qrels": len(data.qrels),
                "hashes": dataset_hashes(converted),
                "source_hashes": {str(p.relative_to(raw)): sha256_file(p) for p in raw.rglob("*.parquet")},
            })
            converted.rename(target)
        except MLXUserError:
            raise
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise MLXUserError(f"Unable to prepare dataset {source.name}: {exc}") from exc
        finally:
            shutil.rmtree(temporary)
