from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.models import (
    CorpusDocument,
    RelevanceJudgment,
    RetrievalDataset,
    RetrievalQuery,
)


class BeirDatasetLoader:
    """Load and validate the small, standard BEIR retrieval dataset layout."""

    def load(self, path: str | Path) -> RetrievalDataset:
        root = Path(path).expanduser()
        if not root.is_dir():
            raise MLXUserError(f"Text-embedding dataset directory not found: {root}")
        corpus_path = self._required(root / "corpus.jsonl", "corpus.jsonl")
        queries_path = self._required(root / "queries.jsonl", "queries.jsonl")
        qrels_path = self._required(root / "qrels" / "test.tsv", "qrels/test.tsv")

        corpus = self._load_corpus(corpus_path)
        queries = self._load_queries(queries_path)
        qrels = self._load_qrels(qrels_path)
        self._validate_references(corpus, queries, qrels)
        return RetrievalDataset(
            name=root.name,
            source_path=root,
            corpus=tuple(corpus),
            queries=tuple(queries),
            qrels=tuple(qrels),
        )

    @staticmethod
    def _required(path: Path, label: str) -> Path:
        if not path.is_file():
            raise MLXUserError(f"BEIR dataset is missing required file: {label} ({path})")
        return path

    def _load_corpus(self, path: Path) -> list[CorpusDocument]:
        rows = self._load_jsonl(path)
        documents: list[CorpusDocument] = []
        seen: set[str] = set()
        for line, row in rows:
            identifier = self._required_text(row, "_id", path, line)
            if identifier in seen:
                raise MLXUserError(f"Duplicate corpus document ID '{identifier}' in {path} at line {line}.")
            title = row.get("title", "")
            text = row.get("text")
            if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
                raise MLXUserError(
                    f"Invalid corpus record in {path} at line {line}: title must be a string and text must be non-empty."
                )
            seen.add(identifier)
            documents.append(CorpusDocument(identifier, title, text))
        if not documents:
            raise MLXUserError(f"BEIR corpus contains no documents: {path}")
        return documents

    def _load_queries(self, path: Path) -> list[RetrievalQuery]:
        rows = self._load_jsonl(path)
        queries: list[RetrievalQuery] = []
        seen: set[str] = set()
        for line, row in rows:
            identifier = self._required_text(row, "_id", path, line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise MLXUserError(f"Invalid query in {path} at line {line}: text must be non-empty.")
            if identifier in seen:
                raise MLXUserError(f"Duplicate query ID '{identifier}' in {path} at line {line}.")
            seen.add(identifier)
            queries.append(RetrievalQuery(identifier, text))
        if not queries:
            raise MLXUserError(f"BEIR query file contains no queries: {path}")
        return queries

    @staticmethod
    def _load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []
        try:
            with path.open(encoding="utf-8") as source:
                for line_number, raw in enumerate(source, start=1):
                    if not raw.strip():
                        continue
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise MLXUserError(
                            f"Malformed JSONL in {path} at line {line_number}: {exc.msg}."
                        ) from exc
                    if not isinstance(value, dict):
                        raise MLXUserError(f"Invalid JSONL object in {path} at line {line_number}.")
                    rows.append((line_number, value))
        except (OSError, UnicodeError) as exc:
            raise MLXUserError(f"Unable to read BEIR JSONL file '{path}': {exc}") from exc
        return rows

    @staticmethod
    def _required_text(row: dict[str, Any], key: str, path: Path, line: int) -> str:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise MLXUserError(f"Invalid record in {path} at line {line}: '{key}' must be non-empty.")
        return value

    @staticmethod
    def _load_qrels(path: Path) -> list[RelevanceJudgment]:
        judgments: list[RelevanceJudgment] = []
        first_record = True
        try:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.reader(source, delimiter="\t")
                for line_number, row in enumerate(reader, start=1):
                    if not row or all(not cell.strip() for cell in row):
                        continue
                    if first_record and row[0].strip().lower() in {"query-id", "query_id"}:
                        first_record = False
                        continue
                    first_record = False
                    if len(row) < 3:
                        raise MLXUserError(
                            f"Invalid qrels row in {path} at line {line_number}: expected query ID, corpus ID, and score."
                        )
                    try:
                        relevance = int(row[2].strip())
                    except ValueError as exc:
                        raise MLXUserError(
                            f"Invalid qrels score in {path} at line {line_number}: {row[2]!r}."
                        ) from exc
                    query_id, document_id = row[0].strip(), row[1].strip()
                    if not query_id or not document_id:
                        raise MLXUserError(
                            f"Invalid qrels identifiers in {path} at line {line_number}."
                        )
                    judgments.append(RelevanceJudgment(query_id, document_id, relevance))
        except (OSError, UnicodeError) as exc:
            raise MLXUserError(f"Unable to read BEIR qrels file '{path}': {exc}") from exc
        if not judgments:
            raise MLXUserError(f"BEIR qrels file contains no relevance judgments: {path}")
        return judgments

    @staticmethod
    def _validate_references(corpus, queries, qrels) -> None:
        document_ids = {item.id for item in corpus}
        query_ids = {item.id for item in queries}
        for judgment in qrels:
            if judgment.query_id not in query_ids:
                raise MLXUserError(
                    f"Qrels reference unknown query ID '{judgment.query_id}'."
                )
            if judgment.document_id not in document_ids:
                raise MLXUserError(
                    f"Qrels reference unknown corpus document ID '{judgment.document_id}'."
                )


__all__ = ["BeirDatasetLoader"]
