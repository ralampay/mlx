from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    text: str


@dataclass(frozen=True)
class RelevanceJudgment:
    query_id: str
    document_id: str
    relevance: int


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    source_path: Path
    corpus: Sequence[CorpusDocument]
    queries: Sequence[RetrievalQuery]
    qrels: Sequence[RelevanceJudgment]


@dataclass(frozen=True)
class EmbeddedText:
    id: str
    text: str
    vector: tuple[float, ...]
    metadata: Mapping[str, Any]


def document_embedding_text(document: CorpusDocument) -> str:
    """Build provider-neutral document text using the BEIR title convention."""

    return f"{document.title}\n{document.text}" if document.title else document.text


__all__ = [
    "CorpusDocument",
    "EmbeddedText",
    "RelevanceJudgment",
    "RetrievalDataset",
    "RetrievalQuery",
    "document_embedding_text",
]
