from __future__ import annotations

from dataclasses import dataclass

from mlx.core.exceptions import MLXUserError


@dataclass(frozen=True)
class RetrievalTextFormatter:
    """Role-specific prefixes applied after document title/body composition."""

    effective_format: str = "none"
    query_prefix: str = ""
    document_prefix: str = ""

    def format_query(self, text: str) -> str:
        return self.query_prefix + text

    def format_document(self, text: str, *, title: str | None = None) -> str:
        if self.effective_format == "embeddinggemma":
            return f"title: {title or 'none'} | text: {text}"
        if title:
            text = f"{title}\n{text}"
        return self.document_prefix + text


def resolve_text_formatter(
    requested: str, *, query_prefix: str = "", document_prefix: str = ""
) -> RetrievalTextFormatter:
    if requested not in ("auto", "none", "e5", "embeddinggemma", "qwen3"):
        raise MLXUserError("--prompt-format must be one of: auto, none, e5, embeddinggemma, qwen3.")
    if requested == "embeddinggemma":
        if query_prefix or document_prefix:
            raise MLXUserError("EmbeddingGemma formatting cannot be combined with custom prefixes.")
        return RetrievalTextFormatter("embeddinggemma", "task: search result | query: ", "")
    if requested == "e5":
        if query_prefix or document_prefix:
            raise MLXUserError(
                "--prompt-format e5 cannot be combined with nonempty --query-prefix "
                "or --document-prefix; choose one formatting method."
            )
        return RetrievalTextFormatter("e5", "query: ", "passage: ")
    if requested == "qwen3":
        if query_prefix or document_prefix:
            raise MLXUserError("Qwen3 formatting cannot be combined with custom prefixes.")
        return RetrievalTextFormatter(
            "qwen3", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", ""
        )
    return RetrievalTextFormatter("none", query_prefix, document_prefix)
