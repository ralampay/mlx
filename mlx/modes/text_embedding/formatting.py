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

    def format_document(self, text: str) -> str:
        return self.document_prefix + text


def resolve_text_formatter(
    requested: str, *, query_prefix: str = "", document_prefix: str = ""
) -> RetrievalTextFormatter:
    if requested not in ("auto", "none", "e5"):
        raise MLXUserError("--prompt-format must be one of: auto, none, e5.")
    if requested == "e5":
        if query_prefix or document_prefix:
            raise MLXUserError(
                "--prompt-format e5 cannot be combined with nonempty --query-prefix "
                "or --document-prefix; choose one formatting method."
            )
        return RetrievalTextFormatter("e5", "query: ", "passage: ")
    return RetrievalTextFormatter("none", query_prefix, document_prefix)
