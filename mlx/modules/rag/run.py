from typing import Callable, Dict

import typer

from mlx.definitions import ModuleConfig


def _rag_vectorization_summary(config: ModuleConfig) -> None:
    from mlx.modules.rag.operations import RagVectorizationSummary

    RagVectorizationSummary(config).execute()


def _rag_batch_insert(config: ModuleConfig) -> None:
    from mlx.modules.rag.operations import RagBatchInsert

    RagBatchInsert(config).execute()


def _rag_delete_all(config: ModuleConfig) -> None:
    from mlx.modules.rag.operations import RagDeleteAll

    RagDeleteAll(config).execute()


def _rag_query(config: ModuleConfig) -> None:
    from mlx.modules.rag.operations import RagQuery

    RagQuery(config).execute()


ACTIONS: Dict[str, Callable[[ModuleConfig], None]] = {
    "vectorization-summary": _rag_vectorization_summary,
    "batch-insert": _rag_batch_insert,
    "delete-all": _rag_delete_all,
    "query": _rag_query,
}


def run(config: ModuleConfig) -> None:
    action = config.get("action")
    handler = ACTIONS.get(action)
    if handler is None:
        available = ", ".join(sorted(ACTIONS))
        raise typer.BadParameter(
            f"Unsupported RAG action '{action}'. Available actions: {available}."
        )

    handler(config)
