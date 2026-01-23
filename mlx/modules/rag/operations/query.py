import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from mlx.definitions import ModuleConfig
from mlx.modules.rag.helpers import (
    _create_embedding_runner,
    _determine_openai_chat_model,
    _generate_hf_answer,
    _generate_local_answer,
    _generate_openai_answer,
    _get_chroma_collection,
    _infer_collection_platform,
    console,
)
from mlx.modules.rag.utils import _resolve_db_config


class RagQuery:
    def __init__(self, config: ModuleConfig) -> None:
        self.table_name = config.get("table_name")
        if not self.table_name:
            raise typer.BadParameter("--table-name is required for query.")

        self.use_local = config.get("local", False)
        self.platform = config.get("platform")
        self.configured_model_name = config.get("model")
        self.generator_override = config.get("model_generator")
        self.top_k = config.get("top_k", 5)

    def execute(self) -> None:
        question = typer.prompt("Enter your question")
        if not question.strip():
            console.print("[yellow]No question provided; aborting.[/]")
            return

        db_config = _resolve_db_config()
        collection, _, _ = _get_chroma_collection(self.table_name, db_config)
        stored_platform = _infer_collection_platform(collection)

        if self.platform == "openai" and not self.use_local and stored_platform and stored_platform != "openai":
            raise typer.BadParameter(
                f"Collection '{self.table_name}' was indexed with platform '{stored_platform}'. "
                "Add --local to query local vectors or rebuild the collection with --platform openai."
            )
        if self.platform == "huggingface" and not self.use_local and stored_platform and stored_platform != "huggingface":
            raise typer.BadParameter(
                f"Collection '{self.table_name}' was indexed with platform '{stored_platform}'. "
                "Add --local to query local vectors or rebuild the collection with --platform huggingface."
            )

        if self.use_local:
            retrieval_platform = "local"
        elif self.platform == "openai":
            retrieval_platform = "openai"
        elif self.platform == "huggingface":
            retrieval_platform = "huggingface"
        elif stored_platform:
            retrieval_platform = stored_platform
        else:
            retrieval_platform = "local"

        embed_fn, _, run_platform = _create_embedding_runner(
            retrieval_platform == "local", self.configured_model_name, retrieval_platform
        )
        query_embeddings, _, _ = embed_fn([question])
        if not query_embeddings:
            console.print("[red]Failed to generate embedding for the query.[/]")
            return

        where_filter = None
        if run_platform:
            where_filter = {"platform": {"$eq": run_platform}}

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=self.top_k,
            include=["documents", "metadatas"],
            where=where_filter,
        )
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        if not documents or not documents[0]:
            console.print("[yellow]No matching documents were found in the collection.[/]")
            return

        retrieved_docs = documents[0]
        retrieved_meta = metadatas[0] if metadatas else []
        context_parts: List[str] = []
        sources: List[str] = []
        for idx, doc in enumerate(retrieved_docs):
            meta = retrieved_meta[idx] if idx < len(retrieved_meta) else {}
            source = meta.get("source", "unknown")
            sources.append(source)
            context_parts.append(f"[Source: {source}]\n{doc}")
        context = "\n\n".join(context_parts)

        console.print(Panel(question, title="User", border_style="cyan"))
        console.print(Panel(context, title="Retrieved Context", border_style="magenta"))

        generation_platform: Optional[str]
        if self.platform in {"openai", "huggingface"}:
            generation_platform = self.platform
        elif self.use_local:
            generation_platform = "local"
        else:
            generation_platform = None

        if generation_platform == "openai":
            model_for_generation = self.generator_override or self.configured_model_name
            if not model_for_generation:
                raise typer.BadParameter(
                    "Specify --model with an OpenAI chat model when using --platform openai for queries."
                )
            answer = _generate_openai_answer(question, context, model_for_generation)
            display_name = _determine_openai_chat_model(model_for_generation)
            console.print(Panel(answer, title=f"Assistant ({display_name})", border_style="green"))
        elif generation_platform == "huggingface":
            model_for_generation = self.generator_override or self.configured_model_name
            if not model_for_generation:
                raise typer.BadParameter(
                    "Specify --model with a Hugging Face text-generation repository when using --platform huggingface."
                )
            answer = _generate_hf_answer(question, context, model_for_generation)
            console.print(Panel(answer, title=f"Assistant ({model_for_generation})", border_style="green"))
        elif generation_platform == "local":
            local_generation_model = (
                os.environ.get("LOCAL_LLM_GENERATION_MODEL")
                or os.environ.get("LOCAL_LLM_MODEL")
            )
            if not local_generation_model:
                raise typer.BadParameter(
                    "Set LOCAL_LLM_GENERATION_MODEL (or fallback LOCAL_LLM_MODEL) to a text-generative GGUF for query responses."
                )
            answer = _generate_local_answer(question, context, local_generation_model)
            display_name = Path(local_generation_model).name
            console.print(Panel(answer, title=f"Assistant ({display_name})", border_style="green"))
        else:
            console.print("[yellow]No generation backend configured for this platform; showing context only.[/]")

        unique_sources = sorted({source for source in sources})
        source_table = Table(title="Sources", show_header=True)
        source_table.add_column("Index", style="cyan", justify="center")
        source_table.add_column("Path", style="white")
        for idx, source in enumerate(unique_sources, start=1):
            source_table.add_row(str(idx), source)
        console.print(source_table)
