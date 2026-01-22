from itertools import groupby
import uuid

import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mlx.modules.rag.utils import _resolve_db_config

from mlx.definitions import ModuleConfig
from mlx.modules.rag.run import (
    _collect_dataset_chunks,
    _create_embedding_runner,
    _get_chroma_collection,
    _render_run_metadata,
)

console = Console()

class RagBatchInsert:
    def __init__(self, config: ModuleConfig) -> None:
        self.chunk_size = config.get("chunk_size", 800)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.use_local = config.get("local", False)
        self.configured_model_name = config.get("model")
        self.platform = config.get("platform")
        self.table_name = config.get("table_name")
        self.dataset_path = config.get("dataset_path")
        self.file_limit = config.get("file_limit")

        if not self.dataset_path:
            raise typer.BadParameter("--dataset-path is required for batch-insert.")
        if not self.table_name:
            raise typer.BadParameter("--table-name is required for batch-insert.")

    def execute(self) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunk_records, processed_files = _collect_dataset_chunks(
            self.dataset_path,
            splitter,
            max_files=self.file_limit,
        )
        if not chunk_records:
            raise typer.BadParameter("No content available for insertion.")

        embed_fn, model_name, run_platform = _create_embedding_runner(
            self.use_local, self.configured_model_name, self.platform
        )
        db_config = _resolve_db_config()
        collection, db_host, db_port = _get_chroma_collection(self.table_name, db_config)

        total_tokens = 0
        embedding_size = 0
        total_chunks = len(chunk_records)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.completed}/{task.total} chunks"),
            transient=True,
            console=console,
        )

        with progress:
            task_id = progress.add_task(
                f"Inserting into {self.table_name}",
                total=total_chunks,
            )
            for source, group in groupby(chunk_records, key=lambda record: record["source"]):
                batch = list(group)
                documents = [item["text"] for item in batch]
                embeddings, token_count, dim = embed_fn(documents)
                total_tokens += token_count
                if embedding_size == 0:
                    if dim:
                        embedding_size = dim
                    elif len(embeddings) > 0:
                        embedding_size = len(embeddings[0])

                ids = [str(uuid.uuid4()) for _ in batch]
                metadatas = [
                    {
                        "chunk_id": item["id"],
                        "chunk_index": item["global_index"],
                        "source": item["source"],
                        "table_name": self.table_name,
                        "model_name": model_name,
                        "platform": run_platform,
                    }
                    for item in batch
                ]
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                progress.advance(task_id, len(batch))

        _render_run_metadata(
            model_name=model_name,
            embedding_size=embedding_size,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            dataset_path=self.dataset_path,
            file_count=processed_files,
            row_count=total_chunks,
            total_tokens=total_tokens,
            db_adapter=db_config["adapter"],
            db_host=db_host,
            db_port=str(db_port),
            table_name=self.table_name,
            embedding_platform=run_platform,
        )
        console.print(
            f"[green]Inserted {total_chunks} chunk(s) into '{self.table_name}'.[/]"
        )
