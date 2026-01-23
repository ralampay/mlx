from itertools import groupby
import json
import uuid
from typing import List

import typer
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for older langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mlx.modules.rag.utils import (
    _connect_postgres,
    _ensure_valid_table_name,
    _resolve_db_config,
)

from mlx.definitions import ModuleConfig
from mlx.modules.rag.helpers import (
    _collect_dataset_chunks,
    _create_embedding_runner,
    _render_run_metadata,
)

console = Console()
def _format_vector(embedding: List[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"

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
        adapter = db_config.get("adapter", "chromadb").lower()
        db_status = "unknown"
        if adapter == "chromadb":
            from mlx.modules.rag.helpers import _get_chroma_collection

            collection, db_host, db_port = _get_chroma_collection(self.table_name, db_config)
            db_status = "connected"
        elif adapter in {"postgres", "postgresql"}:
            _ensure_valid_table_name(self.table_name)
            collection = None
            db_host = db_config.get("host", "unknown")
            db_port = db_config.get("port", "unknown")
            db_status = "connected"
        else:
            raise typer.BadParameter(
                "Unsupported DB_ADAPTER for batch-insert. Use 'chromadb' or 'postgres'."
            )

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

        conn = None
        table_initialized = False
        if adapter in {"postgres", "postgresql"}:
            conn = _connect_postgres(db_config)
            with conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        try:
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
                    if adapter == "chromadb":
                        collection.upsert(
                            ids=ids,
                            documents=documents,
                            metadatas=metadatas,
                            embeddings=embeddings,
                        )
                    else:
                        if embedding_size <= 0:
                            raise typer.BadParameter("Unable to determine embedding size for postgres.")
                        if conn is None:
                            raise typer.BadParameter("Postgres connection failed to initialize.")
                        with conn:
                            with conn.cursor() as cur:
                                if not table_initialized:
                                    cur.execute(
                                        f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
                                        id uuid PRIMARY KEY,
                                        content text NOT NULL,
                                        embedding vector({embedding_size}) NOT NULL,
                                        metadata jsonb NOT NULL
                                        )"""
                                    )
                                    table_initialized = True
                                rows = []
                                for row_id, content, embedding, metadata in zip(
                                    ids, documents, embeddings, metadatas
                                ):
                                    rows.append(
                                        (
                                            row_id,
                                            content,
                                            _format_vector(embedding),
                                            json.dumps(metadata),
                                        )
                                    )
                                cur.executemany(
                                    f"""INSERT INTO {self.table_name}
                                    (id, content, embedding, metadata)
                                    VALUES (%s, %s, %s::vector, %s::jsonb)""",
                                    rows,
                                )
                    progress.advance(task_id, len(batch))
        finally:
            if conn is not None:
                conn.close()

        db_username = db_config.get("username") or "not set"
        db_password_status = "No Password Set"
        if db_config.get("password"):
            db_password_status = "********"

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
            db_status=db_status,
            db_username=db_username,
            db_password_status=db_password_status,
            table_name=self.table_name,
            embedding_platform=run_platform,
        )
        console.print(
            f"[green]Inserted {total_chunks} chunk(s) into '{self.table_name}'.[/]"
        )
