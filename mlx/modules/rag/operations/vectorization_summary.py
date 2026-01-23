import uuid

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for older langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from mlx.definitions import ModuleConfig
from mlx.modules.rag.helpers import (
    _collect_dataset_chunks,
    _create_embedding_runner,
    _get_chroma_collection,
    _render_run_metadata,
    console,
)
from mlx.modules.rag.utils import _connect_postgres, _ensure_valid_table_name, _resolve_db_config


class RagVectorizationSummary:
    def __init__(self, config: ModuleConfig) -> None:
        self.chunk_size = config.get("chunk_size", 800)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.use_local = config.get("local", False)
        self.platform = config.get("platform")
        self.configured_model_name = config.get("model")
        self.table_name = config.get("table_name")
        self.dataset_path = config.get("dataset_path")
        self.file_limit = config.get("file_limit")
        self.show_sample = config.get("show_sample", True)

        if not self.dataset_path:
            raise typer.BadParameter("--dataset-path is required for vectorization-summary.")
        if not self.table_name:
            raise typer.BadParameter("--table-name is required for vectorization-summary.")

    def execute(self) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        def _on_file(path) -> None:
            console.log(f"Processing {path}")
            progress.update(task_id, description=f"Processing {path.name}")

        console.log("Scanning dataset for supported files...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as progress:
            task_id = progress.add_task("Scanning dataset...", total=None)
            chunk_records, processed_files = _collect_dataset_chunks(
                self.dataset_path,
                splitter,
                max_files=self.file_limit,
                on_file=_on_file,
            )
        console.log(
            f"Chunking complete. Files processed: {processed_files}. Chunks: {len(chunk_records)}."
        )
        documents = [record["text"] for record in chunk_records]

        console.log("Generating embeddings for chunks...")
        embed_fn, model_name, run_platform = _create_embedding_runner(
            self.use_local, self.configured_model_name, self.platform
        )
        embeddings, total_tokens, embedding_size = embed_fn(documents)
        if embedding_size == 0 and embeddings:
            embedding_size = len(embeddings[0])
        console.log("Embeddings generated. Resolving database status...")

        db_config = _resolve_db_config()

        db_status = "unknown"
        adapter = db_config.get("adapter", "chromadb").lower()
        if adapter == "chromadb":
            try:
                _get_chroma_collection(self.table_name, db_config)
                db_status = "connected"
            except Exception as exc:  # pragma: no cover - runtime guard
                db_status = f"unreachable: {exc}"
        elif adapter in {"postgres", "postgresql"}:
            try:
                _ensure_valid_table_name(self.table_name)
                conn = _connect_postgres(db_config)
                try:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT to_regclass(%s)", (self.table_name,))
                            exists = cur.fetchone()[0]
                            db_status = "connected" if exists else "missing table"
                finally:
                    conn.close()
            except Exception as exc:  # pragma: no cover - runtime guard
                db_status = f"unreachable: {exc}"
        else:
            db_status = f"unsupported adapter: {adapter}"

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
            row_count=len(documents),
            total_tokens=total_tokens,
            db_adapter=db_config["adapter"],
            db_host=db_config["host"],
            db_port=db_config["port"],
            db_status=db_status,
            db_username=db_username,
            db_password_status=db_password_status,
            table_name=self.table_name,
            embedding_platform=run_platform,
        )

        if self.show_sample:
            console.log("Rendering sample vector record...")
            sample_chunk = chunk_records[0]
            sample_embedding = embeddings[0]
            embedding_preview = [round(float(value), 6) for value in sample_embedding[:5]]
            sample_record = {
                "id": str(uuid.uuid4()),
                "embedding": embedding_preview,
                "content": sample_chunk["text"],
                "metadata": {
                    "chunk_id": sample_chunk["id"],
                    "chunk_index": sample_chunk["global_index"],
                    "embedding_dimensions": len(sample_embedding),
                    "table_name": self.table_name,
                    "model_name": model_name,
                    "platform": run_platform,
                },
                "model_name": model_name,
                "platform": run_platform,
                "source": sample_chunk["source"],
                "table_name": self.table_name,
            }
            console.print("\n[bold cyan]Sample Vector Record[/bold cyan]")
            console.print_json(data=sample_record, indent=2, ensure_ascii=False)
