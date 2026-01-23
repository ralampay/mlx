from typing import List

import typer

from mlx.definitions import ModuleConfig
from mlx.modules.rag.helpers import _get_chroma_collection, console
from mlx.modules.rag.utils import _connect_postgres, _ensure_valid_table_name, _resolve_db_config


class RagDeleteAll:
    def __init__(self, config: ModuleConfig) -> None:
        self.table_name = config.get("table_name")
        if not self.table_name:
            raise typer.BadParameter("--table-name is required for delete-all.")

    def execute(self) -> None:
        db_config = _resolve_db_config()
        adapter = db_config.get("adapter", "chromadb").lower()
        if adapter == "chromadb":
            collection, db_host, db_port = _get_chroma_collection(self.table_name, db_config)

            total_records = collection.count()
            if total_records == 0:
                console.print(f"[yellow]Collection '{self.table_name}' is already empty.[/]")
                return

            deleted = 0
            batch_size = 1000
            while True:
                batch = collection.get(limit=batch_size, include=[])
                ids: List[str] = batch.get("ids") or []
                if not ids:
                    break
                collection.delete(ids=ids)
                deleted += len(ids)

            console.print(
                f"[green]Deleted {deleted} record(s) from collection '{self.table_name}' (host={db_host}, port={db_port}).[/]"
            )
            return

        if adapter in {"postgres", "postgresql"}:
            _ensure_valid_table_name(self.table_name)
            conn = _connect_postgres(db_config)
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT to_regclass(%s)", (self.table_name,))
                        exists = cur.fetchone()[0]
                        if not exists:
                            raise typer.BadParameter(f"Table '{self.table_name}' does not exist.")
                        cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                        total_records = cur.fetchone()[0]
                        if total_records == 0:
                            console.print(f"[yellow]Table '{self.table_name}' is already empty.[/]")
                            return
                        cur.execute(f"TRUNCATE TABLE {self.table_name}")
            finally:
                conn.close()

            console.print(
                "[green]Deleted "
                f"{total_records} record(s) from table '{self.table_name}' "
                f"(host={db_config.get('host', 'unknown')}, port={db_config.get('port', 'unknown')}, "
                f"database={db_config.get('database', 'unknown')}).[/]"
            )
            return

        raise typer.BadParameter("Unsupported DB_ADAPTER for delete-all. Use 'chromadb' or 'postgres'.")
