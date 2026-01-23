import os
import re
from typing import Dict

import typer

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")

def _resolve_db_config() -> Dict[str, str]:
    adapter = os.environ.get("DB_ADAPTER", "postgresql")
    normalized_adapter = adapter.lower()
    config: Dict[str, str] = {
        "adapter": adapter,
        "username": os.environ.get("DB_USERNAME", ""),
        "password": os.environ.get("DB_PASSWORD", ""),
    }
    if normalized_adapter in {"chromadb", "postgres", "postgresql"}:
        config["host"] = os.environ.get("DB_HOST", "not set")
        if normalized_adapter in {"postgres", "postgresql"}:
            config["port"] = os.environ.get("DB_PORT", "5432")
        else:
            config["port"] = os.environ.get("DB_PORT", "not set")
        if normalized_adapter in {"postgres", "postgresql"}:
            config["database"] = os.environ.get("DB_NAME", "not set")
    else:
        config["host"] = "n/a"
        config["port"] = "n/a"
    return config


def _ensure_valid_table_name(table_name: str) -> None:
    if not _TABLE_NAME_RE.match(table_name):
        raise typer.BadParameter(
            "Invalid --table-name for postgres. Use only letters, numbers, underscores, and an optional schema."
        )


def _connect_postgres(db_config: Dict[str, str]):
    try:
        import psycopg  # type: ignore
    except ImportError:
        try:
            import psycopg2 as psycopg  # type: ignore
        except ImportError as exc:
            raise typer.BadParameter(
                "psycopg or psycopg2 is required for postgres operations. Install one to proceed."
            ) from exc

    host = db_config.get("host")
    port_raw = db_config.get("port")
    database = db_config.get("database")
    if host in {"not set", "", None} or port_raw in {"not set", "", None}:
        raise typer.BadParameter("DB_HOST and DB_PORT must be set for postgres operations.")
    if database in {"not set", "", None}:
        raise typer.BadParameter("DB_NAME must be set for postgres operations.")

    try:
        port = int(port_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise typer.BadParameter("DB_PORT must be an integer for postgres operations.")

    conn_kwargs = {
        "host": host,
        "port": port,
        "dbname": database,
    }
    if db_config.get("username"):
        conn_kwargs["user"] = db_config.get("username")
    if db_config.get("password"):
        conn_kwargs["password"] = db_config.get("password")
    return psycopg.connect(**conn_kwargs)
