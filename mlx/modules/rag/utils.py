import os
from typing import Dict

def _resolve_db_config() -> Dict[str, str]:
    adapter = os.environ.get("DB_ADAPTER", "chromadb")
    normalized_adapter = adapter.lower()
    config: Dict[str, str] = {
        "adapter": adapter,
        "username": os.environ.get("DB_USERNAME", ""),
        "password": os.environ.get("DB_PASSWORD", ""),
    }
    if normalized_adapter in {"chromadb", "postgres", "postgresql"}:
        config["host"] = os.environ.get("DB_HOST", "not set")
        config["port"] = os.environ.get("DB_PORT", "not set")
        if normalized_adapter in {"postgres", "postgresql"}:
            config["database"] = os.environ.get("DB_NAME", "not set")
    else:
        config["host"] = "n/a"
        config["port"] = "n/a"
    return config
