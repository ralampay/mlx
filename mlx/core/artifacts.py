from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mlx.core.exceptions import MLXUserError


def json_safe(value: Any) -> Any:
    """Convert common MLX result values into deterministic JSON-compatible data."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return json_safe(to_dict())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (RuntimeError, TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_safe(tolist())
    return str(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except OSError as exc:
        raise MLXUserError(f"Unable to write JSON artifact '{target}': {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    target = Path(path)
    materialized = list(rows)
    columns = list(fieldnames or (materialized[0].keys() if materialized else ()))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {name: _csv_cell(row.get(name)) for name in columns}
                for row in materialized
            )
    except OSError as exc:
        raise MLXUserError(f"Unable to write CSV artifact '{target}': {exc}") from exc
    return target


def _csv_cell(value: Any) -> Any:
    normalized = json_safe(value)
    if isinstance(normalized, (dict, list)):
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return normalized


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    except OSError as exc:
        raise MLXUserError(f"Unable to write PyTorch artifact '{target}': {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = ["atomic_torch_save", "json_safe", "sha256_file", "write_csv", "write_json_atomic"]
