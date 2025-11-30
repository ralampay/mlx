from __future__ import annotations

from typing import Any, Dict

from . import llama, openai

ModuleConfig = Dict[str, Any]


def run_chat(config: ModuleConfig) -> None:
    platform = (config.get("platform") or "openai").lower()

    if platform == "openai":
        openai.run_chat(config)
    elif platform in {"local", "llama"}:
        llama.run_chat(config)
    else:
        raise ValueError(
            f"Unsupported chat platform '{platform}'. Choose 'openai', 'local', or 'llama'."
        )
