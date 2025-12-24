from typing import Any, Dict

ModuleConfig = Dict[str, Any]

from . import local

def run(config: ModuleConfig) -> None:
    platform = (config.get("platform"))

    if platform == "local":
        local.run_agent(config)
    else:
        raise ValueError(
            f"Unsupported agents platform '{platform}'"
        )
