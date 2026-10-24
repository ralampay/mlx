"""SAR composition and dispatch; no provider or raster work at import time."""
import os
from pathlib import Path
from mlx.core.commands import NullWorkflowReporter
from mlx.core.exceptions import MLXUserError
from .configuration import bbox_from_config, search_from_config, settings
from .requests import DownloadRequest, PrepareRequest


def run_sar(config, *, environment=None, provider_registry=None):
    from .artifacts import ArchiveCache
    from .commands import SearchSarScenes, DownloadSarScenes
    from .preparation import PrepareSarImagery
    from .providers import ProviderRegistry
    from .presentation import RichSarReporter, present

    environment = os.environ if environment is None else environment
    action = config.get("action") or "search"
    if action not in ("search", "download", "prepare") or config.get("platform", "local") != "local":
        raise MLXUserError("SAR supports local search, download and prepare actions.")
    output, cache, network = settings(config, environment)
    reporter = NullWorkflowReporter() if config.get("output_format") == "json" else RichSarReporter()
    if action == "prepare":
        if not config.get("input_path"):
            raise MLXUserError("SAR prepare requires --input pointing to SAFE ZIP archives.")
        if config.get("output_path") is None:
            output = output / "prepared"
        command = PrepareSarImagery(PrepareRequest(Path(config["input_path"]).expanduser(), output,
            config.get("polarization") or "VV", config.get("tile_size"), config.get("radiometry", "dn"), bbox_from_config(config)), reporter=reporter)
    else:
        request = search_from_config(config)
        # Shared parser retains the historic detection default. Only explicit CLI values override ASF.
        explicit = config.get("_explicit_options")
        provider_name = config.get("provider") if explicit is None or "provider" in explicit else None
        provider = (provider_registry or ProviderRegistry()).create(provider_name or "asf",
            environment=environment, network=network)
        command = (SearchSarScenes(request, provider, output, reporter) if action == "search" else
                   DownloadSarScenes(DownloadRequest(request, output, cache, config.get("all_scenes", False),
                       config.get("dry_run", False), network.processes), provider, ArchiveCache(cache), reporter))
    result = command.execute()
    if config.get("output_format") != "json":
        present(result)
    return result
