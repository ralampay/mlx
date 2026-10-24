"""CLI/environment precedence resolved at the composition boundary."""
from pathlib import Path
from mlx.core.exceptions import MLXUserError
from .requests import BoundingBox, NetworkOptions, SearchRequest


def bbox_from_config(config):
    values = [config.get(k) for k in ("min_lon", "min_lat", "max_lon", "max_lat")]
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise MLXUserError("Supply all four AOI coordinates.")
    return BoundingBox(*values)


def search_from_config(config):
    return SearchRequest(bbox_from_config(config), config.get("start_date"), config.get("end_date"),
                         config.get("max_results", 20), config.get("polarization"),
                         config.get("flight_direction"), config.get("beam_mode", "IW"), config.get("scene_id"))


def settings(config, environment):
    def value(flag, env, default):
        return config.get(flag) if config.get(flag) is not None else environment.get(env) or default
    try:
        network = NetworkOptions(int(value("download_processes", "MLX_SAR_DOWNLOAD_PROCESSES", 1)),
                                 float(value("request_timeout", "MLX_SAR_REQUEST_TIMEOUT", 60)),
                                 int(value("max_retries", "MLX_SAR_MAX_RETRIES", 3)))
    except (ValueError, TypeError) as exc:
        raise MLXUserError("SAR processes/retries must be integers and timeout must be numeric; check CLI and MLX_SAR_* variables.") from exc
    output = Path(value("output_path", "MLX_SAR_OUTPUT_DIR", "~/Desktop/datasets/sar")).expanduser()
    cache = Path(value("sar_cache_dir", "MLX_SAR_CACHE_DIR", "~/.cache/mlx/sar")).expanduser()
    return output, cache, network
