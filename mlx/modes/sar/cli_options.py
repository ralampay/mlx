"""SAR-only options; provider, input, output and dry-run reuse existing flags."""
from mlx.core.cli_options import CliOption

OPTIONS = tuple(CliOption((f"--{name}",), settings) for name, settings in (
    ("min-lon", {"type": float}), ("min-lat", {"type": float}),
    ("max-lon", {"type": float}), ("max-lat", {"type": float}),
    ("start-date", {"help": "Inclusive UTC ISO date/time lower bound."}),
    ("end-date", {"help": "Inclusive UTC ISO date/time upper bound; dates mean midnight."}),
    ("max-results", {"type": int, "default": 20, "help": "Maximum catalogue candidates inspected, not a global best-scene guarantee."}),
    ("polarization", {"type": str.upper}),
    ("flight-direction", {"type": str.upper, "choices": ("ASCENDING", "DESCENDING")}),
    ("beam-mode", {"type": str.upper, "default": "IW"}),
    ("scene-id", {"help": "Exact ASF product identifier, without an AOI search."}),
    ("all-scenes", {"action": "store_true", "help": "Explicitly download every matching returned SAR scene."}),
    ("sar-cache-dir", {"help": "Verified SAR cache; overrides MLX_SAR_CACHE_DIR."}),
    ("download-processes", {"type": int, "help": "Concurrent scene downloads; default 1."}),
    ("request-timeout", {"type": float, "help": "Network request timeout in seconds; default 60."}),
    ("max-retries", {"type": int, "help": "Transient network retries after the initial attempt; default 3."}),
    ("tile-size", {"type": int, "help": "SAR radar-grid tile side in pixels; edge tiles retain their size."}),
    ("radiometry", {"choices": ("dn", "sigma0", "sigma0-db"), "default": "dn"}),
))
