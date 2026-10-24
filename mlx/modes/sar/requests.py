"""Validated, provider-neutral SAR inputs."""
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path

from mlx.core.exceptions import MLXUserError


def utc_date(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, AttributeError) as exc:
        raise MLXUserError("Dates must be ISO-8601 dates or timestamps, for example 2026-10-01.") from exc


@dataclass(frozen=True)
class BoundingBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self):
        for name, limit in (("min_lon", 180), ("max_lon", 180), ("min_lat", 90), ("max_lat", 90)):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not -limit <= value <= limit:
                raise MLXUserError(f"{name} must be finite and between {-limit} and {limit}.")
        if self.min_lat >= self.max_lat:
            raise MLXUserError("Minimum latitude must be less than maximum latitude.")
        if self.min_lon >= self.max_lon:
            raise MLXUserError("Minimum longitude must be less than maximum longitude. Split antimeridian-crossing AOIs into two searches.")

    def wkt(self) -> str:
        a, b, c, d = self.min_lon, self.min_lat, self.max_lon, self.max_lat
        return f"POLYGON(({a} {b},{c} {b},{c} {d},{a} {d},{a} {b}))"


@dataclass(frozen=True)
class SearchRequest:
    bbox: BoundingBox | None = None
    start_date: str | None = None
    end_date: str | None = None
    max_results: int = 20
    polarization: str | None = None
    flight_direction: str | None = None
    beam_mode: str = "IW"
    scene_id: str | None = None

    def __post_init__(self):
        if isinstance(self.max_results, bool) or not isinstance(self.max_results, int) or self.max_results < 1:
            raise MLXUserError("--max-results must be a positive integer.")
        if not self.scene_id and (self.bbox is None or not self.start_date or not self.end_date):
            raise MLXUserError("Supply all four AOI coordinates, --start-date and --end-date, or --scene-id.")
        if self.scene_id and (self.bbox or self.start_date or self.end_date):
            raise MLXUserError("Use --scene-id alone, or use an AOI/date search; do not combine them.")
        if self.start_date and self.end_date and utc_date(self.start_date) > utc_date(self.end_date):
            raise MLXUserError("Start date must not exceed end date.")
        if self.polarization not in (None, "VV", "VH", "HH", "HV", "VV+VH", "HH+HV"):
            raise MLXUserError("Unsupported polarization; use VV, VH, HH, HV, VV+VH or HH+HV.")
        if self.flight_direction not in (None, "ASCENDING", "DESCENDING"):
            raise MLXUserError("Flight direction must be ASCENDING or DESCENDING.")
        if self.beam_mode not in ("IW", "EW", "SM"):
            raise MLXUserError("Beam mode must be IW, EW or SM.")


@dataclass(frozen=True)
class NetworkOptions:
    processes: int = 1
    timeout: float = 60
    max_retries: int = 3

    def __post_init__(self):
        if not isinstance(self.processes, int) or self.processes < 1:
            raise MLXUserError("Download processes must be a positive integer.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise MLXUserError("Request timeout must be positive and finite.")
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise MLXUserError("Max retries must be a non-negative integer.")


@dataclass(frozen=True)
class DownloadRequest:
    search: SearchRequest
    output: Path
    cache: Path
    all_scenes: bool = False
    dry_run: bool = False
    processes: int = 1


@dataclass(frozen=True)
class PrepareRequest:
    input: Path
    output: Path
    polarization: str = "VV"
    tile_size: int | None = None
    radiometry: str = "dn"
    bbox: BoundingBox | None = None

    def __post_init__(self):
        if self.polarization not in ("VV", "VH", "HH", "HV"):
            raise MLXUserError("Preparation requires one polarization: VV, VH, HH or HV.")
        if self.tile_size is not None and (not isinstance(self.tile_size, int) or self.tile_size < 1):
            raise MLXUserError("--tile-size must be a positive integer.")
        if self.radiometry not in ("dn", "sigma0", "sigma0-db"):
            raise MLXUserError("Radiometry must be dn, sigma0 or sigma0-db.")
