from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    scene_id: str
    acquisition_time: str
    platform: str
    processing_level: str
    beam_mode: str
    polarization: str
    flight_direction: str
    footprint: dict
    filename: str
    url: str | None
    size_bytes: int | None
    metadata: dict
    aoi_coverage: float | None = None
    size_is_exact: bool = False


@dataclass(frozen=True)
class SearchResult:
    scenes: tuple[Scene, ...]
    inspected_count: int
    max_results: int
    selection_scope: str = "Ranked only within returned catalogue results; intersection does not imply complete AOI coverage."
    query: dict | None = None


@dataclass(frozen=True)
class DownloadResult:
    scenes: tuple[Scene, ...]
    downloads: tuple[dict, ...]
    dry_run: bool


@dataclass(frozen=True)
class PrepareResult:
    manifests: tuple[str, ...]
    geographic_cropping: str = "unavailable for native SAFE radar-grid measurements"
