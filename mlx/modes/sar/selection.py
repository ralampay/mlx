"""Deterministic ranking of the inspected catalogue subset."""
from .requests import utc_date


def rank_scenes(scenes, request):
    selected = []
    for scene in scenes:
        if scene.processing_level != "GRD_HD" or not scene.platform.upper().replace("-", "").startswith("SENTINEL1"):
            continue
        if scene.beam_mode != request.beam_mode:
            continue
        if request.polarization and not set(request.polarization.split("+")).issubset(scene.polarization.split("+")):
            continue
        if request.flight_direction and scene.flight_direction != request.flight_direction:
            continue
        selected.append(scene)
    return tuple(sorted(selected, key=lambda s: (
        -utc_date(s.acquisition_time).timestamp(),
        -(s.aoi_coverage if s.aoi_coverage is not None else -1), s.scene_id,
    )))
