from rich.table import Table
from mlx.core.ui import console
from .models import SearchResult, DownloadResult


class RichSarReporter:
    def emit(self, event):
        suffix = f" ({event.current}/{event.total})" if event.total else ""
        console.print(event.message + suffix, markup=False)


def present(result):
    if isinstance(result, (SearchResult, DownloadResult)):
        table = Table(title="SAR catalogue scenes (full archives, not cropped images)")
        for name in ("Scene", "UTC acquisition", "Platform", "Level", "Beam", "Polarization", "Direction", "Footprint", "Bytes (approx.)", "URL", "AOI coverage"):
            table.add_column(name)
        for s in result.scenes:
            table.add_row(s.scene_id, s.acquisition_time, s.platform, s.processing_level, s.beam_mode,
                          s.polarization, s.flight_direction, str(s.footprint), str(s.size_bytes), s.url or "",
                          "unknown" if s.aoi_coverage is None else f"{s.aoi_coverage:.1%}")
        console.print(table)
        if isinstance(result, SearchResult):
            console.print(result.selection_scope)
        elif result.dry_run:
            console.print(f"Dry run: {len(result.scenes)} archive(s); no downloads performed.")
    else:
        for manifest in result.manifests:
            console.print(manifest, markup=False)
