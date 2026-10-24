"""Offline SAFE preparation in the original radar grid, never affine AOI cropping."""
import math
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import zipfile

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.commands import NullWorkflowReporter, WorkflowEvent
from mlx.core.exceptions import MLXUserError
from .artifacts import inspect_archive, safe_component
from .calibration import SigmaCalibration
from .models import PrepareResult
from .requests import utc_date


def measurement_members(members, polarization):
    matches = [name for name in members if f"-{polarization.lower()}-" in PurePosixPath(name).name.lower()]
    if not matches:
        raise MLXUserError(f"SAFE archive has no {polarization} measurement. Select an available polarization.")
    return matches


def annotation_metadata(archive, measurement):
    from defusedxml.ElementTree import fromstring
    member = str(PurePosixPath(measurement).parent.parent / "annotation" / (PurePosixPath(measurement).stem + ".xml"))
    try:
        if archive.getinfo(member).file_size > 64 * 1024 * 1024:
            raise MLXUserError("SAFE annotation exceeds the supported XML size.")
        root = fromstring(archive.read(member))
        header = root.find("{*}adsHeader")
        if header is None:
            raise ValueError("Missing adsHeader")
        result = {key: header.findtext("{*}" + key) for key in ("missionId", "productType", "startTime", "polarisation", "mode")}
        if result["productType"] != "GRD" or result["missionId"] not in ("S1A", "S1B", "S1C", "S1D"):
            raise MLXUserError("Preparation supports Sentinel-1 GRD measurements only.")
        if not result["startTime"]:
            raise ValueError("Missing acquisition time")
        result["startTime"] = utc_date(result["startTime"]).isoformat()
        return result
    except (KeyError, ValueError) as exc:
        raise MLXUserError("SAFE measurement requires its matching annotation XML with acquisition metadata.") from exc


class SafeRasterPreparer:
    def prepare(self, archive_path, request, destination):
        try:
            import rasterio
        except ImportError as exc:
            raise MLXUserError("SAFE preparation requires the SAR extra: pip install -e '.[sar]'") from exc
        members = measurement_members(inspect_archive(archive_path), request.polarization)
        scene_id = safe_component(PurePosixPath(members[0]).parts[0].removesuffix(".SAFE"))
        if (Path(request.output).expanduser() / scene_id).exists():
            raise MLXUserError(f"Prepared scene already exists: {scene_id}. Choose a new output directory or remove the old prepared scene.")
        records = []
        with zipfile.ZipFile(archive_path) as archive:
            for member in members:
                source = annotation_metadata(archive, member)
                if source["polarisation"] != request.polarization:
                    raise MLXUserError("SAFE filename and annotation polarization disagree.")
                calibration = None
                if request.radiometry != "dn":
                    calibration_name = str(PurePosixPath(member).parent.parent / "annotation" / "calibration" / ("calibration-" + PurePosixPath(member).stem + ".xml"))
                    try:
                        if archive.getinfo(calibration_name).file_size > 64 * 1024 * 1024:
                            raise MLXUserError("SAFE calibration XML exceeds the supported size.")
                        calibration = SigmaCalibration(archive.read(calibration_name))
                    except KeyError as exc:
                        raise MLXUserError("SAFE sigmaNought calibration XML is missing.") from exc
                with rasterio.open(f"zip://{archive_path.resolve()}!{member}") as raster:
                    if raster.count != 1 or raster.dtypes[0] != "uint16":
                        raise MLXUserError("Expected a single-band uint16 Sentinel-1 GRD amplitude measurement.")
                    tile = request.tile_size or max(raster.height, raster.width)
                    for row in range(0, raster.height, tile):
                        for col in range(0, raster.width, tile):
                            height, width = min(tile, raster.height-row), min(tile, raster.width-col)
                            name = f"{PurePosixPath(member).stem}-r{row}-c{col}"
                            record = self._write_tile(raster, destination, name, row, col, height, width, request, calibration)
                            records.append({**record, "source_scene_id": scene_id, "source_measurement": member,
                                            "acquisition_time": source["startTime"], "platform": source["missionId"],
                                            "product_type": source["productType"], "polarization": source["polarisation"]})
        manifest = {"schema": "mlx-sar-prepared-v1", "scene_id": scene_id,
                    "source_archive": archive_path.name, "source_sha256": sha256_file(archive_path),
                    "geolocation": "native radar grid; annotation geolocation not used for cropping",
                    "geographic_cropping": False, "terrain_corrected": False, "thermal_noise_removed": False,
                    "tiles": records}
        write_json_atomic(destination / "manifest.json", manifest)
        return scene_id

    @staticmethod
    def _write_tile(raster, destination, name, row, col, height, width, request, calibration):
        import numpy as np
        from PIL import Image
        from rasterio.windows import Window
        arrays, previews = destination / "tiles", destination / "previews"
        arrays.mkdir(exist_ok=True)
        previews.mkdir(exist_ok=True)
        array_path, preview_path = arrays / f"{name}.npy", previews / f"{name}.png"
        data = np.lib.format.open_memmap(array_path, mode="w+", dtype="uint16" if calibration is None else "float32", shape=(height, width))
        try:
            # Bound RAM even when the user requests one full-resolution scene array.
            for offset in range(0, height, 128):
                count = min(128, height-offset)
                dn = raster.read(1, window=Window(col, row+offset, width, count))
                data[offset:offset+count] = dn if calibration is None else calibration.apply(dn, row+offset, col, decibels=request.radiometry == "sigma0-db")
            data.flush()
            stride = max(1, math.ceil(max(height, width)/1024))
            thumbnail = np.array(data[::stride, ::stride], dtype=np.float64)
        finally:
            # Release the mapping before staging cleanup, including on Windows.
            del data
        valid = np.isfinite(thumbnail) & ((thumbnail > 0) if calibration is None else True)
        preview = np.zeros(thumbnail.shape, dtype=np.uint8)
        limits = [0., 0.]
        if valid.any():
            limits = np.percentile(thumbnail[valid], [2, 98]).tolist()
            if limits[1] > limits[0]:
                preview[valid] = (np.clip((thumbnail[valid]-limits[0])/(limits[1]-limits[0]), 0, 1)*255).astype(np.uint8)
        Image.fromarray(preview).save(preview_path)
        return {"array": str(array_path.relative_to(destination)), "preview": str(preview_path.relative_to(destination)),
                "sha256": sha256_file(array_path), "preview_sha256": sha256_file(preview_path),
                "window": {"row": row, "col": col, "height": height, "width": width},
                "dimensions": [height, width], "nodata": 0 if calibration is None else "NaN",
                "crs": None, "geotransform": None, "geographic_bounds": None,
                "preprocessing": {"radiometry": request.radiometry, "lut": "sigmaNought" if calibration else None,
                    "formula": "DN" if calibration is None else ("10*log10(DN^2/LUT^2)" if request.radiometry == "sigma0-db" else "DN^2/LUT^2"),
                    "lut_interpolation": "bilinear" if calibration else None,
                    "preview": {"method": "visualization-only percentile stretch", "percentiles": [2, 98], "limits": limits, "sampling_stride": stride}}}


class PrepareSarImagery:
    def __init__(self, request, preparer=None, reporter=None):
        self.request = request
        self.preparer = preparer or SafeRasterPreparer()
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self):
        request = self.request
        if request.bbox is not None:
            raise MLXUserError("Geographic cropping of native SAFE radar-grid TIFFs is unavailable. Process with ESA SNAP Range-Doppler Terrain Correction first; omit AOI coordinates for radar-grid tiling. Installing Rasterio alone does not enable geographic correction.")
        root = Path(request.input).expanduser()
        archives = [root] if root.is_file() else sorted(root.glob("*.zip"))
        if not archives and (root / "raw").is_dir():
            archives = sorted((root / "raw").glob("*.zip"))
        if not archives:
            raise MLXUserError("No SAFE ZIP archives found. Supply an archive, raw directory, or SAR output root with --input.")
        output = Path(request.output).expanduser()
        manifests = []
        try:
            output.mkdir(parents=True, exist_ok=True)
            for index, archive in enumerate(archives):
                with TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
                    staging = Path(temporary) / "scene"
                    staging.mkdir()
                    scene_id = self.preparer.prepare(archive, request, staging)
                    destination = output / safe_component(scene_id)
                    if destination.exists():
                        raise MLXUserError(f"Prepared scene already exists: {destination}. Choose a new output directory or remove the old prepared scene.")
                    staging.replace(destination)
                    manifests.append(str(destination / "manifest.json"))
                self.reporter.emit(WorkflowEvent("info", f"Prepared {scene_id}", index+1, len(archives)))
        except MLXUserError:
            raise
        except Exception as exc:
            raise MLXUserError("SAFE preparation failed. Check archive annotations, raster readability, and available disk space.") from exc
        return PrepareResult(tuple(manifests))
