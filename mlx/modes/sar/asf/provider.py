"""Official asf_search integration; no SDK objects escape this boundary."""
from mlx.core.exceptions import MLXUserError
from ..artifacts import safe_component
from ..models import Scene
from ..requests import NetworkOptions, utc_date
from .authentication import EarthdataCredentials, authenticate
from .client import create_session, retry_call, validate_download_url


def create_provider(*, environment, network):
    return AsfProvider(
        credentials=EarthdataCredentials(environment.get("EARTHDATA_TOKEN"),
            environment.get("EARTHDATA_USERNAME"), environment.get("EARTHDATA_PASSWORD")),
        network=network,
    )


def _coverage(geometry, bbox):
    if bbox is None or not geometry:
        return None
    try:
        from shapely.geometry import shape, box
        from shapely import transform
        import numpy as np
        from pyproj import Transformer
    except ImportError as exc:
        raise MLXUserError("SAR coverage requires Shapely and PyProj. Install with: pip install -e '.[sar]'") from exc
    footprint = shape(geometry)
    # Dateline footprints need explicit splitting; don't report misleading coverage.
    if not footprint.is_valid or footprint.bounds[2] - footprint.bounds[0] > 180:
        return None
    aoi = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    intersection = footprint.intersection(aoi)
    # Equal-area projection avoids degree-square area comparisons.
    projection = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    def project(coordinates):
        return np.column_stack(projection.transform(coordinates[:, 0], coordinates[:, 1]))
    area = transform(aoi, project).area
    return min(1.0, max(0.0, transform(intersection, project).area / area)) if area else None


def scene_from_product(product, bbox):
    metadata = product.geojson()
    p = metadata["properties"]
    # ASF bytes is exact when supplied; legacy sizeMB is an approximate catalogue size.
    size = p.get("bytes")
    if size is None and p.get("sizeMB") is not None:
        size = round(float(p["sizeMB"]) * 1_000_000)
    return Scene(
        scene_id=p.get("fileID") or p["sceneName"],
        acquisition_time=utc_date(p.get("startTime") or p["sceneDate"]).isoformat(),
        platform=p.get("platform", ""), processing_level=p.get("processingLevel", ""),
        beam_mode=p.get("beamModeType", ""), polarization=p.get("polarization", ""),
        flight_direction=p.get("flightDirection", ""), footprint=metadata.get("geometry") or {},
        filename=p.get("fileName", ""), url=p.get("url"), size_bytes=int(size) if size is not None else None,
        metadata=metadata, aoi_coverage=_coverage(metadata.get("geometry"), bbox),
        size_is_exact=p.get("bytes") is not None,
    )


class AsfProvider:
    def __init__(self, credentials=None, network=None, *, library=None, session_factory=create_session):
        self.credentials = credentials or EarthdataCredentials()
        self.network = network or NetworkOptions()
        self._library = library
        self.session_factory = session_factory

    @property
    def library(self):
        if self._library is None:
            try:
                import asf_search
            except ImportError as exc:
                raise MLXUserError("SAR requires optional dependencies. Install with: pip install -e '.[sar]'") from exc
            self._library = asf_search
        return self._library

    def search(self, request):
        asf = self.library
        with self.session_factory(asf, self.network, public=True) as session:
            opts = asf.ASFSearchOptions(session=session)
            if request.scene_id:
                operation = lambda: asf.product_search([request.scene_id], opts=opts)
            else:
                filters = dict(
                    platform=asf.PLATFORM.SENTINEL1, processingLevel=asf.PRODUCT_TYPE.GRD_HD,
                    beamMode=request.beam_mode, intersectsWith=request.bbox.wkt(),
                    start=request.start_date, end=request.end_date, maxResults=request.max_results, opts=opts,
                )
                if request.polarization:
                    # A request for VV includes VV+VH products; ASF filters exact channel combinations.
                    combinations = {"VV": ["VV", "VV+VH"], "VH": ["VV+VH"],
                                    "HH": ["HH", "HH+HV"], "HV": ["HH+HV"]}
                    filters["polarization"] = combinations.get(request.polarization, request.polarization)
                if request.flight_direction:
                    filters["flightDirection"] = request.flight_direction
                operation = lambda: asf.geo_search(**filters)
            results = retry_call(operation, self.network, description="ASF catalogue search")
            try:
                return [scene_from_product(product, request.bbox) for product in results][:request.max_results]
            except (KeyError, ValueError, TypeError) as exc:
                raise MLXUserError("ASF returned incomplete scene metadata; inspect the catalogue product or retry.") from exc

    def download(self, scene, directory):
        safe_component(scene.filename)
        validate_download_url(scene.url)
        asf = self.library
        target = directory / scene.filename
        with self.session_factory(asf, self.network) as session:
            authenticate(session, self.credentials)
            def operation():
                target.unlink(missing_ok=True)  # restart partial attempts; no byte-range promise
                asf.download_url(scene.url, path=str(directory), filename=scene.filename, session=session)
                return target
            return retry_call(operation, self.network, description="ASF archive download")
