from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.core.artifacts import json_safe
from mlx.modes.sar.requests import BoundingBox, SearchRequest, PrepareRequest, DownloadRequest, NetworkOptions
from mlx.modes.sar.models import Scene
from mlx.modes.sar.commands import SearchSarScenes, DownloadSarScenes
from mlx.modes.sar.artifacts import ArchiveCache, inspect_archive
from mlx.modes.sar.selection import rank_scenes
from mlx.modes.sar.asf.authentication import EarthdataCredentials, authenticate
from mlx.modes.sar.preparation import PrepareSarImagery, measurement_members


def scene(**kwargs):
    base = Scene("S1_TEST-GRD_HD", "2020-01-01T00:00:00Z", "Sentinel-1A", "GRD_HD", "IW", "VV+VH",
                 "ASCENDING", {"type": "Polygon", "coordinates": [[[0,0],[2,0],[2,2],[0,2],[0,0]]]},
                 "S1_TEST.zip", "https://datapool.asf.alaska.edu/GRD_HD/SA/S1_TEST.zip", None,
                 {"properties": {"sceneName": "S1_TEST"}})
    return replace(base, **kwargs)


def request(**kwargs):
    return SearchRequest(**{"bbox": BoundingBox(0,0,1,1), "start_date": "2020-01-01", "end_date": "2020-02-01", **kwargs})


def safe_zip(path, *, polarization="VV", valid_raster=True):
    member = f"S1_TEST.SAFE/measurement/s1a-iw-grd-{polarization.lower()}-test-001.tiff"
    raster_bytes = b"synthetic"
    if valid_raster:
        np = pytest.importorskip("numpy")
        rio = pytest.importorskip("rasterio")
        from rasterio.io import MemoryFile
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=4, height=4, count=1, dtype="uint16") as raster:
                raster.write(np.arange(16, dtype="uint16").reshape(4,4), 1)
            raster_bytes = memory.read()
    annotation = f"""<product><adsHeader><missionId>S1A</missionId><productType>GRD</productType>
    <startTime>2020-01-01T00:00:00Z</startTime><polarisation>{polarization}</polarisation><mode>IW</mode>
    </adsHeader><geolocationGrid><geolocationGridPointList>
    <geolocationGridPoint><line>0</line><pixel>0</pixel><latitude>10</latitude><longitude>20</longitude></geolocationGridPoint>
    <geolocationGridPoint><line>3</line><pixel>3</pixel><latitude>11</latitude><longitude>21</longitude></geolocationGridPoint>
    </geolocationGridPointList></geolocationGrid></product>"""
    calibration = """<calibration><calibrationVectorList>
    <calibrationVector><line>0</line><pixel>0 3</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
    <calibrationVector><line>3</line><pixel>0 3</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
    </calibrationVectorList></calibration>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, raster_bytes)
        archive.writestr("S1_TEST.SAFE/manifest.safe", "<manifest/>")
        stem = Path(member).stem
        archive.writestr(f"S1_TEST.SAFE/annotation/{stem}.xml", annotation)
        archive.writestr(f"S1_TEST.SAFE/annotation/calibration/calibration-{stem}.xml", calibration)
    return path


class FakeProvider:
    def __init__(self, scenes=None, fail=False):
        self.scenes = scenes if scenes is not None else [scene()]
        self.calls = 0
        self.fail = fail

    def search(self, request):
        return self.scenes

    def download(self, selected, directory):
        self.calls += 1
        path = directory / selected.filename
        if self.fail:
            path.write_bytes(b"partial")
            raise MLXUserError("Interrupted download")
        return safe_zip(path, valid_raster=False)


def test_wkt_closed_longitude_latitude():
    assert BoundingBox(1,2,3,4).wkt() == "POLYGON((1 2,3 2,3 4,1 4,1 2))"


@pytest.mark.parametrize("bounds", [(181,0,182,1), (0,-91,1,0), (0,0,1,91), (-181,0,1,1),
                                      (0,1,1,0), (2,0,1,1), (170,0,-170,1), (0,0,0,1), (0,0,1,float('nan'))])
def test_invalid_bounds(bounds):
    with pytest.raises(MLXUserError):
        BoundingBox(*bounds)


@pytest.mark.parametrize("values", [{"max_results": 0}, {"max_results": 1.5}, {"start_date": "2021-01-01"}, {"start_date": "bad"}])
def test_invalid_search(values):
    with pytest.raises(MLXUserError):
        request(**values)


def test_empty_catalogue_writes_json(tmp_path):
    result = SearchSarScenes(request(), FakeProvider([]), tmp_path).execute()
    assert result.scenes == ()
    assert json.loads((tmp_path / "metadata/scenes.json").read_text())["scenes"] == []


def test_filtering_and_deterministic_selection():
    candidates = [scene(scene_id="z"), scene(scene_id="a"), scene(scene_id="new", acquisition_time="2020-01-02T00:00:00Z"),
                  scene(scene_id="slc", processing_level="SLC"), scene(scene_id="wrong", beam_mode="EW")]
    assert [s.scene_id for s in rank_scenes(candidates, request())] == ["new", "a", "z"]
    assert rank_scenes(candidates[::-1], request()) == rank_scenes(candidates, request())
    assert rank_scenes(candidates, request(polarization="HH")) == ()
    assert rank_scenes([scene(aoi_coverage=.1, scene_id="a"), scene(aoi_coverage=.9, scene_id="b")], request())[0].scene_id == "b"


class AuthSession:
    def auth_with_token(self, token):
        self.called = ("token", token)
        return self
    def auth_with_creds(self, username, password):
        self.called = ("creds", username, password)
        return self


def test_token_precedence():
    s = authenticate(AuthSession(), EarthdataCredentials("token", "user", "password"))
    assert s.called == ("token", "token")
    assert "password" not in repr(EarthdataCredentials("token", "user", "password"))


def test_username_fallback():
    assert authenticate(AuthSession(), EarthdataCredentials(username="u", password="p")).called == ("creds", "u", "p")


def test_netrc_fallback_and_missing_credentials():
    assert authenticate(AuthSession(), EarthdataCredentials(), lambda *a, **k: ("u","p")).called == ("creds", "u", "p")
    with pytest.raises(MLXUserError, match="Protected downloads require"):
        authenticate(AuthSession(), EarthdataCredentials(), lambda *a, **k: None)


def test_auth_errors_do_not_leak_secrets(capsys):
    class Bad:
        def auth_with_token(self, token):
            raise RuntimeError("secret-token")
    with pytest.raises(MLXUserError) as exc:
        authenticate(Bad(), EarthdataCredentials(token="secret-token"))
    assert "secret-token" not in str(exc.value)
    assert "secret-token" not in str(capsys.readouterr())
    assert exc.value.__suppress_context__


def test_cache_hit_and_partial_recovery(tmp_path):
    cache = ArchiveCache(tmp_path / "cache")
    provider = FakeProvider(fail=True)
    with pytest.raises(MLXUserError):
        cache.obtain(scene(), provider, tmp_path / "output")
    assert not list((tmp_path / "cache").rglob("verified.json"))
    assert not list((tmp_path / "cache").rglob("*.zip"))
    provider.fail = False
    first = cache.obtain(scene(), provider, tmp_path / "output")
    second = cache.obtain(scene(), provider, tmp_path / "output")
    assert first["sha256"] == second["sha256"]
    assert not first["cache_hit"] and second["cache_hit"]
    assert provider.calls == 2
    assert first["source"]["metadata"] == scene().metadata


def test_corrupt_cache_is_never_reused(tmp_path):
    provider = FakeProvider()
    cache = ArchiveCache(tmp_path / "cache")
    first = cache.obtain(scene(), provider, tmp_path / "out")
    Path(first["path"]).unlink()  # don't modify a hardlinked output
    next((tmp_path / "cache").rglob("*.zip")).write_bytes(b"broken")
    assert not cache.obtain(scene(), provider, tmp_path / "out")["cache_hit"]
    assert provider.calls == 2


def test_archive_integrity_failure(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"partial zip")
    with pytest.raises(MLXUserError, match="validate SAFE ZIP"):
        inspect_archive(archive)


def test_measurement_discovery_and_missing_polarization(tmp_path):
    archive = safe_zip(tmp_path / "safe.zip", valid_raster=False)
    members = inspect_archive(archive)
    assert len(measurement_members(members, "VV")) == 1
    with pytest.raises(MLXUserError, match="no VH"):
        measurement_members(members, "VH")


@pytest.mark.parametrize("radiometry", ["dn", "sigma0", "sigma0-db"])
def test_preparation_arrays_metadata_and_previews(tmp_path, radiometry):
    np = pytest.importorskip("numpy")
    source = safe_zip(tmp_path / "safe.zip")
    result = PrepareSarImagery(PrepareRequest(source, tmp_path / "prepared", tile_size=3, radiometry=radiometry)).execute()
    manifest_path = Path(result.manifests[0])
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["tiles"]) == 4
    tile = manifest["tiles"][0]
    data = np.load(manifest_path.parent / tile["array"])
    expected = np.arange(16, dtype="uint16").reshape(4,4)[:3,:3]
    if radiometry == "dn":
        np.testing.assert_array_equal(data, expected)
        assert data.dtype == np.uint16
    else:
        target = expected.astype(float)**2/4
        target[0,0] = np.nan
        if radiometry == "sigma0-db":
            target = 10*np.log10(target)
        np.testing.assert_allclose(data, target, rtol=1e-6)
        assert data.dtype == np.float32
    assert tile["crs"] is None and tile["geographic_bounds"] is None
    assert tile["acquisition_time"] == "2020-01-01T00:00:00+00:00"
    assert tile["polarization"] == "VV"
    assert len(tile["sha256"]) == 64
    assert (manifest_path.parent / tile["preview"]).exists()
    assert not manifest["terrain_corrected"]


def test_geographic_crop_rejected_even_with_annotation_coordinates(tmp_path):
    source = safe_zip(tmp_path / "safe.zip")
    output = tmp_path / "prepared"
    with pytest.raises(MLXUserError, match="Geographic cropping.*unavailable"):
        PrepareSarImagery(PrepareRequest(source, output, bbox=BoundingBox(20,10,21,11))).execute()
    assert not output.exists()


def test_dry_run_and_single_scene_default(tmp_path):
    provider = FakeProvider([scene(), scene(scene_id="older", filename="older.zip", acquisition_time="2019-01-01")])
    req = DownloadRequest(request(), tmp_path / "out", tmp_path / "cache", dry_run=True)
    result = DownloadSarScenes(req, provider, ArchiveCache(req.cache)).execute()
    assert len(result.scenes) == 1 and result.dry_run and provider.calls == 0
    assert not req.cache.exists()
    result = DownloadSarScenes(replace(req, all_scenes=True), provider, ArchiveCache(req.cache)).execute()
    assert len(result.scenes) == 2


def test_cli_registry_and_clean_json(tmp_path, monkeypatch, capsys):
    from mlx.cli import main
    from mlx.cli_routing import resolve_mode_descriptor
    from mlx.modes.sar.providers import ProviderRegistry
    assert resolve_mode_descriptor("sar").actions == ("search", "download", "prepare")
    monkeypatch.setattr(ProviderRegistry, "create", lambda *a, **k: FakeProvider())
    assert main(["--mode", "sar", "--provider", "asf", "--action", "search", "--scene-id", "S1_TEST-GRD_HD", "--output", str(tmp_path), "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["scenes"][0]["scene_id"] == "S1_TEST-GRD_HD"
    assert not captured.err


def test_environment_precedence(tmp_path):
    from mlx.modes.sar.configuration import settings
    environment = {"MLX_SAR_OUTPUT_DIR": str(tmp_path), "MLX_SAR_DOWNLOAD_PROCESSES": "2"}
    output, _, network = settings({"download_processes": 3}, environment)
    assert output == tmp_path and network.processes == 3
    assert settings({}, environment)[2].processes == 2
    with pytest.raises(MLXUserError):
        settings({}, {"MLX_SAR_MAX_RETRIES": "bad"})


def test_provider_calls_real_sdk_contract_without_credentials():
    asf = pytest.importorskip("asf_search")
    from mlx.modes.sar.asf.provider import AsfProvider
    calls = []
    class Library:
        PLATFORM = asf.PLATFORM
        PRODUCT_TYPE = asf.PRODUCT_TYPE
        ASFSearchOptions = asf.ASFSearchOptions
        ASFSession = asf.ASFSession
        def geo_search(self, **kwargs):
            calls.append(kwargs)
            return []
    provider = AsfProvider(library=Library())
    assert provider.search(request(polarization="VV")) == []
    assert calls[0]["processingLevel"] == "GRD_HD"
    assert calls[0]["polarization"] == ["VV", "VV+VH"]
    assert calls[0]["intersectsWith"] == request().bbox.wkt()
    assert "Authorization" not in calls[0]["opts"].session.headers


def test_unsafe_urls_rejected_before_authentication():
    from mlx.modes.sar.asf.client import validate_download_url
    for url in ("http://datapool.asf.alaska.edu/x", "https://evil.test/x", "https://datapool.asf.alaska.edu.evil.test/x", "https://u:p@datapool.asf.alaska.edu/x", "https://datapool.asf.alaska.edu/x?token=secret"):
        with pytest.raises(MLXUserError):
            validate_download_url(url)


def test_transient_retry_and_authorization():
    from requests import Response
    from requests.exceptions import HTTPError
    from mlx.modes.sar.asf.client import retry_call
    calls, delays = [], []
    def operation():
        calls.append(1)
        if len(calls) < 3:
            r = Response(); r.status_code = 429
            raise HTTPError(response=r)
        return "ok"
    assert retry_call(operation, NetworkOptions(), sleep=delays.append) == "ok"
    assert delays == [1, 2]
    def unauthorized():
        r = Response(); r.status_code = 401
        raise HTTPError(response=r)
    with pytest.raises(MLXUserError, match="authorization failed"):
        retry_call(unauthorized, NetworkOptions(), sleep=delays.append)


def test_equal_area_coverage_known_rectangles():
    pytest.importorskip("shapely"); pytest.importorskip("pyproj")
    from mlx.modes.sar.asf.provider import _coverage
    bbox = BoundingBox(0,0,2,2)
    geometry = {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,2],[0,2],[0,0]]]}
    assert _coverage(geometry, bbox) == pytest.approx(.5)


def test_provider_download_uses_sdk_and_restarts_partial_attempts(tmp_path):
    from mlx.modes.sar.asf.provider import AsfProvider
    from requests.exceptions import ConnectionError
    calls = []
    class Session(AuthSession):
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Library:
        def download_url(self, url, path, filename, session):
            calls.append((url, session.called))
            destination = Path(path) / filename
            assert not destination.exists()
            if len(calls) == 1:
                destination.write_bytes(b"partial")
                raise ConnectionError("broken stream")
            safe_zip(destination, valid_raster=False)
    provider = AsfProvider(EarthdataCredentials(token="test-token"), NetworkOptions(max_retries=1),
                           library=Library(), session_factory=lambda *a, **k: Session())
    destination = provider.download(scene(), tmp_path)
    assert len(inspect_archive(destination)) == 1
    assert len(calls) == 2
    assert all(call[1] == ("token", "test-token") for call in calls)


def test_sdk_session_timeout_and_redirect_credential_policy(monkeypatch):
    asf = pytest.importorskip("asf_search")
    import requests
    from mlx.modes.sar.asf.client import create_session
    observed = []
    def send(self, request, **kwargs):
        observed.append((dict(request.headers), kwargs))
        response = requests.Response(); response.status_code = 200
        return response
    monkeypatch.setattr(asf.ASFSession, "send", send)
    session = create_session(asf, NetworkOptions(timeout=17))
    request = requests.Request("GET", "https://bucket.s3.amazonaws.com/archive", headers={"Authorization":"secret", "Cookie":"private"}).prepare()
    session.send(request)
    assert observed[0][1]["timeout"] == 17
    assert "Authorization" not in observed[0][0] and "Cookie" not in observed[0][0]
    with pytest.raises(MLXUserError, match="unsupported host"):
        session.send(requests.Request("GET", "https://evil.test/archive").prepare())
    assert len(observed) == 1


def test_nonconstant_bilinear_calibration_and_bounds():
    np = pytest.importorskip("numpy")
    from mlx.modes.sar.calibration import SigmaCalibration
    xml = b'''<calibration><calibrationVectorList>
    <calibrationVector><line>0</line><pixel>0 2</pixel><sigmaNought>1 3</sigmaNought></calibrationVector>
    <calibrationVector><line>2</line><pixel>0 2</pixel><sigmaNought>3 5</sigmaNought></calibrationVector>
    </calibrationVectorList></calibration>'''
    calibration = SigmaCalibration(xml)
    assert calibration.apply(np.array([[6]], dtype="uint16"), 1, 1)[0,0] == pytest.approx(4)
    with pytest.raises(MLXUserError, match="does not cover"):
        calibration.apply(np.array([[6]], dtype="uint16"), 3, 1)


def test_missing_polarization_leaves_no_completed_output(tmp_path):
    source = safe_zip(tmp_path / "safe.zip", valid_raster=False)
    output = tmp_path / "out"
    with pytest.raises(MLXUserError, match="no VH"):
        PrepareSarImagery(PrepareRequest(source, output, polarization="VH")).execute()
    assert not list(output.iterdir())


def test_exact_catalogue_size_mismatch_never_verified(tmp_path):
    candidate = scene(size_bytes=5, size_is_exact=True)
    cache = ArchiveCache(tmp_path / "cache")
    with pytest.raises(MLXUserError, match="size differs"):
        cache.obtain(candidate, FakeProvider(), tmp_path / "out")
    assert not list((tmp_path / "cache").rglob("verified.json"))


def test_search_and_artifacts_exclude_environment_secrets(tmp_path, monkeypatch, capsys):
    from mlx.modes.sar.providers import ProviderRegistry
    from mlx.modes.sar.runner import run_sar
    monkeypatch.setattr(ProviderRegistry, "create", lambda *a, **k: FakeProvider())
    result = run_sar({"action":"search", "scene_id":"S1_TEST-GRD_HD", "output_path":str(tmp_path), "output_format":"json"},
                     environment={"EARTHDATA_TOKEN":"private-token", "EARTHDATA_PASSWORD":"private-password"})
    serialized = json.dumps(json_safe(result)) + (tmp_path / "metadata/scenes.json").read_text()
    assert "private-token" not in serialized and "private-password" not in serialized
    assert not capsys.readouterr().out


def test_sar_cli_imports_no_optional_geospatial_providers():
    import subprocess, sys
    code = '''import sys
from mlx.cli import build_parser
build_parser().parse_args(['--mode', 'sar', '--action', 'search'])
assert not {'asf_search','rasterio','shapely','pyproj'} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_download_inventory_preserves_previous_scenes(tmp_path):
    cache = ArchiveCache(tmp_path / "cache")
    req = DownloadRequest(request(), tmp_path / "out", tmp_path / "cache")
    DownloadSarScenes(req, FakeProvider(), cache).execute()
    other = scene(scene_id="second", filename="second.zip")
    DownloadSarScenes(req, FakeProvider([other]), cache).execute()
    records = json.loads((tmp_path / "out/metadata/downloads.json").read_text())
    assert {row["scene_id"] for row in records} == {scene().scene_id, "second"}


def test_zip_crc_corruption_detected(tmp_path):
    archive = safe_zip(tmp_path / "bad.zip", valid_raster=False)
    archive.write_bytes(archive.read_bytes().replace(b"synthetic", b"corrupted", 1))
    with pytest.raises(MLXUserError, match="integrity check failed"):
        inspect_archive(archive)


def test_exact_scene_lookup_calls_product_search():
    asf = pytest.importorskip("asf_search")
    from mlx.modes.sar.asf.provider import AsfProvider
    calls = []
    class Library:
        ASFSession = asf.ASFSession
        ASFSearchOptions = asf.ASFSearchOptions
        def product_search(self, ids, opts):
            calls.append(ids)
            return []
    assert AsfProvider(library=Library()).search(SearchRequest(scene_id="selected-GRD_HD")) == []
    assert calls == [["selected-GRD_HD"]]


def test_mixed_acquisition_archive_rejected(tmp_path):
    archive = safe_zip(tmp_path / "mixed.zip", valid_raster=False)
    with zipfile.ZipFile(archive, 'a') as contents:
        contents.writestr('OTHER.SAFE/measurement/s1a-iw-grd-vv-other.tiff', b'data')
    with pytest.raises(MLXUserError, match="one top-level SAFE"):
        inspect_archive(archive)


def test_invalid_url_port_is_user_error():
    from mlx.modes.sar.asf.client import validate_download_url
    with pytest.raises(MLXUserError, match="Malformed ASF"):
        validate_download_url('https://datapool.asf.alaska.edu:notaport/archive.zip')


def test_truncated_http_stream_is_retried():
    from requests.exceptions import ChunkedEncodingError
    from mlx.modes.sar.asf.client import retry_call
    attempts = []
    def transfer():
        attempts.append(1)
        if len(attempts) == 1:
            raise ChunkedEncodingError('incomplete stream')
        return 'complete'
    assert retry_call(transfer, NetworkOptions(max_retries=1), sleep=lambda _: None) == 'complete'
    assert len(attempts) == 2
