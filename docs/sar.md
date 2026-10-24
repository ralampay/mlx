# Sentinel-1 SAR ingestion

[Architecture](../ARCHITECTURE.md) · [CLI](cli.md)

Synthetic Aperture Radar (SAR) measures microwave returns, allowing observation at night
and through clouds. Sentinel-1 Level-1 Ground Range Detected (GRD) products contain
multi-look detected amplitudes, SAFE XML annotations, calibration lookup tables, and
geolocation tie points. They do not contain ship labels. Native measurement TIFFs are
radar-grid images, not necessarily affine map-projected GeoTIFFs.

MLX's `sar` mode uses the official [ASF Python library](https://docs.asf.alaska.edu/asf_search/basics/).
ASF is the initial provider. Searches intersect footprints with your rectangle; they do
**not** download or crop pixels. Downloads retain the complete original SAFE ZIP.

## Installation

From this checkout, with Python 3.10 or newer:

```bash
python -m pip install -e '.[sar]'
python -m mlx --doctor
```

The extra adds `asf_search>=10,<12`, `rasterio>=1.4`, `shapely>=2`, `pyproj>=3`,
`python-dotenv>=1`, `defusedxml>=0.7`, and `Pillow>=10`. NumPy is already a base dependency.
SAR modules import geospatial libraries lazily; other modes do not require this extra.
No new CUDA dependency is added. The existing MLX base install still includes PyTorch.

Prefer binary Rasterio and PyProj wheels. On Linux, a source build requires a compatible
GDAL development installation (`gdal-bin`, `libgdal-dev`), PROJ development headers,
and a C/C++ toolchain. Use a Python version with wheels for your platform or a dedicated
Conda geospatial environment. This implementation does not invoke SNAP or install Java.
Installing GDAL alone does not add terrain correction to MLX.

## Earthdata setup and configuration

Create an account at [NASA Earthdata Login](https://urs.earthdata.nasa.gov/users/new),
complete account verification, and authorize the ASF application if prompted by NASA.
Generate a user token from [Earthdata user tokens](https://urs.earthdata.nasa.gov/user_tokens).
Treat tokens and passwords as secrets; renew expired tokens.

`.env.dist` is a committed example with empty credentials; `.env` is your private local
configuration and is ignored by Git. Copy the example to `.env` **only if you do not already
have a `.env`**, then fill it locally. With the SAR extra installed, the existing MLX CLI
loads `.env` from the checkout root without replacing already-exported variables. You may
instead export variables in your shell or supply them through a secret manager.

```dotenv
EARTHDATA_TOKEN=
EARTHDATA_USERNAME=
EARTHDATA_PASSWORD=
MLX_SAR_OUTPUT_DIR=~/Desktop/datasets/sar
MLX_SAR_CACHE_DIR=~/.cache/mlx/sar
MLX_SAR_DOWNLOAD_PROCESSES=1
MLX_SAR_REQUEST_TIMEOUT=60
MLX_SAR_MAX_RETRIES=3
```

Precedence is explicit CLI option, environment variable, then the defaults shown above.
The CLI overrides are `--output`, `--sar-cache-dir`, `--download-processes`,
`--request-timeout`, and `--max-retries`. `~` is expanded for storage paths.
For `prepare`, omitted `--output` uses a `prepared` subdirectory of the configured SAR root.

Authentication precedence is token, complete username/password pair, then `.netrc`.
A supplied invalid token fails rather than silently falling back to another identity.
Public catalogue searches and dry runs require no credentials. Downloads authenticate
noninteractively through `ASFSession.auth_with_token()` or `auth_with_creds()`; the same
credential method is used after reading an Earthdata `.netrc` entry.

Alternative `~/.netrc` contents:

```text
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
```

Protect it:

```bash
chmod 600 ~/.netrc
```

The Requests `NETRC` environment override is supported by credential lookup. No interactive
login is started. Credentials are excluded from reports, scene metadata, and manifests.
Initial download URLs must be unsigned HTTPS ASF/Earthdata URLs; redirects outside trusted
hosts are rejected, except S3 redirects where authorization and cookie headers are removed.
Ambient proxy/netrc authentication is disabled on network sessions; `.netrc` is read explicitly
only for Earthdata login. This implementation does not support authenticated proxy configuration.

## Search a rectangle

```bash
python -m mlx --mode sar --provider asf --action search \
  --min-lon 117.50 --min-lat 14.90 --max-lon 118.00 --max-lat 15.40 \
  --start-date 2026-10-01 --end-date 2026-10-24 --max-results 10
```

Add `--format json` for one JSON result on stdout. Progress and Rich tables are suppressed;
handled errors go to stderr. Search also saves `metadata/scenes.json`, including original
ASF GeoJSON metadata. The table includes scene ID, UTC acquisition, satellite, processing
level, beam, polarization, direction, footprint, size, URL, and estimated AOI coverage.
Empty searches succeed with an empty scene list; downloads of an empty selection fail.

Defaults are Sentinel-1, `GRD_HD`, `IW`, and at most 20 catalogue candidates. Optional filters
are `--polarization VV`, `VH`, `HH`, `HV`, `VV+VH`, or `HH+HV`, and
`--flight-direction ASCENDING` or `DESCENDING`. A VV filter includes VV+VH products.
`--beam-mode` accepts IW, EW, or SM, but GRD_HD availability differs by acquisition mode.

Coordinates use longitude/latitude order. Latitude must be within [-90,90], longitude
within [-180,180], and each minimum must be smaller than its maximum. Split a rectangle
crossing the antimeridian into separate searches. Dates are ISO-8601 UTC; date-only values
mean midnight, so `--end-date 2026-10-24` is not the end of that day. Use a full timestamp
when needed. Start must not exceed end; maximum results must be a positive integer.

MLX filters valid Sentinel-1 GRD_HD products and requested channels/beam/direction, then
ranks by newest acquisition, higher AOI coverage, and stable scene identifier. Coverage
uses footprint intersection area divided by AOI area in an equal-area projection
(EPSG:6933). It is catalogue geometry coverage, not a promise of valid pixels or complete
coverage. Invalid/dateline-spanning footprints yield unknown coverage. Ranking considers
**only the limited returned candidates**, not the entire global catalogue; increase
`--max-results` or narrow the search to inspect a different subset.

## Download and dry run

Use the exact scene identifier (`fileID`, including its `-GRD_HD` suffix) from search:

```bash
python -m mlx --mode sar --provider asf --action download \
  --scene-id '<SCENE_ID>' --output ~/Desktop/datasets/sar/
```

Or discover and download the single top-ranked returned scene:

```bash
python -m mlx --mode sar --provider asf --action download \
  --min-lon 117.50 --min-lat 14.90 --max-lon 118.00 --max-lat 15.40 \
  --start-date 2026-10-01 --end-date 2026-10-24 --max-results 5 \
  --output ~/Desktop/datasets/sar/
```

Only explicit `--all-scenes` enables bulk downloading. `--scene-id` is an alternative to
AOI/date arguments, not a filter added to them. The default single scene can still exceed
a gigabyte. Review the plan first:

```bash
python -m mlx --mode sar --provider asf --action download \
  --min-lon 117.50 --min-lat 14.90 --max-lon 118.00 --max-lat 15.40 \
  --start-date 2026-10-01 --end-date 2026-10-24 --dry-run --format json
```

Dry run performs public discovery and saves search metadata, then returns scene URLs,
known expected sizes, and destination paths. It does not authenticate, download, or create
cache entries. Catalogue sizes may be unknown or approximate; exact byte counts are
verified when ASF supplies them. Concurrency defaults to one; `--download-processes N`
uses N concurrent scene workers, each with an independent authenticated SDK session.

### Integrity and cache

The persistent cache holds original ZIPs under `products/<source-identity-hash>/`.
Source identity includes the scene ID, URL, filename, size, and original metadata.
Downloads use temporary directories; transient connection failures, HTTP 429, and server
errors have bounded exponential retry delays. Incomplete downloads restart from zero;
byte-range resume is not promised. Completed archives pass ZIP CRC/structure checks,
exact catalogue size checks when available, and SHA-256 hashing before atomic publication.
Cache hits recheck size, SHA-256, and ZIP integrity. Unverified entries are never reused.

`raw/` uses hard links to avoid duplicate storage on the same filesystem, falling back to
copies elsewhere. Treat both raw and cached ZIPs as immutable: modifying a hard-linked
raw ZIP also modifies the cache. A per-entry `.lock` prevents concurrent cache writers;
after a killed process, remove it only after confirming no download is running. Abandoned
`partial-*` directories are never reused and may be removed. A changed source identity
creates a new entry. Delete an unused cache product directory to invalidate it; deleting
the cache does not remove raw hard links or copied output archives.

Search metadata is separate from verified products and is refreshed on every search.
`metadata/scenes.json` describes the latest query; `metadata/downloads.json` retains
completed acquisition records across sequential downloads. Do not run separate download
commands concurrently against the same output root; use the built-in concurrency instead.

## Offline preparation

No ASF access or credentials are needed:

```bash
python -m mlx --mode sar --action prepare \
  --input ~/Desktop/datasets/sar/raw/ \
  --output ~/Desktop/datasets/sar/prepared/ --polarization VV --tile-size 640
```

Input may be one SAFE ZIP, a directory of ZIPs, or the SAR root containing `raw/`.
Preparation validates archives, selects matching measurement TIFFs and annotations, and
writes edge-preserving radar-grid tiles. Missing polarization fails explicitly; scenes
need not contain both VV and VH. Without `--tile-size`, each measurement becomes one full
array, written in bounded row blocks. Preview dimensions are at most 1024 pixels on the
longest side. Existing prepared scene directories are not overwritten; choose a new
output directory when changing parameters. No archive extraction is required.

Radiometry options:

- `--radiometry dn` (default): original uint16 amplitude DN, unchanged. Zero is nodata.
- `--radiometry sigma0`: float32 linear sigma-nought, `DN² / sigmaNought_LUT²`, with
  bilinear interpolation of SAFE calibration vectors. Zero DN becomes NaN.
- `--radiometry sigma0-db`: `10*log10(sigma0)` in float32; invalid pixels remain NaN.

The [Sentinel-1 product specification](https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Specification-18052021.pdf)
defines the calibration LUTs and formula. Calibration requires valid matching XML and
rejects pixels outside LUT support. It is not thermal-noise removal, speckle filtering,
border-noise correction, radiometric terrain flattening, or terrain correction. Keep
original archives for later scientifically complete processing.

PNG previews are visualization-only sampled 2nd–98th percentile grayscale stretches.
They are not calibrated measurements and do not replace full-precision NumPy arrays.
Manifest entries record the actual stretch limits and sampling stride.

### Geographic cropping limitation

Native SAFE TIFFs have radar-grid geometry and may carry ground-control points rather
than a reliable affine transform. This version deliberately **rejects AOI coordinates on
`prepare`**, even if annotation tie points are present. It neither misuses EPSG:4326 nor
labels image tiling as terrain correction. CRS, affine transform, and geographic bounds
are null in tile manifests. Known-reference fixtures test this rejection boundary.

Use a reliable SAR processor such as [ESA SNAP](https://step.esa.int/main/download/snap-download/)
for orbit/geolocation and Range-Doppler Terrain Correction before geographic subsetting.
There is no automatic SNAP integration in this release, and externally terrain-corrected
rasters are not yet accepted by the native SAFE preparation action. Installing another
package does not silently enable this feature. Plain radar-grid tiling remains available.

## Outputs and future detection

```text
sar/
  raw/<original-scene-name>.zip
  metadata/scenes.json
  metadata/downloads.json
  prepared/<SAFE-scene-name>/
    tiles/<measurement>-r0-c0.npy
    previews/<measurement>-r0-c0.png
    manifest.json
```

Manifests include source archive hash, scene ID, acquisition time, platform/product,
polarization, radiometry and interpolation, original pixel window, dimensions, nodata,
explicitly unavailable geospatial fields, and array/preview hashes. All arrays are lossless;
no invented geospatial directory or labels are generated. Plan for multiple GB per raw
scene and additional storage for calibrated float32 arrays. ZIP CRC and SHA verification
read the full archive and therefore take time, including on cache hits.

Later ship-detection experiments can consume these arrays through a dedicated dataset
adapter. Supervised detection still requires bounding-box annotations tied to the exact
tile windows and preprocessing. MLX's existing detection and segmentation loaders are
unchanged; this output is not automatically a labelled YOLO dataset.

## Troubleshooting and validation

- 401/403: renew the token, verify credentials, and check Earthdata ASF authorization.
  Search success does not prove download authorization.
- Missing credentials: set one supported authentication method or configure `.netrc`.
- Incomplete/corrupt archive: retry; unfinished cache entries are never marked complete.
  If an existing raw output differs, move it aside explicitly before downloading again.
- Geographic crop error: omit coordinates for radar-grid processing; use a SAR terrain
  correction workflow for map-based subsetting.
- Missing optional dependency: install the `sar` extra in the interpreter running MLX.
- Missing polarization or calibration XML: inspect the selected product and choose a
  channel present in that archive; do not substitute another channel silently.

Default tests use synthetic SAFE files and fake providers:

```bash
python -m pytest -q tests/test_sar.py tests/test_sar_live.py
```

Opt-in public catalogue integration (one result, no download):

```bash
MLX_RUN_ASF_LIVE=1 python -m pytest -q tests/test_sar_live.py
```

Authenticated full archive downloads are a manual validation step because they can be
large. Review a dry-run plan, set credentials, and run the exact-scene command above.
