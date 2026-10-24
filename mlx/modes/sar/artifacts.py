"""Verified original archives, isolated staging, and persistent cache entries."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from tempfile import TemporaryDirectory
import zipfile

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError


def safe_component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise MLXUserError("Unsafe scene identifier or archive filename in source metadata.")
    return value


def inspect_archive(path):
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise MLXUserError("SAFE ZIP contains duplicate member names.")
            if any(n.startswith("/") or ".." in Path(n).parts or "\\" in n for n in members):
                raise MLXUserError("SAFE ZIP contains unsafe member paths.")
            measurements = sorted(n for n in members if ".SAFE/measurement/" in n and n.lower().endswith((".tiff", ".tif")))
            if not measurements or not any(n.endswith(".SAFE/manifest.safe") for n in members):
                raise MLXUserError("Archive is not a Sentinel-1 SAFE ZIP with measurement TIFFs and manifest.safe.")
            roots = {PurePosixPath(n).parts[0] for n in measurements}
            if (len(roots) != 1 or any(len(PurePosixPath(n).parts) != 3 for n in measurements)
                    or f"{next(iter(roots))}/manifest.safe" not in members):
                raise MLXUserError("Expected one top-level SAFE acquisition with a matching manifest.")
            if archive.testzip() is not None:
                raise MLXUserError("SAFE ZIP integrity check failed. Remove the corrupt archive and download again.")
            return measurements
    except (zipfile.BadZipFile, OSError, RuntimeError, EOFError) as exc:
        raise MLXUserError(f"Unable to validate SAFE ZIP '{path.name}'; restart the download.") from exc


@contextmanager
def cache_lock(directory):
    lock = directory / ".lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise MLXUserError(f"SAR cache entry is busy: {directory}. If no download is running, remove its stale .lock directory.") from exc
    try:
        yield
    finally:
        lock.rmdir()


class ArchiveCache:
    def __init__(self, root):
        self.root = Path(root).expanduser()

    def obtain(self, scene, provider, output):
        safe_component(scene.scene_id)
        safe_component(scene.filename)
        if not scene.filename.lower().endswith(".zip"):
            raise MLXUserError("Expected an original SAFE .zip archive filename.")
        identity = {"scene_id": scene.scene_id, "url": scene.url, "filename": scene.filename,
                    "size_bytes": scene.size_bytes, "size_is_exact": scene.size_is_exact,
                    "metadata": scene.metadata}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = self.root / "products" / key
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / scene.filename
        record_path = directory / "verified.json"
        with cache_lock(directory):
            record = self._verified(target, record_path, key)
            hit = record is not None
            if not hit:
                record_path.unlink(missing_ok=True)
                with TemporaryDirectory(prefix="partial-", dir=directory) as temporary:
                    downloaded = provider.download(scene, Path(temporary))
                    if downloaded != Path(temporary) / scene.filename:
                        raise MLXUserError("SAR provider returned an unexpected download path.")
                    inspect_archive(downloaded)
                    if scene.size_is_exact and downloaded.stat().st_size != scene.size_bytes:
                        raise MLXUserError("Downloaded archive size differs from the ASF catalogue. Retry the download or refresh the scene metadata.")
                    record = {"identity": key, "scene_id": scene.scene_id, "filename": scene.filename,
                              "size_bytes": downloaded.stat().st_size, "sha256": sha256_file(downloaded),
                              "source": identity}
                    downloaded.replace(target)
                    write_json_atomic(record_path, record)
            raw = Path(output) / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            destination = raw / scene.filename
            if destination.exists() and sha256_file(destination) != record["sha256"]:
                raise MLXUserError(f"Existing output archive differs: {destination}. Move it aside before retrying.")
            if not destination.exists():
                with TemporaryDirectory(prefix=".sar-", dir=raw) as temporary:
                    staged = Path(temporary) / scene.filename
                    try:
                        os.link(target, staged)
                    except OSError:
                        shutil.copyfile(target, staged)
                    staged.replace(destination)
            return {**record, "path": str(destination), "cache_hit": hit}

    @staticmethod
    def _verified(target, record_path, key):
        try:
            record = json.loads(record_path.read_text())
            if (record["identity"] != key or target.stat().st_size != record["size_bytes"]
                    or sha256_file(target) != record["sha256"]):
                return None
            inspect_archive(target)
            return record
        except (OSError, ValueError, KeyError, TypeError, MLXUserError):
            return None
