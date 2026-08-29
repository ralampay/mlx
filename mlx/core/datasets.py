from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Generic, Mapping, Optional, Protocol, TypeVar
from urllib.parse import urlparse

from mlx.core.commands import NullWorkflowReporter, WorkflowReporter, emit
from mlx.core.exceptions import MLXUserError


DEFAULT_DATASET_CACHE = Path("~/.cache/mlx/datasets")
_CACHE_MANIFEST = "mlx-dataset-cache.json"
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
ResultT_co = TypeVar("ResultT_co", covariant=True)


class Executable(Protocol[ResultT_co]):
    def execute(self) -> ResultT_co:
        ...


class S3Client(Protocol):
    def head_object(self, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def get_object(self, **kwargs: Any) -> Mapping[str, Any]:
        ...


DatasetRootResolver = Callable[[Path], Path]


@dataclass(frozen=True)
class S3Location:
    uri: str
    bucket: str
    key: str


@dataclass(frozen=True)
class StagedDataset:
    dataset_path: Path
    cache_path: Path
    source: Mapping[str, Any]


def parse_s3_zip_uri(uri: str) -> S3Location:
    value = str(uri or "").strip()
    parsed = urlparse(value)
    key = parsed.path.lstrip("/")
    if parsed.scheme.lower() != "s3" or not parsed.netloc or not key:
        raise MLXUserError(
            "--dataset-s3-uri must be an S3 object URI such as "
            "s3://bucket/path/dataset.zip."
        )
    if parsed.params or parsed.query or parsed.fragment:
        raise MLXUserError("--dataset-s3-uri must not contain parameters, a query, or a fragment.")
    if not key.lower().endswith(".zip"):
        raise MLXUserError("--dataset-s3-uri must point to a .zip object.")
    return S3Location(uri=f"s3://{parsed.netloc}/{key}", bucket=parsed.netloc, key=key)


def create_s3_client(*, profile: Optional[str] = None) -> S3Client:
    try:
        import boto3
    except ImportError as exc:
        raise MLXUserError(
            "S3 dataset staging requires Boto3. Install MLX with the 'aws' extra."
        ) from exc
    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        return session.client("s3")
    except Exception as exc:
        label = f" using AWS profile '{profile}'" if profile else ""
        raise MLXUserError(f"Unable to create the S3 client{label}: {exc}") from exc


def extract_zip_safely(
    archive: str | Path,
    destination: str | Path,
    *,
    max_uncompressed_bytes: Optional[int] = None,
) -> None:
    archive_path = Path(archive)
    destination_path = Path(destination)
    try:
        destination_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as source:
            members = source.infolist()
            total_size = sum(member.file_size for member in members)
            available = shutil.disk_usage(destination_path).free
            effective_limit = available
            if max_uncompressed_bytes is not None:
                effective_limit = min(effective_limit, max_uncompressed_bytes)
            if total_size > effective_limit:
                if max_uncompressed_bytes is not None and total_size > max_uncompressed_bytes:
                    raise MLXUserError(
                        "The extracted dataset would exceed the configured extraction limit."
                    )
                raise MLXUserError(
                    f"The dataset requires {total_size} extracted bytes, but only "
                    f"{available} bytes are available."
                )

            validated: list[tuple[zipfile.ZipInfo, PurePosixPath, int]] = []
            seen: dict[str, int] = {}
            for member in members:
                raw = member.filename.replace("\\", "/")
                path_text = raw[:-1] if raw.endswith("/") else raw
                path = PurePosixPath(raw)
                if (
                    not raw
                    or "\x00" in raw
                    or raw.startswith("/")
                    or _WINDOWS_DRIVE.match(raw)
                    or any(part in {"", ".", ".."} for part in path_text.split("/"))
                ):
                    raise MLXUserError(f"Dataset ZIP contains an unsafe path: {member.filename}")
                normalized = path.as_posix().rstrip("/")
                collision_key = normalized.casefold()
                if not normalized or collision_key in seen:
                    raise MLXUserError(
                        f"Dataset ZIP contains a duplicate path: {member.filename}"
                    )

                mode = member.external_attr >> 16
                kind = stat.S_IFMT(mode)
                is_directory = member.is_dir() or raw.endswith("/")
                if kind == stat.S_IFLNK:
                    raise MLXUserError(
                        f"Dataset ZIP contains a symbolic link: {member.filename}"
                    )
                if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise MLXUserError(
                        f"Dataset ZIP contains an unsupported special file: {member.filename}"
                    )
                effective_kind = kind if kind else (stat.S_IFDIR if is_directory else stat.S_IFREG)
                parent_parts = path.parts[:-1]
                for index in range(1, len(parent_parts) + 1):
                    parent = PurePosixPath(*parent_parts[:index]).as_posix().casefold()
                    if seen.get(parent) == stat.S_IFREG:
                        raise MLXUserError(
                            f"Dataset ZIP contains conflicting paths near: {member.filename}"
                        )
                prefix = f"{collision_key}/"
                if effective_kind == stat.S_IFREG and any(
                    existing.startswith(prefix) for existing in seen
                ):
                    raise MLXUserError(
                        f"Dataset ZIP contains conflicting paths near: {member.filename}"
                    )
                seen[collision_key] = effective_kind
                validated.append((member, path, effective_kind))

            root = destination_path.resolve()
            for member, path, kind in validated:
                target = destination_path.joinpath(*path.parts)
                resolved = target.resolve()
                if os.path.commonpath((str(root), str(resolved))) != str(root):
                    raise MLXUserError(f"Dataset ZIP contains an unsafe path: {member.filename}")
                if kind == stat.S_IFDIR:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_stream, target.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    except zipfile.BadZipFile as exc:
        raise MLXUserError(f"Dataset ZIP is invalid: {archive_path}") from exc
    except OSError as exc:
        raise MLXUserError(f"Unable to extract dataset ZIP '{archive_path}': {exc}") from exc


def resolve_object_detection_dataset_root(extracted_path: Path) -> Path:
    candidates = sorted(path for path in extracted_path.rglob("data.yaml") if path.is_file())
    if len(candidates) != 1:
        raise MLXUserError(
            "The extracted object-detection dataset must contain exactly one data.yaml; "
            f"found {len(candidates)}."
        )
    return candidates[0].parent


def resolve_split_dataset_root(
    extracted_path: Path,
    *,
    required_paths: tuple[str, ...] = ("train", "val"),
    dataset_label: str = "dataset",
) -> Path:
    directories = [extracted_path, *(path for path in extracted_path.rglob("*") if path.is_dir())]
    candidates = [
        path for path in directories if all((path / required).is_dir() for required in required_paths)
    ]
    if len(candidates) != 1:
        required = ", ".join(f"{path}/" for path in required_paths)
        raise MLXUserError(
            f"The extracted {dataset_label} must contain exactly one root with {required}; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def classification_dataset_root(extracted_path: Path) -> Path:
    return resolve_split_dataset_root(
        extracted_path,
        dataset_label="image-classification dataset",
    )


def segmentation_dataset_root(extracted_path: Path) -> Path:
    return resolve_split_dataset_root(
        extracted_path,
        required_paths=("train/images", "train/masks", "val/images", "val/masks"),
        dataset_label="segmentation dataset",
    )


def video_anomaly_dataset_root(extracted_path: Path) -> Path:
    return resolve_split_dataset_root(
        extracted_path,
        required_paths=("train/normal", "val/normal"),
        dataset_label="video-anomaly dataset",
    )


class StageS3Dataset:
    def __init__(
        self,
        dataset_s3_uri: str,
        *,
        root_resolver: DatasetRootResolver,
        cache_dir: str | Path = DEFAULT_DATASET_CACHE,
        s3_client: Optional[S3Client] = None,
        profile: Optional[str] = None,
        reporter: Optional[WorkflowReporter] = None,
    ) -> None:
        self.location = parse_s3_zip_uri(dataset_s3_uri)
        self.root_resolver = root_resolver
        self.cache_dir = _validated_cache_dir(cache_dir).expanduser()
        self.s3_client = s3_client
        self.profile = profile
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> StagedDataset:
        client = self.s3_client or create_s3_client(profile=self.profile)
        identity = self._head(client)
        cache_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        target = self.cache_dir / cache_key
        cached = self._load_cached(target, identity)
        if cached is not None:
            emit(self.reporter, "info", f"Using cached S3 dataset at {cached.dataset_path}")
            return cached

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        build = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=self.cache_dir))
        archive = build / "dataset.zip"
        extracted = build / "dataset"
        try:
            digest = self._download(client, identity, archive)
            emit(self.reporter, "info", "Extracting S3 dataset ZIP")
            extract_zip_safely(archive, extracted)
            archive.unlink()
            root = self.root_resolver(extracted).resolve()
            try:
                root_relative = root.relative_to(build.resolve()).as_posix()
            except ValueError as exc:
                raise MLXUserError(
                    "The resolved dataset root must remain inside the extracted archive."
                ) from exc
            source = {
                **identity,
                "sha256": digest,
                "cache_identity": cache_key,
                "dataset_root": root_relative,
            }
            _write_json_atomic(build / _CACHE_MANIFEST, source)
            self._publish_cache(build, target, identity)
            result = self._load_cached(target, identity)
            if result is None:  # pragma: no cover - defensive filesystem guard
                raise MLXUserError("The staged dataset cache could not be validated after publication.")
            emit(self.reporter, "info", f"S3 dataset ready at {result.dataset_path}")
            return result
        except Exception:
            if build.exists():
                shutil.rmtree(build, ignore_errors=True)
            raise

    def _head(self, client: S3Client) -> dict[str, Any]:
        try:
            value = client.head_object(Bucket=self.location.bucket, Key=self.location.key)
        except Exception as exc:
            raise MLXUserError(
                f"Unable to inspect S3 dataset '{self.location.uri}': {exc}"
            ) from exc
        version = value.get("VersionId")
        etag = str(value.get("ETag") or "").strip('"') or None
        size = int(value.get("ContentLength", -1))
        if size < 0:
            raise MLXUserError("S3 did not return a valid ContentLength for the dataset ZIP.")
        modified = value.get("LastModified")
        return {
            "uri": self.location.uri,
            "bucket": self.location.bucket,
            "key": self.location.key,
            "version_id": str(version) if version is not None else None,
            "etag": etag,
            "content_length": size,
            "last_modified": modified.isoformat() if hasattr(modified, "isoformat") else str(modified or ""),
        }

    def _download(self, client: S3Client, identity: Mapping[str, Any], target: Path) -> str:
        params: dict[str, Any] = {"Bucket": identity["bucket"], "Key": identity["key"]}
        if identity.get("version_id"):
            params["VersionId"] = identity["version_id"]
        try:
            response = client.get_object(**params)
            body = response["Body"]
            response_etag = str(response.get("ETag") or "").strip('"') or None
            if response_etag and identity.get("etag") and response_etag != identity["etag"]:
                raise MLXUserError("The S3 dataset changed between inspection and download; retry training.")
            response_version = response.get("VersionId")
            if (
                response_version is not None
                and identity.get("version_id")
                and str(response_version) != identity["version_id"]
            ):
                raise MLXUserError("The S3 dataset version changed during download; retry training.")
            digest = hashlib.sha256()
            downloaded = 0
            total = int(identity["content_length"])
            emit(self.reporter, "info", f"Downloading {identity['uri']}", current=0, total=total)
            with target.open("xb") as output:
                while True:
                    chunk = body.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    emit(
                        self.reporter,
                        "progress",
                        "Downloading S3 dataset",
                        current=downloaded,
                        total=total,
                        payload={"event": "dataset_download"},
                    )
        except MLXUserError:
            raise
        except Exception as exc:
            raise MLXUserError(f"Unable to download S3 dataset '{identity['uri']}': {exc}") from exc
        finally:
            close = getattr(locals().get("body"), "close", None)
            if callable(close):
                close()
        if downloaded != total:
            raise MLXUserError(
                f"S3 dataset download was incomplete: expected {total} bytes, received {downloaded}."
            )
        return digest.hexdigest()

    @staticmethod
    def _load_cached(target: Path, identity: Mapping[str, Any]) -> Optional[StagedDataset]:
        manifest_path = target / _CACHE_MANIFEST
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in identity.items():
                if manifest.get(key) != value:
                    return None
            dataset_path = (target / str(manifest["dataset_root"])).resolve()
            dataset_path.relative_to(target.resolve())
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        if not dataset_path.is_dir():
            return None
        return StagedDataset(dataset_path=dataset_path, cache_path=target, source=manifest)

    @staticmethod
    def _publish_cache(build: Path, target: Path, identity: Mapping[str, Any]) -> None:
        if target.exists():
            cached = StageS3Dataset._load_cached(target, identity)
            if cached is not None:
                shutil.rmtree(build, ignore_errors=True)
                return
            shutil.rmtree(target)
        try:
            build.rename(target)
        except OSError as exc:
            cached = StageS3Dataset._load_cached(target, identity)
            if cached is None:
                raise MLXUserError(f"Unable to publish dataset cache at '{target}': {exc}") from exc
            shutil.rmtree(build, ignore_errors=True)


class TrainWithDatasetSource(Generic[RequestT, ResultT]):
    def __init__(
        self,
        request: RequestT,
        *,
        trainer_factory: Callable[[RequestT], Executable[ResultT]],
        root_resolver: DatasetRootResolver,
        artifact_dir_resolver: Callable[[RequestT], Path],
        s3_client: Optional[S3Client] = None,
        profile: Optional[str] = None,
        reporter: Optional[WorkflowReporter] = None,
    ) -> None:
        self.request = request
        self.trainer_factory = trainer_factory
        self.root_resolver = root_resolver
        self.artifact_dir_resolver = artifact_dir_resolver
        self.s3_client = s3_client
        self.profile = profile
        self.reporter = reporter or NullWorkflowReporter()

    def execute(self) -> ResultT:
        uri = str(getattr(self.request, "dataset_s3_uri", None) or "").strip()
        if not uri:
            return self.trainer_factory(self.request).execute()
        local_path = str(getattr(self.request, "dataset_path", "") or "").strip()
        if local_path not in {"", "./tmp/dataset"}:
            raise MLXUserError("Provide either a local dataset path or an S3 dataset URI, not both.")
        output_path = str(getattr(self.request, "output_path", None) or "").strip()
        if not output_path:
            raise MLXUserError("S3 dataset training requires --output for persistent artifacts.")
        staged = StageS3Dataset(
            uri,
            root_resolver=self.root_resolver,
            cache_dir=_validated_cache_dir(
                getattr(self.request, "dataset_cache_dir", DEFAULT_DATASET_CACHE)
            ),
            s3_client=self.s3_client,
            profile=self.profile,
            reporter=self.reporter,
        ).execute()
        resolved_request = replace(self.request, dataset_path=str(staged.dataset_path))
        artifact_dir = self.artifact_dir_resolver(resolved_request).expanduser()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(artifact_dir / "dataset_source.json", dict(staged.source))
        result = self.trainer_factory(resolved_request).execute()
        _write_json_atomic(artifact_dir / "dataset_source.json", dict(staged.source))
        return result


def validate_dataset_source_options(config: Mapping[str, Any], *, action: str) -> None:
    explicit = set(config.get("_explicit_options") or ())
    has_s3 = bool(str(config.get("dataset_s3_uri") or "").strip())
    if not has_s3:
        return
    if action != "train":
        raise MLXUserError("--dataset-s3-uri is supported only for training actions.")
    if "dataset_path" in explicit:
        raise MLXUserError("Use either --dataset or --dataset-s3-uri for training, not both.")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise MLXUserError(f"Unable to write dataset provenance '{path}': {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _validated_cache_dir(value: str | Path) -> Path:
    if not str(value).strip():
        raise MLXUserError("--dataset-cache-dir cannot be empty.")
    return Path(value)


__all__ = [
    "DEFAULT_DATASET_CACHE",
    "StageS3Dataset",
    "StagedDataset",
    "TrainWithDatasetSource",
    "classification_dataset_root",
    "create_s3_client",
    "extract_zip_safely",
    "parse_s3_zip_uri",
    "resolve_object_detection_dataset_root",
    "resolve_split_dataset_root",
    "segmentation_dataset_root",
    "validate_dataset_source_options",
    "video_anomaly_dataset_root",
]
