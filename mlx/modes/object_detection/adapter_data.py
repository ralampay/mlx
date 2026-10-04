"""Deterministic dataset preparation for detector-adapter experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import yaml

from mlx.core.exceptions import MLXUserError


FOUNDATION_CLASSES = ("person", "bicycle", "motorcycle", "car", "bus", "truck")
DAWN_CLASS_MAPPING = {
    "Person": 0,
    "Bicycle": 1,
    "Motorcycle": 2,
    "Car": 3,
    "Bus": 4,
    "Truck": 5,
}


@dataclass(frozen=True)
class _DawnRecord:
    image_id: str
    image_bytes: bytes
    image_suffix: str
    width: int
    height: int
    objects: tuple[dict, ...]
    source_split: str

    @property
    def class_counts(self) -> Counter:
        return Counter(DAWN_CLASS_MAPPING[item["class_name"]] for item in self.objects)

    @property
    def weather(self) -> str:
        prefix = self.image_id.lower().split("-")[0]
        return prefix.rstrip("0123456789_") or "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dawn_records(source: Path) -> tuple[list[_DawnRecord], dict[str, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency failure
        raise MLXUserError(
            "DAWN preparation requires pyarrow. Install MLX's object-detection dependencies."
        ) from exc

    files = sorted((source / "data").glob("*.parquet"))
    if not files:
        raise MLXUserError(f"Expected DAWN Parquet files under {source / 'data'}")
    records: list[_DawnRecord] = []
    checksums = {}
    seen = set()
    for path in files:
        checksums[path.name] = _sha256_file(path)
        source_split = path.name.split("-", 1)[0]
        for row in parquet.read_table(path).to_pylist():
            image_id = str(row["image_id"])
            if image_id in seen:
                raise MLXUserError(f"Duplicate DAWN image_id: {image_id}")
            seen.add(image_id)
            image = row["image"]
            suffix = Path(image.get("path") or "image.jpg").suffix.lower() or ".jpg"
            if suffix not in {".jpg", ".jpeg", ".png"}:
                raise MLXUserError(f"Unsupported image suffix for {image_id}: {suffix}")
            objects = tuple(row["objects"] or ())
            for item in objects:
                if item.get("class_name") not in DAWN_CLASS_MAPPING:
                    raise MLXUserError(
                        f"Unknown DAWN class {item.get('class_name')!r} in {image_id}"
                    )
                x, y = float(item["x_min"]), float(item["y_min"])
                width, height = float(item["width"]), float(item["height"])
                if width <= 0 or height <= 0 or x < 0 or y < 0:
                    raise MLXUserError(f"Invalid bounding box in DAWN image {image_id}")
                if x + width > row["width"] + 1e-6 or y + height > row["height"] + 1e-6:
                    raise MLXUserError(f"Out-of-bounds annotation in DAWN image {image_id}")
            records.append(
                _DawnRecord(
                    image_id=image_id,
                    image_bytes=image["bytes"],
                    image_suffix=suffix,
                    width=int(row["width"]),
                    height=int(row["height"]),
                    objects=objects,
                    source_split=source_split,
                )
            )
    return records, checksums


def _split_score(partitions: dict[str, list[_DawnRecord]], totals: Counter) -> float:
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    all_weather = Counter(record.weather for values in partitions.values() for record in values)
    score = 0.0
    for split, records in partitions.items():
        counts = sum((record.class_counts for record in records), Counter())
        weather = Counter(record.weather for record in records)
        for class_id, total in totals.items():
            score += abs(counts[class_id] / total - ratios[split])
        for condition, total in all_weather.items():
            score += 0.2 * abs(weather[condition] / total - ratios[split])
    return score


def _stratified_partitions(
    records: list[_DawnRecord], seed: int
) -> dict[str, list[_DawnRecord]]:
    """Search deterministic exact-size splits balanced by class and weather.

    DAWN does not publish sequence identifiers. Image IDs expose weather but not
    reliable capture groups, so near-frame leakage cannot be ruled out.
    """
    count = len(records)
    train_count = round(count * 0.70)
    val_count = round(count * 0.15)
    sizes = (train_count, val_count, count - train_count - val_count)
    totals = sum((record.class_counts for record in records), Counter())
    rng = random.Random(seed)
    best = None
    for _ in range(3000):
        shuffled = list(records)
        rng.shuffle(shuffled)
        partitions = {
            "train": shuffled[: sizes[0]],
            "val": shuffled[sizes[0] : sizes[0] + sizes[1]],
            "test": shuffled[sizes[0] + sizes[1] :],
        }
        if any(
            sum((record.class_counts for record in partitions[split]), Counter())[class_id] < 1
            for split in partitions
            for class_id in range(len(FOUNDATION_CLASSES))
        ):
            continue
        score = _split_score(partitions, totals)
        if best is None or score < best[0]:
            best = (score, partitions)
    if best is None:
        raise MLXUserError("Could not create DAWN splits containing every foundation class")
    return {
        split: sorted(values, key=lambda record: record.image_id)
        for split, values in best[1].items()
    }


class PrepareDawnAdapterDataset:
    """Convert immutable DAWN Parquet sources to a mapped YOLO dataset."""

    def __init__(self, source: Path, destination: Path, *, seed: int = 42):
        self.source = Path(source).expanduser().resolve()
        self.destination = Path(destination).expanduser().resolve()
        self.seed = int(seed)

    def execute(self) -> dict:
        manifest_path = self.destination / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("seed") != self.seed or manifest.get("source") != str(self.source):
                raise MLXUserError(
                    f"Existing prepared dataset differs at {self.destination}; choose a new path"
                )
            return manifest
        if self.destination.exists() and any(self.destination.iterdir()):
            raise MLXUserError(f"Dataset output directory is not empty: {self.destination}")

        records, source_checksums = _load_dawn_records(self.source)
        if len(records) < 1000:
            raise MLXUserError(f"Expected at least 1,000 DAWN images, found {len(records)}")
        partitions = _stratified_partitions(records, self.seed)
        split_stats = {}
        selected = {}
        for split, values in partitions.items():
            image_dir = self.destination / "images" / split
            label_dir = self.destination / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            object_counts = Counter()
            weather_counts = Counter()
            selected[split] = []
            for record in values:
                filename = f"{record.image_id}{record.image_suffix}"
                (image_dir / filename).write_bytes(record.image_bytes)
                lines = []
                for item in record.objects:
                    class_id = DAWN_CLASS_MAPPING[item["class_name"]]
                    object_counts[FOUNDATION_CLASSES[class_id]] += 1
                    box_width, box_height = float(item["width"]), float(item["height"])
                    center_x = (float(item["x_min"]) + box_width / 2) / record.width
                    center_y = (float(item["y_min"]) + box_height / 2) / record.height
                    lines.append(
                        f"{class_id} {center_x:.8f} {center_y:.8f} "
                        f"{box_width / record.width:.8f} {box_height / record.height:.8f}"
                    )
                (label_dir / f"{record.image_id}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                weather_counts[record.weather] += 1
                selected[split].append(record.image_id)
            split_stats[split] = {
                "images": len(values),
                "objects": sum(object_counts.values()),
                "objects_per_class": dict(object_counts),
                "weather_images": dict(weather_counts),
            }

        selection_sha256 = hashlib.sha256(
            json.dumps(selected, sort_keys=True).encode("utf-8")
        ).hexdigest()
        data = {
            "path": str(self.destination),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": list(FOUNDATION_CLASSES),
        }
        (self.destination / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "dataset": "DAWN",
            "source": str(self.source),
            "source_format": "Hugging Face Parquet conversion of DAWN v3",
            "source_url": "https://data.mendeley.com/datasets/766ygrbt8y/3",
            "license": "CC BY-NC 3.0",
            "seed": self.seed,
            "classes": list(FOUNDATION_CLASSES),
            "class_mapping": DAWN_CLASS_MAPPING,
            "source_checksums": source_checksums,
            "selection_sha256": selection_sha256,
            "splits": split_stats,
            "selected_images": selected,
            "limitations": [
                "DAWN does not publish capture-sequence identifiers; near-frame leakage cannot be excluded.",
                "Bicycle and motorcycle are rare, so per-class estimates have high uncertainty.",
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest


def load_prepared_adapter_dataset(source: Path) -> dict:
    source = Path(source).expanduser().resolve()
    manifest_path = source / "manifest.json"
    yaml_path = source / "data.yaml"
    if not manifest_path.is_file() or not yaml_path.is_file():
        raise MLXUserError(
            f"Prepared adapter dataset requires manifest.json and data.yaml at {source}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classes") != list(FOUNDATION_CLASSES):
        raise MLXUserError("Prepared dataset class order does not match the foundation taxonomy")
    for split in ("train", "val", "test"):
        if not (source / "images" / split).is_dir() or not (source / "labels" / split).is_dir():
            raise MLXUserError(f"Prepared dataset is missing {split} images or labels")
    return manifest


def prepare_adapter_dataset(source: Path, destination: Path, seed: int = 42) -> dict:
    """Compatibility wrapper for callers of the original preparation helper."""
    return PrepareDawnAdapterDataset(source, destination, seed=seed).execute()
