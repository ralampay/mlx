"""Official NEU-DET download, validation and leakage-aware deterministic splits."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import requests
import yaml

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError


CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
SOURCE = "https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/"
DOWNLOAD = "https://drive.usercontent.google.com/download"
FILE_ID = "1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k"


def grouped_split(records, seed=42):
    """Keep identical decoded images together; stratify by source defect class."""
    groups = defaultdict(list)
    for record in records:
        groups[record["pixel_sha256"]].append(record)
    strata = defaultdict(list)
    for digest, items in sorted(groups.items()):
        classes = {item["source_class"] for item in items}
        if len(classes) != 1:
            raise MLXUserError("Duplicate pixels have conflicting source classes")
        strata[next(iter(classes))].append((digest, items))
    result = {}
    rng = random.Random(seed)
    for category, items in sorted(strata.items()):
        rng.shuffle(items)
        total = sum(len(group) for _, group in items)
        targets = {"train": total * .7, "val": total * .15, "test": total * .15}
        counts = Counter()
        for digest, group in items:
            split = max(targets, key=lambda key: targets[key] - counts[key])
            counts[split] += len(group)
            for record in group:
                result[record["stem"]] = split
    return result


class PrepareNeuDetDataset:
    def __init__(self, destination: Path, seed: int = 42):
        self.destination = Path(destination).expanduser().resolve()
        self.seed = seed

    def execute(self):
        original = self.destination / "original"
        processed = self.destination / "processed"
        if processed.exists():
            raise MLXUserError(f"Refusing to overwrite prepared data: {processed}")
        original.mkdir(parents=True, exist_ok=True)
        archive = original / "NEU-DET.zip"
        if not archive.exists():
            partial = archive.with_suffix(".download")
            try:
                with requests.get(DOWNLOAD, params={"id": FILE_ID, "export": "download", "confirm": "t"},
                                  stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                if not zipfile.is_zipfile(partial):
                    raise MLXUserError("Official NEU-DET download was not a ZIP; inspect Google Drive availability")
                partial.rename(archive)
            except requests.RequestException as exc:
                raise MLXUserError(f"Cannot download official NEU-DET: {exc}") from exc
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (original / member.filename).resolve()
                if not target.is_relative_to(original) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise MLXUserError(f"Unsafe ZIP member: {member.filename}")
            source.extractall(original)
        records = self._inspect(original)
        assignment = grouped_split(records, self.seed)
        processed.mkdir()
        statistics = {}
        for split in ("train", "val", "test"):
            images = processed / "images" / split
            labels = processed / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            coco = {"images": [], "annotations": [], "categories": [
                {"id": i, "name": name} for i, name in enumerate(CLASSES)]}
            objects = Counter()
            for record in records:
                if assignment[record["stem"]] != split:
                    continue
                name = record["stem"] + ".jpg"
                shutil.copy2(record["image"], images / name)
                image_id = len(coco["images"]) + 1
                width, height = record["size"]
                coco["images"].append({"id": image_id, "file_name": name, "width": width, "height": height})
                lines = []
                for category, x, y, w, h in record["boxes"]:
                    objects[CLASSES[category]] += 1
                    lines.append(f"{category} {(x+w/2)/width:.10f} {(y+h/2)/height:.10f} {w/width:.10f} {h/height:.10f}")
                    coco["annotations"].append({"id": len(coco["annotations"])+1, "image_id": image_id,
                        "category_id": category, "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0})
                (labels / (record["stem"] + ".txt")).write_text("\n".join(lines) + "\n")
            write_json_atomic(processed / f"{split}.json", coco)
            statistics[split] = {"images": len(coco["images"]), "objects": sum(objects.values()), "objects_per_class": dict(objects)}
        (processed / "data.yaml").write_text(yaml.safe_dump({"path": str(processed), "train": "images/train",
            "val": "images/val", "test": "images/test", "nc": len(CLASSES), "names": list(CLASSES)}))
        selection = [{k: v for k, v in record.items() if k != "image"} | {"split": assignment[record["stem"]]}
                     for record in records]
        manifest = {"dataset": "NEU-DET", "source_url": SOURCE, "download_url": DOWNLOAD,
            "source_file_id": FILE_ID, "archive_sha256": sha256_file(archive), "seed": self.seed,
            "classes": list(CLASSES), "splits": statistics, "selected_images": selection,
            "selection_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
            "coordinate_policy": "VOC 1-based inclusive boxes converted to zero-based xywh",
            "duplicate_groups": len(records)-len({r['pixel_sha256'] for r in records}),
            "limitations": ["No published acquisition-sequence identifiers; exact-pixel duplicates are grouped, but near-duplicate/sequence leakage cannot be excluded.",
                "Custom seed-42 split, not a claimed official benchmark split.", "Source page does not provide an explicit redistribution license; local research use only."]}
        write_json_atomic(processed / "manifest.json", manifest)
        return manifest

    def _inspect(self, original):
        images = {}
        for path in sorted(original.rglob("*.jpg")):
            if path.stem in images:
                raise MLXUserError(f"Duplicate image filename: {path.stem}")
            images[path.stem] = path
        records = []
        seen = set()
        for path in sorted(original.rglob("*.xml")):
            if path.stem in seen or path.stem not in images:
                raise MLXUserError(f"Unmatched or duplicated annotation: {path}")
            seen.add(path.stem)
            root = ET.parse(path).getroot()
            # The official release omits the extension in some XML filename fields.
            if Path(root.findtext("filename") or "").stem != images[path.stem].stem:
                raise MLXUserError(f"XML filename does not match image: {path}")
            with Image.open(images[path.stem]) as image:
                image.load()
                width, height = image.size
                digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if (int(root.findtext("size/width")), int(root.findtext("size/height"))) != (width, height):
                raise MLXUserError(f"Image/XML size mismatch: {path}")
            boxes = []
            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name not in CLASSES:
                    raise MLXUserError(f"Unknown NEU class {name!r} in {path}")
                x1,y1,x2,y2 = [int(obj.findtext(f"bndbox/{key}")) for key in ("xmin","ymin","xmax","ymax")]
                if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
                    raise MLXUserError(f"Invalid VOC box in {path}: {(x1,y1,x2,y2)}")
                boxes.append([CLASSES.index(name), x1-1, y1-1, x2-x1+1, y2-y1+1])
            category = path.stem.rsplit("_", 1)[0]
            if category not in CLASSES or not boxes:
                raise MLXUserError(f"Invalid source category or empty annotation: {path}")
            records.append({"stem": path.stem, "image": str(images[path.stem]), "source_class": category,
                "pixel_sha256": digest, "size": [width,height], "boxes": boxes})
        if seen != set(images) or len(records) != 1800:
            raise MLXUserError(f"Expected 1800 paired detection records, found {len(records)} XML and {len(images)} images")
        if Counter(record["source_class"] for record in records) != Counter({name:300 for name in CLASSES}):
            raise MLXUserError("Expected 300 NEU-DET images per source defect category")
        return records
