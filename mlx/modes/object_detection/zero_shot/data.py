"""Native COCO study inputs and deterministic metadata strata."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import random
import re

from mlx.core.artifacts import json_safe, sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise MLXUserError(f"Cannot read study JSON {path}: {exc}") from exc


def metadata(filename, kind):
    if kind == "acdc":
        parts = Path(filename).parts
        if len(parts) != 4:
            raise MLXUserError(f"Invalid ACDC filename: {filename}")
        return {
            "group": parts[0],
            "sequence": f"{parts[0]}/{parts[2]}",
            "frame": int(re.search(r"frame_(\d+)", parts[3]).group(1)),
        }
    if kind == "mrtmd":
        match = re.fullmatch(r"(video\d+)_frame_(\d+)\.jpg", filename)
        if not match:
            raise MLXUserError(f"Invalid MRTMD filename: {filename}")
        return {"group": match[1], "sequence": match[1], "frame": int(match[2])}
    return {"group": "all", "sequence": "all", "frame": 0}


def stratified_ids(images, count=128, seed=42):
    groups = defaultdict(list)
    for image in sorted(images, key=lambda x: x["id"]):
        groups[image["sequence"]].append(image["id"])
    rng = random.Random(seed)
    for ids in groups.values():
        rng.shuffle(ids)
    result = []
    while len(result) < min(count, len(images)):
        for key in sorted(groups):
            if groups[key] and len(result) < count:
                result.append(groups[key].pop())
    return result


class InspectTransferDataset:
    def __init__(self, spec, classes):
        self.spec, self.classes = spec, classes

    def execute(self):
        spec = self.spec
        annotations = Path(spec["annotations"]).expanduser().resolve()
        source = read_json(annotations)
        categories = {c["id"]: c["name"] for c in source["categories"]}
        if categories != {int(k): v for k, v in self.classes.items()}:
            raise MLXUserError(
                f"Native category IDs/names do not match foundation: {annotations}"
            )
        root = Path(spec["images"]).expanduser().resolve()
        images, hashes = [], []
        for item in source["images"]:
            path = (root / item["file_name"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise MLXUserError(f"Missing or unsafe dataset image: {path}")
            images.append(
                {
                    **item,
                    **metadata(item["file_name"], spec.get("metadata")),
                    "path": str(path),
                }
            )
            hashes.append({"id": item["id"], "sha256": sha256_file(path)})
        if len({i["id"] for i in images}) != len(images):
            raise MLXUserError("Duplicate native COCO image IDs")
        if len({a["id"] for a in source["annotations"]}) != len(source["annotations"]):
            raise MLXUserError("Duplicate native COCO annotation IDs")
        return {
            **spec,
            "annotations": str(annotations),
            "images_root": str(root),
            "annotation_sha256": sha256_file(annotations),
            "yaml_sha256": sha256_file(Path(spec["yaml"]).expanduser()),
            "image_manifest": hashes,
            "images": images,
            "latency_image_ids": stratified_ids(images),
            "objects_per_class": dict(
                Counter(a["category_id"] for a in source["annotations"])
            ),
            "crowd_annotations": sum(
                a.get("iscrowd", 0) for a in source["annotations"]
            ),
            "groups": dict(Counter(i["group"] for i in images)),
        }


def preserve_identity(path, payload):
    path = Path(path)
    payload = json_safe(payload)
    if path.exists() and read_json(path) != payload:
        raise MLXUserError(f"Study inputs changed; use a new output directory: {path}")
    write_json_atomic(path, payload)
