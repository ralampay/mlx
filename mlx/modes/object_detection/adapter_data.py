"""Deterministic, source-preserving split of a prepared YOLO detection set."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import yaml

from mlx.core.exceptions import MLXUserError


def prepare_adapter_dataset(source: Path, destination: Path, seed: int = 42) -> dict:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    yaml_path = source / "data.yaml"
    if not yaml_path.is_file():
        raise MLXUserError(f"Expected YOLO data.yaml at {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text())
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if not isinstance(names, list) or not names:
        raise MLXUserError("Dataset must declare an ordered, nonempty class mapping")
    split_dirs = {split: source / str(config.get(split, "images/test"))
                  for split in ("train", "val", "test")}
    split_dirs = {split: path.resolve() for split, path in split_dirs.items()}
    distinct_paths = len(set(split_dirs.values()))
    if distinct_paths not in {1, 3}:
        raise MLXUserError("YOLO train, val and test paths must be all distinct or point to one source pool")
    existing_splits = distinct_paths == 3
    images = sorted(split_dirs["test"].glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if existing_splits:
        partitions = {split: sorted((p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                                    key=lambda p: p.name)
                      for split, folder in split_dirs.items()}
        images = [p for paths in partitions.values() for p in paths]
    if len(images) < 30:
        raise MLXUserError("Adapter split requires at least 30 labeled source images")
    if existing_splits:
        if any(not paths for paths in partitions.values()):
            raise MLXUserError("Existing YOLO train, val and test splits must all contain images")
        if len({p.resolve() for p in images}) != len(images):
            raise MLXUserError("Existing YOLO splits share image files; use disjoint splits")
    else:
        # ACDC frames share video sequences. Keep whole sequences in one split.
        groups = {}
        for image in images:
            groups.setdefault(image.stem.split("_frame_")[0], []).append(image)
        if len(groups) < 3:
            raise MLXUserError("Need at least three source sequences for independent splits")
        group_counts = {}
        for group, members in groups.items():
            tally = Counter()
            for image in members:
                label = source / "labels" / "test" / f"{image.stem}.txt"
                if not label.is_file():
                    raise MLXUserError(f"Missing label for {image}: {label}")
                tally.update(int(line.split()[0]) for line in label.read_text().splitlines() if line.strip())
            group_counts[group] = tally
        rng = random.Random(seed)
        best = None
        for _ in range(2000):
            keys = list(groups)
            rng.shuffle(keys)
            candidate = {"train": [], "val": [], "test": []}
            for key in keys:
                sizes = {split: sum(len(groups[k]) for k in chosen) for split, chosen in candidate.items()}
                targets = {"train": .7, "val": .15, "test": .15}
                split = max(targets, key=lambda s: targets[s] * len(images) - sizes[s])
                candidate[split].append(key)
            split_counts = {split: sum((group_counts[k] for k in keys), Counter())
                            for split, keys in candidate.items()}
            if any(any(split_counts[s][class_id] < 3 for class_id in range(len(names)))
                   for s in ("val", "test")):
                continue
            score = sum(abs(sum(len(groups[k]) for k in candidate[s]) / len(images) - target)
                        for s, target in (("train", .7), ("val", .15), ("test", .15)))
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is None:
            raise MLXUserError("Could not create a sequence-disjoint split covering all classes")
        partitions = {split: sorted((image for key in keys for image in groups[key]), key=lambda p: p.name)
                      for split, keys in best[1].items()}
    selected = {split: [p.name for p in paths] for split, paths in partitions.items()}
    identity = hashlib.sha256(json.dumps({"source": str(source), "seed": seed, "selected": selected},
                                     sort_keys=True).encode()).hexdigest()
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("selection_sha256") != identity:
            raise MLXUserError(f"Existing dataset split differs at {destination}; choose a new output directory")
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise MLXUserError(f"Dataset output directory is not empty: {destination}")
    counts = {}
    for split, paths in partitions.items():
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
        tally = Counter()
        for image in paths:
            label = source / "labels" / (split if existing_splits else "test") / f"{image.stem}.txt"
            if not label.is_file():
                raise MLXUserError(f"Missing label for {image}: {label}")
            for line in label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5 or not fields[0].isdigit() or int(fields[0]) >= len(names):
                    raise MLXUserError(f"Invalid YOLO annotation in {label}: {line}")
                tally[names[int(fields[0])]] += 1
            (destination / "images" / split / image.name).symlink_to(image)
            (destination / "labels" / split / label.name).symlink_to(label)
        counts[split] = {"images": len(paths), "instances": dict(tally)}
    output_yaml = {"path": str(destination), "train": "images/train", "val": "images/val",
                   "test": "images/test", "names": names}
    (destination / "data.yaml").write_text(yaml.safe_dump(output_yaml, sort_keys=False))
    manifest = {"source": str(source), "source_images": len(images), "seed": seed,
                "selection_sha256": identity, "classes": names,
                "class_mapping": {str(i): i for i in range(len(names))},
                "splits": counts, "selected_images": selected}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
