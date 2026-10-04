"""COCO matching once, with crowd-aware threshold metrics and cached slice accumulation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path

import numpy as np

from mlx.core.artifacts import json_safe, write_json_atomic
from .data import read_json


def fixed_counts(records, image_ids, confidence=0.25):
    counts = {int(i): {"tp": 0, "fp": 0, "fn": 0, "ignored": 0} for i in image_ids}
    for record in records:
        if (
            record is None
            or record["aRng"] != [0, 1e10]
            or record["image_id"] not in counts
        ):
            continue
        scores = np.asarray(record["dtScores"])
        matches = np.asarray(record["dtMatches"])[0]
        ignored = np.asarray(record["dtIgnore"])[0].astype(bool)
        selected = scores >= confidence
        tp = int(np.sum(selected & ~ignored & (matches > 0)))
        row = counts[record["image_id"]]
        row["tp"] += tp
        row["fp"] += int(np.sum(selected & ~ignored & (matches == 0)))
        row["fn"] += int(np.sum(~np.asarray(record["gtIgnore"]).astype(bool))) - tp
        row["ignored"] += int(np.sum(selected & ignored))
    for row in counts.values():
        row.update(prf(row))
    return counts


def prf(row):
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


class ScoreTransferPredictions:
    def __init__(self, dataset, predictions, output, image_ids=None):
        self.dataset, self.predictions, self.output = (
            dataset,
            Path(predictions),
            Path(output),
        )
        self.image_ids = image_ids

    def execute(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with redirect_stdout(io.StringIO()):
            gt = COCO(self.dataset["annotations"])
            predictions = read_json(self.predictions)
            if predictions:
                dt = gt.loadRes(predictions)
            else:
                dt = COCO()
                dt.dataset = {**deepcopy(gt.dataset), "annotations": []}
                dt.createIndex()
            evaluator = COCOeval(gt, dt, "bbox")
            ids = sorted(self.image_ids if self.image_ids is not None else gt.imgs)
            evaluator.params.imgIds = ids
            evaluator.evaluate()
            # JSON, not pickle: cached matches are independently inspectable and safe to load.
            with gzip.open(self.output / "coco-matches.json.gz", "wt") as stream:
                json.dump(json_safe(evaluator.evalImgs), stream, separators=(",", ":"))
            per_image = fixed_counts(evaluator.evalImgs, ids)
            full_records = evaluator.evalImgs
            full_params = deepcopy(evaluator._paramsEval)
            image_positions = {image_id: n for n, image_id in enumerate(ids)}
            slices = {"all": ids}
            selected = set(ids)
            for field in ("group", "sequence"):
                for value in sorted({i[field] for i in self.dataset["images"]}):
                    slices[f"{field}:{value}"] = [
                        i["id"]
                        for i in self.dataset["images"]
                        if i[field] == value and i["id"] in selected
                    ]
            if self.dataset.get("metadata") == "mrtmd":
                for value in self.dataset["groups"]:
                    slices[f"without:{value}"] = [
                        i["id"]
                        for i in self.dataset["images"]
                        if i["group"] != value and i["id"] in selected
                    ]
            results = {}
            for name, slice_ids in slices.items():
                if not slice_ids:
                    continue
                # pycocotools accumulate indexes positions, not image IDs. Reindex
                # the cached K×A×I records before changing the image subset.
                slice_ids = sorted(slice_ids)
                evaluator.evalImgs = [
                    full_records[offset + image_positions[i]]
                    for offset in range(0, len(full_records), len(ids))
                    for i in slice_ids
                ]
                evaluator._paramsEval = deepcopy(full_params)
                evaluator._paramsEval.imgIds = slice_ids
                evaluator.params = deepcopy(evaluator._paramsEval)
                evaluator.accumulate()
                evaluator.summarize()
                stats = evaluator.stats
                row = dict(
                    zip(
                        (
                            "mAP50_95",
                            "mAP50",
                            "mAP75",
                            "APsmall",
                            "APmedium",
                            "APlarge",
                            "AR1",
                            "AR10",
                            "AR100",
                            "ARsmall",
                            "ARmedium",
                            "ARlarge",
                        ),
                        stats,
                    )
                )
                # -1 denotes no evaluable GT, not zero performance.
                row = {k: float(v) if v >= 0 else None for k, v in row.items()}
                counts = {
                    key: sum(per_image[i][key] for i in slice_ids)
                    for key in ("tp", "fp", "fn", "ignored")
                }
                precision = evaluator.eval["precision"][:, :, :, 0, -1]
                class_ap = {}
                for n, cat in enumerate(evaluator.params.catIds):
                    values = precision[:, :, n]
                    valid = values[values >= 0]
                    class_ap[gt.cats[cat]["name"]] = (
                        float(valid.mean()) if valid.size else None
                    )
                results[name] = {
                    **row,
                    **counts,
                    **prf(counts),
                    "images": len(slice_ids),
                    "fp_per_image": counts["fp"] / len(slice_ids),
                    "per_class_AP": class_ap,
                    "noncrowd_objects": counts["tp"] + counts["fn"],
                }
        write_json_atomic(self.output / "per-image.json", per_image)
        write_json_atomic(self.output / "scores.json", results)
        return results
