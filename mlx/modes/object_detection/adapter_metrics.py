"""Fixed-operating-point detection precision/recall for adapter experiments."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch


def _iou(box, others: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], others[:, :2])
    bottom_right = torch.minimum(box[2:], others[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    box_area = (box[2:] - box[:2]).clamp(min=0).prod()
    other_area = (others[:, 2:] - others[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (box_area + other_area - intersection).clamp(min=1e-9)


def measure_precision_recall(model, dataset: Path, *, image_size: int,
                             confidence: float = 0.25, match_iou: float = 0.5) -> dict[str, float]:
    """Class-aware greedy matching at a declared score and IoU threshold."""
    true_positive = false_positive = false_negative = 0
    for image in sorted((dataset / "images" / "test").iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(image) as picture:
            width, height = picture.size
        label = dataset / "labels" / "test" / f"{image.stem}.txt"
        ground_truth = []
        for line in label.read_text().splitlines():
            cls, cx, cy, bw, bh = (float(value) for value in line.split())
            ground_truth.append((int(cls), [width * (cx - bw / 2), height * (cy - bh / 2),
                                            width * (cx + bw / 2), height * (cy + bh / 2)]))
        results = model.predict(source=str(image), conf=confidence, iou=0.6,
                                imgsz=image_size, save=False)
        result = results[0] if isinstance(results, list) else results
        boxes = result.boxes
        gt_classes = torch.tensor([row[0] for row in ground_truth], dtype=torch.long)
        gt_boxes = torch.tensor([row[1] for row in ground_truth], dtype=torch.float32).reshape(-1, 4)
        matched = set()
        if boxes is not None and len(boxes.xyxy):
            pred_boxes = torch.as_tensor(boxes.xyxy).detach().cpu().float()
            pred_classes = torch.as_tensor(boxes.cls).detach().cpu().long()
            scores = torch.as_tensor(boxes.conf).detach().cpu().float()
            for index in scores.argsort(descending=True).tolist():
                candidates = [j for j in range(len(ground_truth))
                              if j not in matched and gt_classes[j] == pred_classes[index]]
                if not candidates:
                    false_positive += 1
                    continue
                overlaps = _iou(pred_boxes[index], gt_boxes[candidates])
                best = int(overlaps.argmax())
                if float(overlaps[best]) >= match_iou:
                    matched.add(candidates[best])
                    true_positive += 1
                else:
                    false_positive += 1
        false_negative += len(ground_truth) - len(matched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {"precision": precision, "recall": recall,
            "true_positives": true_positive, "false_positives": false_positive,
            "false_negatives": false_negative, "precision_confidence": confidence,
            "precision_iou": match_iou}
