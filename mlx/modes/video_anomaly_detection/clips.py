from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipWindow:
    source: str
    frame_paths: tuple[str, ...]
    frame_indices: tuple[int, ...]
    start_frame: int
    end_frame: int
    ground_truth: int


def window_start_indices(
    frame_count: int,
    *,
    clip_length: int,
    frame_stride: int,
    window_stride: int = 1,
) -> tuple[int, ...]:
    span = (clip_length - 1) * frame_stride + 1
    if frame_count < span:
        return ()
    return tuple(range(0, frame_count - span + 1, window_stride))


def aggregate_frame_scores(
    records: list[dict],
    *,
    method: str = "mean",
) -> list[dict]:
    if method not in {"mean", "max"}:
        raise ValueError("Frame aggregation must be 'mean' or 'max'.")
    covered: dict[tuple[str, int], dict[str, list[float] | list[int]]] = {}
    for record in records:
        # A temporal window covers its full start/end span even when the model
        # samples only every Nth frame inside that span.
        frame_indices = record.get("frame_indices") or ()
        start_frame = int(
            record["start_frame"] if "start_frame" in record else min(frame_indices)
        )
        end_frame = int(
            record["end_frame"] if "end_frame" in record else max(frame_indices)
        )
        for frame_index in range(start_frame, end_frame + 1):
            item = covered.setdefault(
                (str(record["source"]), int(frame_index)),
                {"scores": [], "labels": [], "thresholds": []},
            )
            item["scores"].append(float(record["anomaly_score"]))
            item["thresholds"].append(float(record["threshold"]))
            if record.get("ground_truth") is not None:
                item["labels"].append(int(record["ground_truth"]))

    rows = []
    for (source, frame_index), values in sorted(covered.items()):
        scores = values["scores"]
        score = sum(scores) / len(scores) if method == "mean" else max(scores)
        labels = values["labels"]
        thresholds = values["thresholds"]
        threshold = float(thresholds[0])
        if any(value != threshold for value in thresholds[1:]):
            raise ValueError("All windows in one frame aggregation must use the same threshold.")
        rows.append(
            {
                "source": source,
                "frame": frame_index,
                "ground_truth": max(labels) if labels else None,
                "anomaly_score": score,
                "threshold": threshold,
                "predicted_anomaly": int(score > threshold),
                "covering_windows": len(scores),
            }
        )
    return rows


__all__ = ["ClipWindow", "aggregate_frame_scores", "window_start_indices"]
