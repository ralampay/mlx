"""CSV parsing, temporal partitioning and training-only scaling."""

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mlx.core.exceptions import MLXUserError

RESERVED = {"synthetic_series_id", "step", "elapsed_time"}


def file_digest(path):
    digest = hashlib.sha256()
    try:
        with Path(path).expanduser().open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, TypeError) as exc:
        raise MLXUserError(f"Cannot read required file '{path}': {exc}") from exc
    return digest.hexdigest()


def read_rows(path):
    if not path:
        raise MLXUserError("Provide an input CSV path.")
    try:
        with Path(path).expanduser().open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            names = reader.fieldnames
            if not names or len(names) != len(set(names)):
                raise MLXUserError("CSV must have unique column names.")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MLXUserError(f"Cannot read CSV '{path}': {exc}") from exc
    if not rows:
        raise MLXUserError("CSV contains no observations.")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise MLXUserError(
            "CSV rows must have the same number of fields as the header."
        )
    return names, rows


def parse_time(value):
    try:
        number = float(value)
        if not np.isfinite(number):
            raise MLXUserError("Timestamps must be finite.")
        return number, "numeric"
    except ValueError:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            # Naive ISO timestamps are explicitly interpreted as UTC.
            return dt.replace(
                tzinfo=timezone.utc
            ).timestamp() if dt.tzinfo is None else dt.timestamp(), "seconds"
        except (ValueError, OverflowError) as exc:
            raise MLXUserError(
                f"Invalid timestamp '{value}'; use numeric or ISO-8601 values."
            ) from exc


@dataclass
class PreparedData:
    partitions: dict
    metadata: dict

    def windows(self, split, stride=None, standardized=True):
        length = self.metadata["window_size"]
        stride = stride or self.metadata["stride"]
        windows, groups = [], []
        for group, values in self.partitions[split].items():
            for start in range(0, len(values) - length + 1, stride):
                windows.append(values[start : start + length])
                groups.append(group)
        result = np.asarray(windows, dtype=np.float32)
        if standardized:
            result = (result - self.metadata["mean"]) / self.metadata["scale"]
        if not np.isfinite(result).all():
            raise MLXUserError("Preprocessed values exceed supported numeric range.")
        return result.astype(np.float32), np.asarray(groups)


class TemporalCsvLoader:
    def load(
        self,
        path,
        *,
        time_column,
        feature_columns,
        series_id_column=None,
        window_size=128,
        stride=32,
        expected=None,
    ):
        names, rows = read_rows(path)
        features = list(feature_columns)
        if not time_column or not features or len(features) != len(set(features)):
            raise MLXUserError("Provide time-column and distinct feature-columns.")
        selected = [time_column, *features] + (
            [series_id_column] if series_id_column else []
        )
        if len(selected) != len(set(selected)) or any(c not in names for c in selected):
            raise MLXUserError(
                "Time, feature and series-ID columns must be distinct and present in the CSV."
            )
        if RESERVED.intersection(features):
            raise MLXUserError(
                f"Feature names cannot use reserved export columns: {sorted(RESERVED)}."
            )
        groups, units = {}, set()
        for index, row in enumerate(rows, 2):
            group = row[series_id_column] if series_id_column else "series"
            if not group.strip():
                raise MLXUserError(f"Missing series ID at CSV row {index}.")
            timestamp, unit = parse_time(row[time_column])
            units.add(unit)
            try:
                values = [float(row[name]) for name in features]
            except ValueError as exc:
                raise MLXUserError(
                    f"Non-numeric or missing feature at CSV row {index}."
                ) from exc
            if not np.isfinite(values).all():
                raise MLXUserError(f"Non-finite feature at CSV row {index}.")
            groups.setdefault(group, []).append((timestamp, values))
        if len(units) != 1:
            raise MLXUserError("Use one timestamp format throughout the CSV.")
        partitions = {name: {} for name in ("train", "val", "test")}
        boundaries, interval = {}, None
        for group, records in sorted(groups.items()):
            records.sort(key=lambda pair: pair[0])
            times = np.array([r[0] for r in records])
            differences = np.diff(times)
            if len(differences) == 0 or np.any(differences <= 0):
                raise MLXUserError(
                    f"Series '{group}' requires distinct increasing timestamps."
                )
            step = float(np.median(differences))
            time_tolerance = max(1e-8, float(np.spacing(np.abs(times).max())) * 4)
            if not np.allclose(differences, step, rtol=1e-5, atol=time_tolerance):
                raise MLXUserError(
                    f"Series '{group}' is irregularly sampled; resample it explicitly before training."
                )
            if interval is not None and not np.isclose(
                step, interval, rtol=1e-5, atol=time_tolerance
            ):
                raise MLXUserError("All series must use the same sampling interval.")
            interval = step
            values = np.array([r[1] for r in records], dtype=np.float64)
            a, b = int(len(values) * 0.7), int(len(values) * 0.85)
            for name, start, stop in (
                ("train", 0, a),
                ("val", a, b),
                ("test", b, len(values)),
            ):
                if stop - start < window_size:
                    raise MLXUserError(
                        f"Series '{group}' {name} partition has {stop - start} rows; needs window-size {window_size}. Supply more rows or reduce window-size."
                    )
                partitions[name][group] = values[start:stop]
            boundaries[group] = {
                "rows": len(values),
                "train_end": a,
                "val_end": b,
                "first_time": float(times[0]),
                "last_time": float(times[-1]),
            }
        training = np.concatenate(list(partitions["train"].values()))
        mean, scale = training.mean(0), training.std(0)
        scale[scale == 0] = 1
        metadata = {
            "time_column": time_column,
            "feature_columns": features,
            "series_id_column": series_id_column,
            "window_size": window_size,
            "stride": stride,
            "sampling_interval": interval,
            "time_unit": units.pop(),
            "splits": boundaries,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "source_sha256": file_digest(path),
        }
        if expected is not None and metadata != expected:
            raise MLXUserError(
                "Reference CSV or preprocessing differs from the training checkpoint. Use the original training CSV."
            )
        return PreparedData(partitions, metadata)


def load_synthetic(path, metadata):
    names, rows = read_rows(path)
    required = [
        "synthetic_series_id",
        "step",
        "elapsed_time",
        *metadata["feature_columns"],
    ]
    if names != required:
        raise MLXUserError(f"Synthetic CSV columns must be {required} in that order.")
    groups = {}
    try:
        for row in rows:
            group = row["synthetic_series_id"]
            if not group:
                raise ValueError("empty synthetic series ID")
            groups.setdefault(group, []).append(
                (
                    int(row["step"]),
                    float(row["elapsed_time"]),
                    [float(row[c]) for c in metadata["feature_columns"]],
                )
            )
        arrays = []
        for records in groups.values():
            records.sort()
            if [r[0] for r in records] != list(range(metadata["window_size"])):
                raise ValueError(
                    "each synthetic sequence must have exactly the trained window length and unique steps"
                )
            if not np.allclose(
                [r[1] for r in records],
                np.arange(len(records)) * metadata["sampling_interval"],
            ):
                raise ValueError(
                    "elapsed_time does not match the trained sampling interval"
                )
            arrays.append([r[2] for r in records])
        result = np.asarray(arrays, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("non-finite synthetic values")
    except (ValueError, OverflowError) as exc:
        raise MLXUserError(f"Invalid synthetic CSV: {exc}.") from exc
    return result
