"""Versioned checkpoints and staged, non-overwriting artifact directories."""

import csv
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from mlx.core.exceptions import MLXUserError

from .data import RESERVED
from .requests import finite_number, positive_int

FORMAT = "mlx-temporal-vae-v1"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


@contextmanager
def output_directory(path):
    if not path:
        raise MLXUserError("Provide --output for the artifact directory.")
    target = Path(path).expanduser()
    if target.exists():
        raise MLXUserError(f"Output '{target}' already exists. Choose a new directory.")
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        yield temporary
        # Reserve the final name exclusively before publishing files.
        target.mkdir()
        try:
            for item in temporary.iterdir():
                os.replace(item, target / item.name)
        except OSError:
            shutil.rmtree(target)
            raise
    except OSError as exc:
        raise MLXUserError(
            f"Cannot write temporal artifacts to '{target}': {exc}"
        ) from exc
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


class TemporalCheckpointStore:
    def save(self, path, payload):
        import torch

        torch.save(payload, path)

    def load(self, path, registry):
        import torch

        if not path:
            raise MLXUserError(
                "Provide --model-path for a trained temporal checkpoint."
            )
        try:
            payload = torch.load(
                Path(path).expanduser(), map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise MLXUserError(
                f"Cannot safely load temporal checkpoint '{path}': {exc}"
            ) from exc
        try:
            if not isinstance(payload, dict) or payload["format"] != FORMAT:
                raise ValueError("unsupported checkpoint format")
            config, meta = payload["config"], payload["preprocessing"]
            if not isinstance(config, dict) or not isinstance(meta, dict):
                raise TypeError("configuration and preprocessing must be mappings")
            name = payload["model"]
            if (
                name not in registry.entries
                or payload["definition"] != registry.entries[name]
            ):
                raise ValueError(
                    "model definition is not trusted by the caller's registry"
                )
            for key in ("window_size", "features", "latent_dim"):
                positive_int(config[key], key, 4 if key == "window_size" else 1)
            if (
                not isinstance(config["hidden_dims"], (list, tuple))
                or not config["hidden_dims"]
            ):
                raise ValueError("empty hidden dimensions")
            for width in config["hidden_dims"]:
                positive_int(width, "hidden dimension")
            if not isinstance(config["model_config"], dict):
                raise TypeError("invalid model options")
            features = meta["feature_columns"]
            if (
                not isinstance(features, list)
                or not features
                or any(not isinstance(f, str) or not f for f in features)
                or len(set(features)) != len(features)
                or len(features) != config["features"]
                or meta["window_size"] != config["window_size"]
            ):
                raise ValueError("inconsistent feature/window metadata")
            for key in ("mean", "scale"):
                values = np.asarray(meta[key], dtype=float)
                if values.shape != (len(features),) or not np.isfinite(values).all():
                    raise ValueError(f"invalid {key}")
            finite_number(meta["sampling_interval"], "sampling_interval", strict=True)
            if np.any(np.asarray(meta["scale"]) <= 0):
                raise ValueError("invalid scale or sampling interval")
            positive_int(meta["stride"], "stride")
            if (
                not isinstance(meta["source_sha256"], str)
                or len(meta["source_sha256"]) != 64
            ):
                raise ValueError("invalid input provenance")
            if (
                not isinstance(meta["time_column"], str)
                or not meta["time_column"]
                or not isinstance(meta["splits"], dict)
                or meta["time_unit"] not in ("seconds", "numeric")
            ):
                raise ValueError("invalid temporal metadata")
            series_id = meta["series_id_column"]
            if series_id is not None and (
                not isinstance(series_id, str) or not series_id
            ):
                raise ValueError("invalid series-ID column")
            columns = [meta["time_column"], *features] + (
                [series_id] if series_id else []
            )
            if len(columns) != len(set(columns)) or RESERVED.intersection(features):
                raise ValueError("invalid or overlapping column names")
            state = payload["state_dict"]
            if (
                not isinstance(state, dict)
                or not state
                or any(
                    not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                    for v in state.values()
                )
            ):
                raise ValueError("invalid model weights")
        except (KeyError, TypeError, ValueError, MLXUserError) as exc:
            raise MLXUserError(f"Invalid temporal checkpoint '{path}': {exc}.") from exc
        return payload


class SyntheticCsvWriter:
    """Validate, inverse-scale, and stream independent generated sequences."""

    def write(self, path, batches, metadata, expected_count):
        count = 0
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "synthetic_series_id",
                    "step",
                    "elapsed_time",
                    *metadata["feature_columns"],
                ]
            )
            for batch in batches:
                batch = np.asarray(batch, dtype=np.float64)
                if (
                    batch.ndim != 3
                    or batch.shape[1:]
                    != (metadata["window_size"], len(metadata["feature_columns"]))
                    or not np.isfinite(batch).all()
                ):
                    raise MLXUserError(
                        "Generator returned invalid shapes or non-finite values."
                    )
                batch = batch * metadata["scale"] + metadata["mean"]
                if not np.isfinite(batch).all():
                    raise MLXUserError(
                        "Generated values overflowed during inverse scaling."
                    )
                for sequence in batch:
                    for step, values in enumerate(sequence):
                        writer.writerow(
                            [
                                f"synthetic_{count:06d}",
                                step,
                                step * metadata["sampling_interval"],
                                *values,
                            ]
                        )
                    count += 1
            if count != expected_count:
                raise MLXUserError(
                    "Generator returned a different sequence count than requested."
                )
        return count
