"""Freeze regularization settings using only held-out document reconstruction."""
from copy import deepcopy
from pathlib import Path
import math
import tempfile

import numpy as np

from mlx.core.artifacts import write_json_atomic, sha256_file
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.experiment_artifacts import read_json, file_hashes
from mlx.modes.text_embedding.experiment_config import load_experiment_config
from mlx.modes.text_embedding.configured_statistics import validate_cells


class SelectAutoencoderExperimentSettings:
    def __init__(self, input_path, output_path):
        self.input = Path(input_path).expanduser()
        self.output = Path(output_path).expanduser()

    def execute(self):
        config, pilot, scores = self._load_pilot()
        selected, evidence = self._select(config, scores)
        output = self._confirmation(config, pilot, selected, evidence)
        self._publish(output)
        return {"config": str(self.output / "confirmation.json"), "selected": selected}

    def _load_pilot(self):
        identity = read_json(self.input / "experiment.json")
        pilot = read_json(self.input / "pilot/results.json")
        state = read_json(self.input / "pilot/stage.json")
        if state.get("hashes") != file_hashes(self.input / "pilot"):
            raise MLXUserError("Pilot results changed or are incomplete.")
        config_path = identity.get("config", {}).get("experiment_config")
        if not config_path or not Path(config_path).expanduser().is_file() or sha256_file(Path(config_path).expanduser()) != identity.get("experiment_config_sha256"):
            raise MLXUserError("Pilot config is missing or changed; preserve the original pilot config.")
        config = load_experiment_config(config_path)
        if config.phase != "pilot":
            raise MLXUserError("Settings selection requires a completed pilot experiment.")
        validate_cells(config, pilot["cells"])
        scores = {}
        verified = set()
        for cell in pilot["cells"]:
            directory = self.input / cell["training"]
            if directory not in verified and read_json(directory / "stage.json").get("hashes") != file_hashes(directory):
                raise MLXUserError(f"Pilot training stage changed: {directory}.")
            verified.add(directory)
            rec = cell["reconstruction"]
            if not isinstance(rec, (float, int)) or not math.isfinite(rec) or rec <= 0:
                raise MLXUserError("Every pilot cell must have finite positive held-out reconstruction MSE.")
            scores[cell["dataset"], cell["variant"], cell["dimension"], cell["seed"]] = rec
        return config, pilot, scores

    def _select(self, config, scores):
        selected, evidence = {}, {}
        for loss, control in (("mse-least-volume", "spectral-mse"), ("mse-covariance", "mse")):
            candidates = []
            for variant in config.variants:
                if variant.loss != loss:
                    continue
                ratios = []
                for (dataset, name, dimension, seed), value in scores.items():
                    if name == variant.name:
                        key = (dataset, control, dimension, seed)
                        if key not in scores:
                            raise MLXUserError(f"Missing matched pilot control {key}.")
                        ratios.append(value / scores[key])
                if not ratios:
                    raise MLXUserError(f"No completed pilot cells for {variant.name}.")
                candidates.append({"name": variant.name, "rho": variant.loss_config["rho"],
                                   "mean_reconstruction_ratio": float(np.mean(ratios))})
            if not candidates:
                raise MLXUserError(f"Pilot has no candidates for {loss}.")
            best = min(x["mean_reconstruction_ratio"] for x in candidates)
            winner = min((x for x in candidates if x["mean_reconstruction_ratio"] <= best + 1e-8), key=lambda x: x["rho"])
            selected[loss] = winner["rho"]
            evidence[loss] = {"selected": winner, "candidates": candidates}
        return selected, evidence

    def _confirmation(self, config, pilot, selected, evidence):
        from mlx.modes.text_embedding.experiment_templates import confirmation_config
        output = confirmation_config(selected)
        runtime = pilot["runtime"]
        output["training"] = {key: runtime[key] for key in output["training"]}
        # Carry the actual pilot architectures/options into the frozen recipe.
        pilot_variants = {v.name: v for v in config.variants}
        for variant in output["variants"]:
            name = variant["name"]
            source_name = name
            if name in ("least-volume", "covariance"):
                loss = variant["loss"]
                source_name = evidence[loss]["selected"]["name"]
            if source_name in pilot_variants:
                source = pilot_variants[source_name]
                variant.update(model=source.model, loss=source.loss, dimensions=list(source.dimensions),
                               model_config=deepcopy(source.model_config), loss_config=deepcopy(source.loss_config))
                if source.evaluation_dimensions:
                    variant["evaluation_dimensions"] = list(source.evaluation_dimensions)
            elif name in ("mse-similarity", "truncate"):
                variant["dimensions"] = list(pilot_variants["mse"].dimensions)
            elif name == "pca":
                variant.update(dimensions=[max(pilot_variants["mse"].dimensions)],
                               evaluation_dimensions=list(pilot_variants["mse"].dimensions))
        output["selection"] = {"pilot_identity_sha256": sha256_file(self.input / "experiment.json"),
                               "pilot_results_sha256": sha256_file(self.input / "pilot/results.json"),
                               "criterion": "mean matched-control reconstruction ratio; no test retrieval scores",
                               "evidence": evidence}
        return output

    def _publish(self, output):
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise MLXUserError("Selection output must be empty; use a new --output directory.")
        try:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".ae-selection-", dir=self.output.parent) as temporary:
                pending = Path(temporary) / "selection"
                write_json_atomic(pending / "confirmation.json", output)
                write_json_atomic(pending / "selection.json", output["selection"])
                if self.output.exists():
                    self.output.rmdir()  # Only an empty destination can be replaced.
                pending.rename(self.output)
        except OSError as exc:
            raise MLXUserError(f"Unable to publish selection to {self.output}: {exc}") from exc
