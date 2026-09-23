"""Prepare provider arguments and preserve distillation identity on resume."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mlx.core.exceptions import MLXUserError


class PrepareDistillationRun:
    def __init__(self, config: Mapping[str, Any], run_dir: Path, resume: Path | None):
        self.config = config
        self.run_dir = run_dir
        self.resume = resume

    def execute(self) -> dict[str, Any]:
        manifest = self.run_dir / "distillation.json"
        previous = manifest
        if self.resume is not None:
            checkpoint_dir = self.resume.parent
            previous = (checkpoint_dir.parent if checkpoint_dir.name == "weights" else checkpoint_dir) / "distillation.json"
        teacher_path = self.config.get("distiller")
        if not teacher_path:
            if previous.exists():
                raise MLXUserError("This run used distillation. Supply the original --distiller and settings to resume, or choose a new output directory.")
            return {}
        teacher = Path(teacher_path).expanduser().resolve()
        digest = hashlib.sha256()
        with teacher.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        loss = self.config.get("distill_loss") or "cwd"
        weight = self.config.get("distill_weight")
        settings = {
            "distill_loss_type": loss,
            "dis": float(weight) if weight is not None else (1.0 if loss == "cwd" else 2e-5),
            "distill_tau": float(self.config.get("distill_temperature") or 1.0),
            "distill_mask_ratio": float(self.config["distill_mask_ratio"]) if self.config.get("distill_mask_ratio") is not None else 0.65,
        }
        record = {"version": 1, "teacher_sha256": digest.hexdigest(), "teacher_path": str(teacher), "settings": settings}
        if previous.exists():
            try:
                saved = json.loads(previous.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise MLXUserError(f"Cannot read distillation provenance: {previous}") from exc
            if not isinstance(saved, dict) or any(saved.get(key) != record[key] for key in ("version", "teacher_sha256", "settings")):
                raise MLXUserError("Teacher contents or distillation settings differ from this run. Use a new output directory for a new experiment.")
        elif self.resume is not None:
            raise MLXUserError("Cannot resume with distillation: distillation.json is missing. Use fine-tune with an explicit student checkpoint and a new output directory.")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest)
        return {"distill_model": str(teacher), **settings}
