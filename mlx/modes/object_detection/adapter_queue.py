"""CUDA queue checks and read-only reuse of completed transfer studies."""

import json
import os
from pathlib import Path
import subprocess

from mlx.core.exceptions import MLXUserError


def active_cuda_jobs():
    """Keep unknown clients blocking; verify Nautilus through procfs, not its name."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    jobs = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid_text, name = line.split(",", 1)
        pid, name = int(pid_text), name.strip()
        if pid == os.getpid() or Path(name).name in {"ptyxis", "gnome-shell", "Xorg"}:
            continue
        if name == "/usr/bin/nautilus":
            try:
                if os.readlink(f"/proc/{pid}/exe") == name:
                    continue
            except OSError:
                pass  # Unverified processes remain blocking.
        jobs.append({"pid": pid, "name": name})
    return jobs


class ReuseCompletedTransferStudy:
    """Resume a queue without rewriting already-completed dataset artifacts."""

    def __init__(self, configuration, *, resume=False):
        self.configuration, self.resume = Path(configuration), resume

    def execute(self):
        from mlx.modes.object_detection.adapter_transfer import RunQueuedTransferStudy

        config = json.loads(self.configuration.read_text())
        output = Path(config["output"])
        status = output / "status.json"
        if self.resume and status.exists() and json.loads(status.read_text()).get("status") == "completed":
            RunQueuedTransferStudy._verify(config)
            conditions = config.get("conditions") or [{"id": m} for m in config["methods"]]
            for condition in conditions:
                for seed in config["seeds"]:
                    run = output / condition["id"] / f"seed-{seed}"
                    metrics = run / "metrics.json"
                    if not metrics.is_file() or json.loads(metrics.read_text()).get("status") != "completed":
                        raise MLXUserError(f"Completed study has missing or incomplete metrics: {run}")
                    if not all((run / name).is_file() for name in ("predictions.json", "inference.json", "scores.json")):
                        raise MLXUserError(f"Completed study has missing evaluation artifacts: {run}")
            return
        return RunQueuedTransferStudy(self.configuration, resume=self.resume).execute()
