"""Runtime and source provenance for LibreYOLO zero-shot evaluation."""
import hashlib
import importlib.metadata
import platform
import json
import tarfile
from pathlib import Path
import subprocess
from mlx.core.artifacts import sha256_file, write_json_atomic


class CollectTransferProvenance:
    def __init__(self, output, device):
        self.output = Path(output)
        self.device = device

    def execute(self):
        import torch
        import libreyolo

        path = self.output / "environment.json"
        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(self.device),
            "vram_bytes": torch.cuda.get_device_properties(
                self.device
            ).total_memory,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
            "amp": False,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "packages": sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
            ),
        }
        if not path.exists():
            write_json_atomic(path, environment)
        sources = self.output / "source"
        sources.mkdir(exist_ok=True)
        records = []
        for name, root in (
            ("mlx", Path(__file__).resolve().parents[4]),
            ("libreyolo", Path(libreyolo.__file__).resolve().parents[1]),
        ):
            if not (root / ".git").exists():
                distribution = importlib.metadata.distribution(name)
                metadata = {"version": distribution.version,
                            "source": json.loads(distribution.read_text("direct_url.json") or "{}")}
                records.append(self._snapshot_package(name, root / name, sources, metadata))
                continue
            diff = subprocess.check_output(
                ["git", "-C", str(root), "diff", "HEAD", "--binary"]
            )
            untracked = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
                text=True,
            ).splitlines()
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            fingerprint = hashlib.sha256(
                diff
                + commit.encode()
                + "".join(
                    filename + sha256_file(root / filename)
                    for filename in untracked
                    if (root / filename).is_file()
                ).encode()
            ).hexdigest()
            destination = sources / name / fingerprint
            destination.mkdir(parents=True, exist_ok=True)
            archive = destination / "source.tar"
            if not archive.exists():
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(root),
                        "archive",
                        "--format=tar",
                        "HEAD",
                        "-o",
                        str(archive),
                    ],
                    check=True,
                )
                (destination / "changes.patch").write_bytes(diff)
                # Include untracked implementation files absent from git archive/diff.
                with tarfile.open(destination / "untracked.tar", "w") as tar:
                    for filename in untracked:
                        tar.add(root / filename, arcname=filename)
                write_json_atomic(
                    destination / "manifest.json",
                    {
                        "commit": commit,
                        "archive_sha256": sha256_file(archive),
                        "patch_sha256": sha256_file(destination / "changes.patch"),
                        "untracked_sha256": sha256_file(destination / "untracked.tar"),
                    },
                )
            records.append(
                {
                    "repository": name,
                    "revision": fingerprint,
                    "directory": str(destination.relative_to(self.output)),
                }
            )
        self.source_revision = records
        mode = Path(__file__).resolve().parent.parent
        relevant = [
            *(mode / "zero_shot").glob("*.py"),
            Path(__file__),
            mode / "libreyolo/zero_shot_backend.py",
            mode / "libreyolo/adapter_slice_backend.py",
            mode / "libreyolo/adapter_backend.py",
        ]
        self.evaluation_signature = {
            str(p.relative_to(mode)): sha256_file(p) for p in relevant
        }
        self.evaluation_signature["libreyolo"] = records[1]["revision"]

        return {"source_revision": self.source_revision, "evaluation_signature": self.evaluation_signature}

    def _snapshot_package(self, name, root, destination, metadata):
        """Snapshot installed package files without walking the containing environment."""
        files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        identity = {"files": files, **metadata}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = destination / name / fingerprint
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "source.tar"
        if not (target / "manifest.json").is_file():
            with tarfile.open(archive, "w") as stream:
                for relative in files:
                    stream.add(root / relative, arcname=relative)
            write_json_atomic(target / "manifest.json", {**identity, "archive_sha256": sha256_file(archive)})
        return {"repository": name, "revision": fingerprint,
                "directory": str(target.relative_to(self.output))}
