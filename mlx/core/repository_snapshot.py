"""Explicit source snapshots for reproducible local experiments."""
from pathlib import Path
import shutil
import subprocess

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError


class SnapshotRepositories:
    def __init__(self, repositories, destination):
        self.repositories, self.destination = repositories, Path(destination)

    def execute(self):
        if self.destination.exists():
            raise MLXUserError(f"Source snapshot already exists: {self.destination}")
        sources = {}
        for name, source in self.repositories.items():
            if Path(name).name != name or name in {'.', '..'}:
                raise MLXUserError(f"Invalid repository snapshot name: {name}")
            source = Path(source).resolve()
            destination = self.destination / name
            files = subprocess.check_output(
                ['git', '-C', str(source), 'ls-files', '-co', '--exclude-standard', '-z']
            ).decode().split('\0')
            checksums = {}
            for relative in sorted(set(filter(None, files))):
                original = source / relative
                if not original.is_file():
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                checksums[relative] = sha256_file(target)
            sources[name] = {'original': str(source), 'snapshot': str(destination), 'files': checksums,
                             'commit': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip(),
                             'status': subprocess.check_output(['git','-C',str(source),'status','--short'], text=True)}
            write_json_atomic(destination / '.mlx-source.json', {'commit': sources[name]['commit']})
        return sources
