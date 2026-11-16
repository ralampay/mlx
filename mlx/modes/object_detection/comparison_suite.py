"""Budgeted, immutable multi-dataset scratch comparisons."""
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import time
import subprocess
import re
from contextlib import contextmanager

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from .comparison_experiment import CSPDraxComparisonRequest


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise MLXUserError(f'Cannot read comparison JSON {path}: {exc}') from exc


@contextmanager
def suite_lock(root):
    import fcntl
    with (root/'suite.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MLXUserError('Another process owns this experiment queue.') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def balanced_schedule(models, seeds):
    orders = tuple(itertools.permutations(models))
    return [(seed, model) for i, seed in enumerate(seeds) for model in orders[i % len(orders)]]


def include_snapshot_file(file, root):
    excluded = ('env', '.git', '__pycache__', '.pytest_cache', '.venv')
    if any(part in excluded for part in file.relative_to(root).parts):
        return False
    return file.suffix in ('.py', '.md', '.toml', '.txt', '.yaml', '.json', '.cff') or file.name in (
        'PKG-INFO', 'METADATA', 'LICENSE', 'NOTICE', 'MANIFEST.in')


def admit_datasets(estimates, budget_seconds):
    admitted, used = [], 0.0
    for name, seconds in estimates:
        if used + seconds * 1.25 > budget_seconds:
            break
        admitted.append(name)
        used += seconds * 1.25
    if not admitted:
        raise MLXUserError('First dataset exceeds remaining runtime budget; no main runs started.')
    return admitted, used


class RunComparisonSuite:
    def __init__(self, config, backend, reporter):
        self.config = config
        self.root = Path(config['output']).expanduser().resolve()
        self.backend, self.reporter = backend, reporter

    def execute(self, action):
        if action == 'status':
            return read_json(self.root/'status.json')
        if action == 'prepare':
            return self._prepare()
        manifest = self._verify()
        if action == 'analyze':
            return self.reporter(self.root)
        if action not in ('pilot', 'run'):
            raise MLXUserError(f'Unknown comparison action: {action}')
        self.backend.verify_device()
        with suite_lock(self.root):
            try:
                return self._pilot(manifest) if action == 'pilot' else self._run(manifest)
            except Exception as exc:
                self._status('failed', error=str(exc))
                raise

    def _status(self, status, **values):
        write_json_atomic(self.root/'status.json', dict(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **values))

    def _prepare(self):
        config = self.config
        models, seeds = config['models'], config['seeds']
        if config['epochs'] < 1 or config['budget_hours'] <= 0 or config['image_size'] < 32 or config['image_size'] % 32:
            raise MLXUserError('Epochs/budget must be positive and image size a positive multiple of 32.')
        if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) for name in [*models,*config['datasets']]):
            raise MLXUserError('Model/dataset identifiers must be simple names, not paths.')
        if not 2 <= len(models) <= 3 or len(set(models)) != len(models):
            raise MLXUserError('Supply two or three distinct model identifiers.')
        if not 2 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds):
            raise MLXUserError('Supply 2–20 unique paired seeds.')
        if self.root.exists():
            raise MLXUserError(f'Experiment output already exists: {self.root}')
        environment = self.backend.verify_device()
        files = {}
        for name, dataset in config['datasets'].items():
            directory = Path(dataset)
            audit = read_json(directory/'audit.json')
            if audit['status'] != 'audited':
                raise MLXUserError(f'{name}: unresolved dataset audit at {directory}')
            for file in sorted(directory.rglob('*')):
                if file.is_file():
                    files[str(file)] = sha256_file(file)
        sources, revisions = {}, {}
        for name, directory in config['sources'].items():
            directory = Path(directory)
            revisions[name] = subprocess.run(['git','-C',str(directory),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
            files_to_copy = [p for p in directory.rglob('*') if p.is_file() and include_snapshot_file(p, directory)]
            for file in files_to_copy:
                destination = self.root/'source'/name/file.relative_to(directory)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, destination)
                sources[str(file)] = sha256_file(file)
        manifest = dict(config=config, dataset_files=files, source_files=sources, environment=environment, revisions=revisions)
        write_json_atomic(self.root/'manifest.json', manifest)
        self._status('prepared')
        return manifest

    def _verify(self):
        manifest = read_json(self.root/'manifest.json')
        if manifest['config'] != self.config:
            raise MLXUserError('Configuration differs from prepared experiment.')
        if (self.root/'protocol-sha256.json').exists():
            if read_json(self.root/'protocol-sha256.json')['sha256'] != sha256_file(self.root/'protocol.json'):
                raise MLXUserError('Frozen statistical protocol was changed.')
        for key in ('dataset_files', 'source_files'):
            for file, digest in manifest[key].items():
                if not Path(file).is_file() or sha256_file(file) != digest:
                    raise MLXUserError(f'Experiment input changed: {file}')
        return manifest

    def _request(self, dataset, batch):
        return CSPDraxComparisonRequest(dataset=Path(self.config['datasets'][dataset]), output=self.root,
            source_checkpoint=Path('.'), scratch=True, seeds=tuple(self.config['seeds']),
            epochs=self.config['epochs'], image_size=self.config['image_size'], batch_size=batch,
            learning_rate=.001, device='cuda', workers=0, equivalence_margin=.01)

    def _condition(self, dataset, model, seed, batch, pilot):
        directory = self.root/('pilot' if pilot else 'runs')/dataset/model/f'seed-{seed}'
        complete = directory/'complete.json'
        identity = sha256_file(self.root/'manifest.json')
        if complete.exists():
            record = read_json(complete)
            if record.get('identity') != identity or record['checkpoint_sha256'] != sha256_file(record['checkpoint']):
                raise MLXUserError(f'Completed result identity changed: {directory}')
            return record
        if directory.exists():
            raise MLXUserError(f'Interrupted run preserved at {directory}; archive it before restarting the same seed.')
        if shutil.disk_usage(self.root).free < 3*1024**3:
            raise MLXUserError('Less than 3 GiB disk headroom; training not started.')
        self._status('pilot' if pilot else 'training', dataset=dataset, model=model, seed=seed)
        record = self.backend.run(self._request(dataset,batch), model, seed, directory, pilot=pilot)
        record.update(dataset=dataset, identity=identity, checkpoint_sha256=sha256_file(record['checkpoint']))
        write_json_atomic(complete, record)
        return record

    def _pilot(self, manifest):
        if (self.root/'protocol.json').exists():
            return read_json(self.root/'protocol.json')
        started = time.perf_counter()
        calibration_path = self.root/'calibration.json'
        if calibration_path.exists():
            calibration = read_json(calibration_path)
        else:
            calibration = self.backend.calibrate(self.config['models'], self.config['image_size'])
            write_json_atomic(calibration_path, calibration)
        batch = calibration['physical_batch']
        estimates, pilot_seconds = [], 0.0
        for dataset in self.config['datasets']:
            rows = [self._condition(dataset, model, 300, batch, True) for model in self.config['models']]
            pilot_seconds += sum(r['training_seconds']+r['evaluation_seconds'] for r in rows)
            # Test inference is not scored during pilots. Scale validation evaluation
            # time by test/validation counts as a conservative workload estimate.
            counts = read_json(Path(self.config['datasets'][dataset])/'audit.json')['counts']
            ratio = counts['test']['images']/counts['val']['images']
            seconds = sum(r['seconds_per_epoch']*self.config['epochs'] + r['evaluation_seconds']*max(1,ratio) for r in rows)*len(self.config['seeds'])
            estimates.append((dataset, seconds))
        remaining = self.config['budget_hours']*3600 - max(time.perf_counter()-started, pilot_seconds)
        selected, seconds = admit_datasets(estimates, remaining)
        # Conservative persistent-checkpoint estimate using observed pilot disk use.
        pilot_bytes = sum(p.stat().st_size for p in (self.root/'pilot').rglob('*') if p.is_file())
        needed = pilot_bytes / len(self.config['datasets']) * len(selected)*len(self.config['seeds']) * 1.25
        if shutil.disk_usage(self.root).free < needed + 3*1024**3:
            raise MLXUserError(f'Insufficient storage: require approximately {needed/1024**3:.1f} GiB plus headroom; pilot preserved.')
        protocol = dict(datasets=selected, models=self.config['models'], seeds=self.config['seeds'],
                        physical_batch=batch, effective_batch=8, epochs=self.config['epochs'],
                        estimates_seconds=dict(estimates), estimated_main_hours_with_contingency=seconds/3600,
                        equivalence_margin_pp=1.0, alpha=.05, family='all pairwise superiority and equivalence claims across admitted datasets',
                        frozen_at=datetime.now(timezone.utc).isoformat())
        write_json_atomic(self.root/'protocol.json', protocol)
        write_json_atomic(self.root/'protocol-sha256.json', {'sha256':sha256_file(self.root/'protocol.json')})
        self._status('ready', protocol=protocol)
        return protocol

    def _run(self, manifest):
        protocol = read_json(self.root/'protocol.json')
        for dataset in protocol['datasets']:
            for seed, model in balanced_schedule(protocol['models'], protocol['seeds']):
                self._condition(dataset, model, seed, protocol['physical_batch'], False)
                self.reporter(self.root)
        result = self.reporter(self.root)
        self._status('completed')
        return result
