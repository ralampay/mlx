"""Run the MLX DAWN command, then restore explicitly held local dispatchers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError


def matching_process(item):
    root = Path('/proc') / str(item['pid'])
    try:
        # Fields after the parenthesized command start with state (field 3).
        ticks = (root / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        command = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return ticks == item['start_ticks'] and item['token'] in command
    except FileNotFoundError:
        return False


def stopped_process(item):
    if not matching_process(item):
        return False
    state = (Path('/proc') / str(item['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    return state in {'T', 't'}


class SupervisePriority:
    def __init__(self, config_path, held_path):
        self.config_path, self.held_path = Path(config_path), Path(held_path)

    def execute(self):
        config = json.loads(self.config_path.read_text())
        held = json.loads(self.held_path.read_text())
        child = None
        def stop(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            if not all(matching_process(p) for p in held['processes']):
                raise MLXUserError('Held queue identity changed; inspect processes before continuing.')
            suspended = held.get('mode') == 'suspended-in-memory'
            if suspended and not all(stopped_process(p) for p in held['processes']):
                raise MLXUserError('A held process is not stopped; priority training was not started.')
            while not suspended:
                state = json.loads(Path(held['active_condition_status']).read_text())
                if state.get('status') == 'completed':
                    break
                if state.get('status') == 'failed':
                    raise MLXUserError('Previous DIOR condition failed; inspect its artifacts.')
                if not Path(f"/proc/{held['active_condition_pid']}").exists():
                    raise MLXUserError('Previous training process disappeared without completion.')
                time.sleep(15)
            from mlx.modes.object_detection.adapter_queue import active_cuda_jobs
            # Completed stage may still be tearing down its CUDA context.
            for _ in range(20):
                jobs = active_cuda_jobs()
                allowed = {p['pid'] for p in held['processes'] if suspended and stopped_process(p)}
                if not [job for job in jobs if job['pid'] not in allowed]:
                    break
                time.sleep(3)
            else:
                raise MLXUserError('CUDA is occupied; DAWN was not started.')
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      '--config', str(self.config_path), '--train'])
            code = child.wait()
            if code:
                raise MLXUserError(f'DAWN subprocess exited with code {code}; original queues will resume.')
            return 0
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            restored = []
            for item in held['processes']:
                valid = matching_process(item)
                if valid:
                    os.kill(item['pid'], signal.SIGCONT)
                restored.append({**item, 'resumed': valid,
                                 'restored_at': datetime.now(timezone.utc).isoformat(),
                                 'timing_caveat': 'An interrupted wall-clock timing region includes suspension; exclude or remeasure it.'
                                 if held.get('mode') == 'suspended-in-memory' else None})
            write_json_atomic(Path(config['output']) / 'queue-restoration.json', restored)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--held')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.train:
        config = json.loads(Path(args.config).read_text())
        if config.get('study_kind') == 'dawn-paired-overall':
            from mlx.modes.object_detection.dawn_paired import RunPairedDawnStudy
            RunPairedDawnStudy(config).execute()
        else:
            from mlx.modes.object_detection.dawn_priority import RunDawnPriorityStudy
            RunDawnPriorityStudy(config).execute()
        return 0
    if not args.held:
        parser.error('--held is required for the supervisor')
    return SupervisePriority(args.config, args.held).execute()


if __name__ == '__main__':
    sys.exit(main())
