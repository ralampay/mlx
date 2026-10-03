#!/usr/bin/env python3
"""POSIX process supervision for resumable local experiment scripts."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_state(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


class RunExperimentJob:
    def __init__(self, output, command, directory=None):
        self.output = Path(output).expanduser().resolve()
        self.command = command
        self.directory = Path(directory) if directory else None

    def execute(self):
        launches = Path(str(self.output) + '.launches')
        launches.mkdir(parents=True, exist_ok=True)
        with (launches / 'job.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'An experiment already holds the output lock: {self.output}', file=sys.stderr)
                return 73
            state = {'pid': os.getpid(), 'output': str(self.output), 'command': self.command,
                     'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
            if self.directory:
                write_state(self.directory / 'status.json', state)
                (self.directory / 'job.pid').write_text(str(os.getpid()) + '\n')
            child = None
            def stop(signum, frame):
                if child is not None:
                    child.send_signal(signum)
            previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                # The child also retains the lock if this supervisor is killed abruptly.
                child = subprocess.Popen(self.command, pass_fds=(lock.fileno(),))
                state['child_pid'] = child.pid
                if self.directory:
                    write_state(self.directory / 'status.json', state)
                code = child.wait()
            except OSError as exc:
                print(f'Unable to start experiment: {exc}', file=sys.stderr)
                state['error'] = str(exc)
                code = 127
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            state.update(exit_code=code, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if self.directory:
                write_state(self.directory / 'status.json', state)
            return code if code >= 0 else 128 - code


class StartExperimentJob:
    def __init__(self, output, command):
        self.output, self.command = output, command

    def execute(self):
        root = Path(str(Path(self.output).expanduser().resolve()) + '.launches')
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        directory = root / f'{stamp}-{os.getpid()}'
        directory.mkdir()
        log = directory / 'output.log'
        worker = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output', self.output,
                  '--directory', str(directory), '--', *self.command]
        with log.open('w') as target:
            process = subprocess.Popen(['nohup', *worker], stdin=subprocess.DEVNULL, stdout=target,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            if process.poll() is not None:
                print(f'Job exited during startup ({process.returncode}). Log: {log}', file=sys.stderr)
                return process.returncode
            if (directory / 'status.json').exists():
                state = json.loads((directory / 'status.json').read_text())
                if 'child_pid' in state:
                    print(json.dumps({'pid': process.pid, 'log': str(log), 'status': str(directory / 'status.json'),
                                      'output': self.output}, indent=2))
                    return 0
            time.sleep(.05)
        print(f'Worker PID {process.pid}; startup not confirmed. Inspect {log}', file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'start', 'worker'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--directory')
    arguments, command = parser.parse_known_args()
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('Provide an experiment command after --.')
    if arguments.action == 'start':
        return StartExperimentJob(arguments.output, command).execute()
    return RunExperimentJob(arguments.output, command, arguments.directory).execute()


if __name__ == '__main__':
    sys.exit(main())
