import json
from types import SimpleNamespace

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection import adapter_queue as queue
from mlx.modes.object_detection.adapter_transfer import RunQueuedTransferStudy


@pytest.mark.parametrize("executable,ignored", [("/usr/bin/nautilus", True), ("/tmp/nautilus", False), (None, False)])
def test_nautilus_requires_verified_executable(monkeypatch, executable, ignored):
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="999, /usr/bin/nautilus\n888, python\n"))
    def readlink(path):
        if executable is None:
            raise PermissionError(path)
        return executable
    monkeypatch.setattr(queue.os, "readlink", readlink)
    result = queue.active_cuda_jobs()
    assert {r["pid"] for r in result} == ({888} if ignored else {888, 999})


def test_existing_clients_and_self_ignored(monkeypatch):
    monkeypatch.setattr(queue.os, "getpid", lambda: 123)
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="123, python\n456, /usr/bin/ptyxis\n789, /tmp/nautilus\n\n"))
    assert queue.active_cuda_jobs() == [{"pid": 789, "name": "/tmp/nautilus"}]


def test_completed_study_reused_without_execution_or_writes(tmp_path, monkeypatch):
    config = tmp_path / "plan.json"
    config.write_text(json.dumps({"output": str(tmp_path), "methods": ["lora"], "seeds": [1]}))
    (tmp_path / "status.json").write_text('{"status":"completed"}')
    run = tmp_path / "lora/seed-1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"status":"completed"}')
    for name in ("predictions.json", "inference.json", "scores.json"):
        (run / name).write_text('{}')
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    verified = []
    monkeypatch.setattr(RunQueuedTransferStudy, "_verify", lambda c: verified.append(c))
    monkeypatch.setattr(RunQueuedTransferStudy, "execute", lambda self: pytest.fail("Must not replay completed study"))
    queue.ReuseCompletedTransferStudy(config, resume=True).execute()
    assert verified
    assert before == {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    (run / "scores.json").unlink()
    with pytest.raises(MLXUserError, match="missing evaluation"):
        queue.ReuseCompletedTransferStudy(config, resume=True).execute()


def test_waiting_study_delegates_with_explicit_resume(tmp_path, monkeypatch):
    config = tmp_path / "plan.json"
    config.write_text(json.dumps({"output": str(tmp_path)}))
    (tmp_path / "status.json").write_text('{"status":"queued"}')
    calls = []
    monkeypatch.setattr(RunQueuedTransferStudy, "execute", lambda self: calls.append((self.path, self.resume)))
    queue.ReuseCompletedTransferStudy(config, resume=True).execute()
    assert calls == [(config, True)]
