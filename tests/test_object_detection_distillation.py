from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mlx.cli import build_parser
from mlx.cli_config import build_runtime_config
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.distillation import validate_distillation_options
from mlx.modes.object_detection.libreyolo.distillation import PrepareDistillationRun
from mlx.modes.object_detection.libreyolo.training import TrainLibreYOLOObjectDetection
from mlx.modes.object_detection.requests import TrainObjectDetectionRequest


@pytest.fixture
def teacher(tmp_path):
    path = tmp_path / 'teacher.pt'
    path.write_bytes(b'teacher weights')
    return path


def test_cli_request_roundtrip(teacher):
    config = build_runtime_config(build_parser().parse_args([
        '--mode', 'object-detection', '--provider', 'libreyolo', '--action', 'train',
        '--distiller', str(teacher), '--distill-loss', 'cwd', '--distill-weight', '0.5',
    ]))
    request = TrainObjectDetectionRequest.from_config(config)
    assert request.distiller == str(teacher)
    assert request.to_config()['distill_weight'] == 0.5


@pytest.mark.parametrize('change', [
    {'provider': 'ultralytics'}, {'platform': 'aws'}, {'action': 'benchmark'},
    {'mode': 'segmentation'}, {'distill_weight': float('nan')},
    {'distill_temperature': 0}, {'distill_mask_ratio': 1, 'distill_loss': 'mgd'},
    {'distill_loss': 'bad'}, {'distill_mask_ratio': 0.2},
    {'distill_loss': 'mgd', 'distill_temperature': 1},
])
def test_invalid_options(teacher, change):
    with pytest.raises(MLXUserError):
        validate_distillation_options({'provider': 'libreyolo', 'distiller': str(teacher), **change})


def test_missing_teacher():
    with pytest.raises(MLXUserError, match='require --distiller'):
        validate_distillation_options({'provider': 'libreyolo', 'distill_loss': 'cwd'})


def test_resume_identity_and_disabled_distillation(teacher, tmp_path):
    run = tmp_path / 'run'
    config = {'distiller': str(teacher)}
    result = PrepareDistillationRun(config, run, None).execute()
    assert result['distill_loss_type'] == 'cwd'
    checkpoint = run / 'weights' / 'last.pt'
    assert PrepareDistillationRun(config, run, checkpoint).execute() == result
    with pytest.raises(MLXUserError, match='used distillation'):
        PrepareDistillationRun({}, run, checkpoint).execute()
    with pytest.raises(MLXUserError, match='differ'):
        PrepareDistillationRun({**config, 'distill_weight': 2}, run, checkpoint).execute()
    teacher.write_bytes(b'changed weights')
    with pytest.raises(MLXUserError, match='differ'):
        PrepareDistillationRun(config, run, checkpoint).execute()
    (run / 'distillation.json').unlink()
    with pytest.raises(MLXUserError, match='missing'):
        PrepareDistillationRun(config, run, checkpoint).execute()


@pytest.mark.parametrize('loss', ['cwd', 'mgd'])
@pytest.mark.parametrize('warm_start', [False, True])
def test_training_forwarding(monkeypatch, teacher, tmp_path, loss, warm_start):
    calls = []
    class Model:
        def train(self, **kwargs):
            calls.append(kwargs)
            return {}
    monkeypatch.setitem(sys.modules, 'libreyolo', SimpleNamespace(LibreYOLO=lambda *a, **k: Model()))
    monkeypatch.setattr('mlx.modes.object_detection.libreyolo.training.build_scratch_model', lambda *a, **k: Model())
    data = tmp_path / 'dataset'
    data.mkdir()
    (data / 'data.yaml').write_text('names: [person]\n')
    config = {'provider': 'libreyolo', 'model': 'yolox-drax-mobilenet-v3-large-l',
              'dataset_path': str(data), 'output_path': str(tmp_path / 'runs'),
              'distiller': str(teacher), 'distill_loss': loss, 'pretrained': False}
    if warm_start:
        config['model_path'] = str(teacher)
    TrainLibreYOLOObjectDetection(config).execute()
    assert calls[0]['distill_model'] == str(teacher)
    assert calls[0]['distill_loss_type'] == loss
    assert calls[0]['dis'] == (1 if loss == 'cwd' else 2e-5)
    assert calls[0]['resume'] is False
    assert ('pretrained' in calls[0]) is not warm_start


def test_script_modes_and_failure_status(teacher, tmp_path):
    root = Path(__file__).resolve().parents[1]
    capture = tmp_path / 'args.json'
    fake = tmp_path / 'fake python'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\nsys.exit(int(os.environ.get("FAKE_EXIT",0)))\n')
    fake.chmod(0o755)
    out = tmp_path / 'output with spaces'
    env = {**os.environ, 'PYTHON': str(fake), 'CAPTURE': str(capture),
           'TEACHER_CHECKPOINT': str(teacher), 'OUTPUT_DIR': str(out)}
    def run(mode, *options, **overrides):
        return subprocess.run(['bash', str(root / 'train-foundational-drax-distillation-local.sh'), mode, *options],
                              env={**env, **overrides}, cwd=tmp_path, capture_output=True, text=True)
    assert run('scratch').returncode == 0
    args = json.loads(capture.read_text())
    assert '--no-pretrained' in args and '--model-path' not in args
    assert args[args.index('--output') + 1] == str(out)
    assert run('scratch', FAKE_EXIT='7').returncode == 7
    assert run('fine-tune').returncode == 2
    assert run('fine-tune', STUDENT_CHECKPOINT=str(teacher)).returncode == 0
    assert '--model-path' in json.loads(capture.read_text())
    checkpoint = out / 'foundational-urban-drax-cwd' / 'weights' / 'last.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    assert run('scratch').returncode == 2
    assert run('resume').returncode == 0
    assert run('scratch', '--help', TEACHER_CHECKPOINT='/missing.pt').returncode == 0
    assert run('scratch', '--model').returncode == 2
    assert run('scratch', '--unknown').returncode == 2
    local = tmp_path / 'custom data'
    local.mkdir()
    assert run('scratch', '--model', 'yolox-s', '--teacher', 'teacher.pt',
               '--dataset', 'custom data', '--output', 'custom output',
               '--distill-loss', 'mgd', '--distill-mask-ratio', '0.3',
               '--epochs', '3', '--batch-size', '2', '--height', '320', '--width', '512',
               '--device', 'cpu', '--no-amp', '--no-validate-after-training', EPOCHS='99').returncode == 0
    custom = json.loads(capture.read_text())
    for option, value in [('--model', 'yolox-s'), ('--dataset', 'custom data'),
                          ('--output', 'custom output'), ('--epochs', '3'),
                          ('--height', '320'), ('--width', '512'), ('--distill-mask-ratio', '0.3')]:
        assert custom[custom.index(option) + 1] == value
    assert '--dataset-s3-uri' not in custom and '--distill-temperature' not in custom
    assert '--distill-weight' not in custom  # Keep MGD's provider default, not CWD's weight.
    assert '--no-amp' in custom and '--no-validate-after-training' in custom
    assert (tmp_path / 'custom output' / 'training.log').exists()
    assert run('scratch', '--output', 's3 output', '--dataset-s3-uri', 's3://other/data.zip',
               STUDENT_MODEL='yolo9-s').returncode == 0
    custom = json.loads(capture.read_text())
    assert custom[custom.index('--model') + 1] == 'yolo9-s'
    assert custom[custom.index('--dataset-s3-uri') + 1] == 's3://other/data.zip'
    assert run('scratch', '--output', 'unused', '--dataset', 'custom data',
               '--dataset-s3-uri', 's3://other/data.zip').returncode == 2
    assert run('scratch', '--output', 'unused', '--student-checkpoint', str(teacher)).returncode == 2


def test_cli_default_action_and_user_error(monkeypatch, teacher):
    from mlx.cli import main
    seen = []
    monkeypatch.setattr('mlx.cli._resolve_mode_runner', lambda _: lambda config: seen.append(config))
    assert main(['--mode', 'object_detection', '--provider', 'libreyolo', '--distiller', str(teacher)]) == 0
    assert seen[0]['action'] == 'train'
    assert main(['--mode', 'object_detection', '--provider', 'libreyolo', '--distiller', '/missing/teacher.pt', '--no-verbose']) == 1


def test_aws_yaml_rejects_distillation(tmp_path, teacher):
    from mlx.modes.object_detection.aws.config import load_aws_training_config
    config = tmp_path / 'aws.yaml'
    config.write_text(f'training:\n  provider: libreyolo\n  distiller: {teacher}\n')
    with pytest.raises(MLXUserError, match='only local'):
        load_aws_training_config(str(config), {})
