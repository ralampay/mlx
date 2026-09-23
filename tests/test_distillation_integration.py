"""Opt-in, offline real-provider check: MLX_DISTILLATION_SMOKE=1 pytest this file."""
import math
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('MLX_DISTILLATION_SMOKE') != '1', reason='opt-in real LibreYOLO training')


@pytest.mark.parametrize('loss_type', ['cwd', 'mgd'])
def test_real_train_resume_and_student_only_inference(tmp_path, monkeypatch, caplog, loss_type):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    yaml = pytest.importorskip('yaml')
    from PIL import Image
    from libreyolo import LibreYOLO, LibreYOLOX
    from libreyolo.distillation import Distiller
    from mlx.modes.object_detection.libreyolo.training import TrainLibreYOLOObjectDetection

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        data = tmp_path / 'data'
        for split in ('train', 'val'):
            for folder in ('images', 'labels'):
                (data / folder / split).mkdir(parents=True)
            for i in range(2):
                image = np.full((64, 64, 3), 80, dtype=np.uint8)
                image[16:48, 16:48] = 220
                Image.fromarray(image).save(data / 'images' / split / f'{i}.jpg')
                (data / 'labels' / split / f'{i}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
        (data / 'data.yaml').write_text(yaml.safe_dump({'path': str(data), 'train': 'images/train', 'val': 'images/val', 'names': ['square']}))
        teacher_path = tmp_path / 'teacher.pt'
        LibreYOLOX(None, size='n', device='cpu').save(str(teacher_path))
        captured = []
        original_init = Distiller.__init__
        def capture(self, *a, **kw):
            original_init(self, *a, **kw)
            captured.append((self, {k: v.detach().clone() for k, v in self.teacher.state_dict().items()}))
        monkeypatch.setattr(Distiller, '__init__', capture)
        original_kwargs = TrainLibreYOLOObjectDetection._build_train_kwargs
        def small_kwargs(self, **kw):
            values = original_kwargs(self, **kw)
            values.update(workers=0, patience=0, no_aug_epochs=0, mosaic_prob=0, mixup_prob=0)
            return values
        monkeypatch.setattr(TrainLibreYOLOObjectDetection, '_build_train_kwargs', small_kwargs)
        config = dict(provider='libreyolo', model='yolox-drax-mobilenet-v3-large-n',
                      dataset_path=str(data), output_path=str(tmp_path / 'run'), run_name='smoke',
                      distiller=str(teacher_path), distill_loss=loss_type, pretrained=False,
                      epochs=1, batch_size=2, nbs=2, height=64, width=64, device='cpu',
                      amp=False, warmup_epochs=0, lr0=0.001, optimizer='sgd', plots=False,
                      use_best=False, random_seed=42)
        result = TrainLibreYOLOObjectDetection(config).execute()
        assert math.isfinite(result['final_loss'])
        checkpoint = torch.load(result['checkpoint_path'], map_location='cpu', weights_only=False)
        assert 'distiller' in checkpoint
        assert any(v.get('momentum_buffer') is not None and v['momentum_buffer'].abs().sum() > 0
                   for v in checkpoint['optimizer']['state'].values())
        result = TrainLibreYOLOObjectDetection({**config, 'epochs': 2}).execute()
        resumed = torch.load(result['checkpoint_path'], map_location='cpu', weights_only=False)
        assert resumed['epoch'] == 1
        assert 'Could not load' not in caplog.text
        assert len(resumed['optimizer']['param_groups']) == len(checkpoint['optimizer']['param_groups'])
        # Warm-start an independent fine-tuning run through the same provider.
        fine = TrainLibreYOLOObjectDetection({**config, 'model_path': result['checkpoint_path'],
                   'output_path': str(tmp_path / 'fine')}).execute()
        assert math.isfinite(fine['final_loss'])
        for distiller, original in captured:
            assert all(not p.requires_grad and p.grad is None for p in distiller.teacher.parameters())
            assert all(torch.equal(original[k], v) for k, v in distiller.teacher.state_dict().items())
        student = LibreYOLO(result['checkpoint_path'], device='cpu')
        count = sum(p.numel() for p in student.model.parameters())
        deployment = tmp_path / 'student.pt'
        student.save(str(deployment))
        teacher_path.unlink()
        loaded = LibreYOLO(str(deployment), device='cpu')
        assert sum(p.numel() for p in loaded.model.parameters()) == count
        assert not any('distiller' in k or 'teacher' in k for k in loaded.model.state_dict())
        loaded.model.eval()
        with torch.no_grad():
            prediction = loaded.model(torch.zeros(1, 3, 64, 64))
        tensors, _ = torch.utils._pytree.tree_flatten(prediction)
        assert tensors and all(torch.isfinite(tensor).all() for tensor in tensors)
    finally:
        torch.set_num_threads(previous_threads)
