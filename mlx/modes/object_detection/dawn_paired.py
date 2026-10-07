"""Fixed-budget, same-taxonomy DAWN comparisons using existing training commands."""
from dataclasses import replace
from datetime import datetime, timezone
import gc
from pathlib import Path

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest, RunAdapterExperiment
from mlx.modes.object_detection.libreyolo.adapter_backend import CalibrateAdapterBatchSize
from mlx.modes.object_detection.dawn_paired_report import GeneratePairedDawnReport


class RunPairedDawnStudy:
    def __init__(self, config, *, experiment=RunAdapterExperiment,
                 calibrator=CalibrateAdapterBatchSize, reporter=GeneratePairedDawnReport):
        self.config = config
        self.experiment, self.calibrator, self.reporter = experiment, calibrator, reporter

    def execute(self):
        c = self.config
        root = Path(c['output'])
        if (root / 'status.json').exists():
            raise MLXUserError('Study already has state; inspect before restarting. No runs overwritten.')
        self._verify()
        requests = self._requests()
        try:
            for label, request in requests:
                self._state('calibrating', condition=label)
                result = self.calibrator(
                    request.model, request.checkpoint, root/'calibration'/label,
                    device=request.device, image_size=request.image_size, amp=request.amp,
                    reduction=request.reduction, maximum_batch=request.batch_size,
                    profiles=request.methods, rank=request.rank, alpha=request.alpha,
                    target=request.target).execute()
                if result['selected_physical_batch_size'] != request.batch_size:
                    raise MLXUserError(f'{label}: batch 8 failed calibration; fixed protocol not changed.')
                self._release_cache()
            for label, request in requests:
                self._state('smoke-testing', condition=label)
                self.experiment(replace(request, output=root/'smoke'/label,
                                        seeds=(9001,), epochs=1)).execute()
                self._release_cache()
            self._check_protocol(requests)
            for index, (label, request) in enumerate(requests):
                self._state('training', condition=label, completed_conditions=index)
                self.experiment(request).execute()
                self.reporter(c, final=False).execute()
                self._release_cache()
            self._state('reporting')
            result = self.reporter(c, final=True).execute()
            self._state('completed', trained_runs=len(requests)*len(c['seeds']))
            return result
        except Exception as exc:
            self._state('failed', error=str(exc))
            raise

    def _requests(self):
        c = self.config
        base = AdapterExperimentRequest(
            model='yolox-l', checkpoint=Path(c['checkpoint']), dataset=Path(c['dataset']),
            output=Path(c['output']), methods=('drax-residual-fusion',), seeds=tuple(c['seeds']),
            epochs=20, batch_size=8, image_size=640, device='cuda', amp=True, workers=0,
            reduction=16, rank=8, alpha=1., target='neck', train_head=False,
            head_policy='preserve', lr=.0001, seed_adapter_initialization=True)
        return [(item['id'], replace(base, output=base.output/'runs'/item['id'],
                                     methods=(item['method'],), rank=item['rank'], alpha=item['alpha']))
                for item in c['conditions']]

    def _verify(self):
        c = self.config
        if len(c['seeds']) != 20 or len(set(c['seeds'])) != 20:
            raise MLXUserError('This study requires exactly 20 distinct paired seeds.')
        expected = [('drax-residual-fusion','drax-residual-fusion',8,1.),
                    ('lora-r4','lora',4,.5), ('lora-r100','lora',100,12.5),
                    ('full-finetune','full-finetune',8,1.)]
        actual = [(v['id'],v['method'],v['rank'],v['alpha']) for v in c['conditions']]
        if actual != expected:
            raise MLXUserError('Four-method DAWN protocol differs from the declared recipe.')
        if c['analysis'] != {'metric':'mAP50_95','equivalence_margin':.01,
                            'family_alpha':.05,'multiplicity':'holm-six-claims'}:
            raise MLXUserError('Statistical protocol differs from the fixed overall-AP analysis.')
        if sha256_file(c['checkpoint']) != c['checkpoint_sha256']:
            raise MLXUserError('Foundation checksum changed.')
        for relative, digest in c['dataset_files'].items():
            if sha256_file(Path(c['dataset'])/relative) != digest:
                raise MLXUserError(f'Dataset changed: {relative}')
        for source in c['sources'].values():
            for relative, digest in source['files'].items():
                if sha256_file(Path(source['snapshot'])/relative) != digest:
                    raise MLXUserError(f'Source snapshot changed: {relative}')

    def _check_protocol(self, requests):
        import yaml
        fields = ('imgsz','batch','nbs','amp','amp_dtype','optimizer','lr0','weight_decay',
                  'scheduler','warmup_epochs','min_lr_ratio','no_aug_epochs','mosaic_prob',
                  'mixup_prob','hsv_prob','flip_prob','degrees','translate','shear','mosaic_scale',
                  'mixup_scale','ema','ema_decay','workers','max_det','eval_interval')
        configs = {}
        for label, request in requests:
            path = Path(self.config['output'])/'smoke'/label/request.methods[0]/'seed-9001/training/train_config.yaml'
            configs[label] = yaml.safe_load(path.read_text())
        reference = next(iter(configs.values()))
        for label, config in configs.items():
            different = [key for key in fields if config.get(key) != reference.get(key)]
            if different:
                raise MLXUserError(f'Matched training protocol differs for {label}: {different}')
        write_json_atomic(Path(self.config['output'])/'protocol-verification.json',
                          {'matched':True,'fields':fields,'conditions':list(configs)})

    def _state(self, status, **fields):
        write_json_atomic(Path(self.config['output'])/'status.json',
                          {'status':status,'planned_runs':80,
                           'updated_at':datetime.now(timezone.utc).isoformat(), **fields})

    @staticmethod
    def _release_cache():
        import torch
        gc.collect()
        torch.cuda.empty_cache()
