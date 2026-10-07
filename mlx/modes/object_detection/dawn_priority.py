"""Paired same-taxonomy adapter study; OS process control stays in its launcher."""
from dataclasses import replace
import gc
import json
from pathlib import Path

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest, RunAdapterExperiment
from mlx.modes.object_detection.adapter_report import GenerateAdapterReport
from mlx.modes.object_detection.libreyolo.adapter_backend import CalibrateAdapterBatchSize
from mlx.modes.object_detection.libreyolo.adapter_checkpoint import ExportBestAdapterDetector


class RunDawnPriorityStudy:
    def __init__(self, config, *, experiment=RunAdapterExperiment,
                 calibrator=CalibrateAdapterBatchSize, reporter=GenerateAdapterReport,
                 exporter=ExportBestAdapterDetector):
        self.config = config
        self.experiment, self.calibrator = experiment, calibrator
        self.reporter, self.exporter = reporter, exporter

    def execute(self):
        import torch
        c = self.config
        root = Path(c['output'])
        if (root / 'status.json').exists():
            raise MLXUserError('Priority study already has state; inspect before restarting.')
        self._verify_inputs()
        request = AdapterExperimentRequest(
            model='yolox-l', checkpoint=Path(c['checkpoint']), dataset=Path(c['dataset']),
            output=root / 'study', methods=('drax-residual-fusion', 'lora'), seeds=(1,2,3,4,5),
            epochs=20, batch_size=8, image_size=640, device='cuda', amp=True,
            workers=0, reduction=16, rank=8, alpha=1., target='neck', train_head=False,
            head_policy='preserve', lr=.0001, seed_adapter_initialization=True,
        )
        try:
            self._state('calibrating')
            calibration = self.calibrator(
                request.model, request.checkpoint, root / 'calibration',
                device=request.device, image_size=request.image_size, amp=request.amp,
                reduction=16, maximum_batch=8, profiles=request.methods,
            ).execute()
            if calibration['selected_physical_batch_size'] != 8:
                raise MLXUserError('DAWN batch 8 failed calibration; protocol was not changed.')
            self._state('smoke-testing')
            self.experiment(replace(request, output=root / 'smoke', seeds=(42,), epochs=1)).execute()
            self._verify_smoke_protocol(root, request.methods)
            gc.collect()
            torch.cuda.empty_cache()
            self._state('training', planned_runs=15 if c.get('include_lora100', False) else 10)
            self.experiment(request).execute()
            extra_runs = self._lora100(request) if c.get('include_lora100', False) else ()
            self._state('reporting')
            self.reporter(request.output, comparison_method='drax-residual-fusion', extra_runs=extra_runs).execute()
            self._html(request.output / 'aggregate')
            self._state('exporting')
            receipt = self.exporter(request.output, c['export'], seeds=request.seeds).execute()
            self._state('completed', export=receipt)
            return receipt
        except Exception as exc:
            self._state('failed', error=str(exc))
            raise

    def _lora100(self, request):
        """Keep rank-100 run artifacts separate; alias only the comparison rows."""
        import torch
        root = Path(self.config['output'])
        gc.collect()
        torch.cuda.empty_cache()
        self._state('calibrating-lora100', planned_runs=15)
        calibration = self.calibrator(
            request.model, request.checkpoint, root / 'calibration-lora100',
            device=request.device, image_size=request.image_size, amp=request.amp,
            maximum_batch=8, profiles=('lora',), rank=100, alpha=12.5,
        ).execute()
        if calibration['selected_physical_batch_size'] != 8:
            raise MLXUserError('LoRA-100 batch 8 failed calibration; protocol was not changed.')
        high = replace(request, methods=('lora',), output=root / 'lora-r100-study', rank=100, alpha=12.5)
        self._state('smoke-testing-lora100', planned_runs=15)
        self.experiment(replace(high, output=root / 'smoke-lora100', seeds=(42,), epochs=1)).execute()
        self._verify_smoke_protocol(root, ('lora',), smoke_directory='smoke-lora100')
        self._state('training-lora100', planned_runs=15)
        results = self.experiment(high).execute()
        return tuple({**row, 'method':'lora-r100', 'experiment_id':f"lora-r100-seed-{row['seed']}",
                      'source_run':str(high.output/'lora'/f"seed-{row['seed']}")}
                     for row in results if row['method'] == 'lora')

    def _verify_inputs(self):
        c = self.config
        if sha256_file(c['checkpoint']) != c['checkpoint_sha256']:
            raise MLXUserError('Foundation snapshot changed.')
        for relative, digest in c['dataset_files'].items():
            if sha256_file(Path(c['dataset']) / relative) != digest:
                raise MLXUserError(f'DAWN input changed: {relative}')
        for source in c['sources'].values():
            for relative, digest in source['files'].items():
                if sha256_file(Path(source['snapshot']) / relative) != digest:
                    raise MLXUserError(f'Source snapshot changed: {relative}')

    def _state(self, status, **fields):
        from datetime import datetime, timezone
        write_json_atomic(Path(self.config['output']) / 'status.json',
                          {'status': status, 'updated_at': datetime.now(timezone.utc).isoformat(), **fields})

    @staticmethod
    def _verify_smoke_protocol(root, methods, *, smoke_directory='smoke'):
        import yaml
        reference = yaml.safe_load((root / 'provenance/train_config.yaml').read_text())
        fields = ('imgsz','batch','nbs','amp','amp_dtype','optimizer','lr0','weight_decay',
                  'scheduler','warmup_epochs','min_lr_ratio','no_aug_epochs','mosaic_prob',
                  'mixup_prob','hsv_prob','flip_prob','degrees','translate','shear','mosaic_scale',
                  'mixup_scale','ema','ema_decay','workers','max_det','eval_interval')
        for method in methods:
            actual = yaml.safe_load((root / smoke_directory / method / 'seed-42/training/train_config.yaml').read_text())
            differences = {k: {'historical':reference.get(k), 'current':actual.get(k)}
                           for k in fields if reference.get(k) != actual.get(k)}
            if differences:
                raise MLXUserError(f'Historical DAWN protocol mismatch for {method}: {differences}')
        write_json_atomic(root / f'protocol-verification-{smoke_directory}.json',
                          {'matched_fields':list(fields),'methods':list(methods),'matched':True})

    @staticmethod
    def _html(aggregate):
        # Presentation is optional and kept outside the training/report statistics.
        import html
        text = (aggregate / 'summary.md').read_text()
        results = json.loads((aggregate / 'results.json').read_text())
        rows = []
        for row in results['runs']:
            rows.append('<tr>'+''.join('<td>'+html.escape(str(row.get(k, '')))+'</td>'
                        for k in ('method','seed','mAP50','mAP50_95','trainable_params',
                                  'training_seconds','peak_cuda_memory_mb'))+'</tr>')
        (aggregate / 'summary.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>DAWN paired adapter results</title>'
            '<style>body{font:16px system-ui;margin:2em}pre{white-space:pre-wrap}td,th{border:1px solid #ccc;padding:.5em}table{border-collapse:collapse}</style>'
            '<a href="results.csv">CSV</a> · <a href="results.json">JSON</a> · '
            '<a href="summary.md">Markdown</a><h1>DAWN paired adapter results</h1>'
            '<p>AP values are fractions. Five training seeds; frozen rows are contextual.</p>'
            '<table><tr><th>Method</th><th>Seed</th><th>AP50</th><th>AP50–95</th>'
            '<th>Trainable parameters</th><th>Training seconds</th><th>CUDA MiB</th></tr>'
            +''.join(rows)+'</table><pre>'+html.escape(text)+'</pre>')
