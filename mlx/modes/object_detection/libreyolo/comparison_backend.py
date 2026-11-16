"""CUDA/provider boundary for repeated scratch architecture experiments."""
from dataclasses import replace
import gc
import random
import time
import platform
import subprocess
import math
import json
from pathlib import Path

from mlx.core.artifacts import write_json_atomic, write_csv, sha256_file
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.comparison_experiment import RunCSPDraxComparison
from mlx.modes.object_detection.evaluation import normalize_detection_metrics
from mlx.modes.object_detection.requests import BenchmarkObjectDetectionRequest


def validate_detection_outputs(outputs, classes):
    import torch
    if isinstance(outputs, (list,tuple)):
        valid = len(outputs)>0 and all(isinstance(x,torch.Tensor) and x.ndim==4 and x.shape[0]==1 and x.shape[1]==classes+5 and torch.isfinite(x).all() for x in outputs)
    else:
        valid = isinstance(outputs,torch.Tensor) and outputs.ndim==3 and outputs.shape[0]==1 and outputs.shape[-1]==classes+5 and torch.isfinite(outputs).all()
    if not valid:
        raise MLXUserError('Reloaded detector output failed shape/finite validation.')


class LibreYOLOComparisonBackend:
    def verify_device(self):
        import torch
        if not torch.cuda.is_available():
            raise MLXUserError('CUDA was requested but is not available. Training was not started.')
        from mlx.modes.object_detection.adapter_queue import active_cuda_jobs
        if active_cuda_jobs():
            raise MLXUserError('Another CUDA compute process is active; comparison not started.')
        from libreyolo.utils.serialization import get_libreyolo_version
        version = get_libreyolo_version()
        if not isinstance(version, str) or not version:
            raise MLXUserError('LibreYOLO package-version metadata is missing; repair the source snapshot before training.')
        x = torch.ones(2, device='cuda')
        assert x.sum().item() == 2
        driver = subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
        return dict(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(), driver=driver, libreyolo=version,
                    cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                    vram_bytes=torch.cuda.get_device_properties(0).total_memory)

    def calibrate(self, models, image_size):
        import torch
        from .model_factory import build_scratch_model
        from .utils import resolve_model_spec
        self.verify_device()
        measurements = []
        selected = 8
        for name in models:
            safe = 0
            for batch in (1, 2, 4, 8):
                model = optimizer = images = targets = loss = None
                try:
                    model = build_scratch_model(resolve_model_spec(name), device='cuda').model.cuda().train()
                    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
                    images = torch.rand(batch, 3, image_size, image_size, device='cuda') * 255
                    targets = torch.zeros(batch, 600, 5, device='cuda')
                    targets[:, :100] = torch.tensor([0, image_size/2, image_size/2, 30, 30], device='cuda')
                    torch.cuda.reset_peak_memory_stats()
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = model(images, targets)
                        loss = output['total_loss']
                    if not torch.isfinite(loss):
                        raise MLXUserError(f'Non-finite calibration loss for {name}')
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_reserved()
                    free, total = torch.cuda.mem_get_info()
                    measurements.append(dict(model=name, batch=batch, peak_reserved_bytes=peak, free_bytes=free))
                    del output
                    if peak > .75 * total or free < .20 * total:
                        break
                    safe = batch
                except torch.cuda.OutOfMemoryError:
                    measurements.append(dict(model=name, batch=batch, status='oom'))
                    break
                finally:
                    del model, optimizer, images, targets, loss
                    gc.collect()
                    torch.cuda.empty_cache()
            if safe == 0:
                raise MLXUserError(f'No safe calibration batch for {name}')
            selected = min(selected, safe)
        return dict(physical_batch=selected, effective_batch=8, measurements=measurements)

    def run(self, request, model, seed, directory, *, pilot=False):
        import numpy as np
        import torch
        from libreyolo import LibreYOLO
        directory = Path(directory)
        self.verify_device()
        command = RunCSPDraxComparison(request)
        training = command._training_request(model, seed, None, directory, False)
        training = replace(training, epochs=3 if pilot else request.epochs, nbs=8,
                           no_aug_epochs=0 if pilot else request.no_aug_epochs,
                           plots=False, workers=0, patience=0, random_seed=seed + 10000,
                           max_labels=600)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        write_json_atomic(directory / 'config.json', training)
        # Model construction consumes its own seed; the provider resets the data RNG
        # at train() entry, after construction and class-count rebuilding.
        from mlx.modes.object_detection.commands import TrainObjectDetectionModel
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = TrainObjectDetectionModel(training).execute()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        losses = result.get('epoch_losses', [])
        if len(losses) != training.epochs or not all(math.isfinite(value) for value in losses):
            raise MLXUserError('Incomplete training or non-finite epoch loss; run not marked complete.')
        if not pilot and not result.get('best_checkpoint'):
            raise MLXUserError('No validation-selected best checkpoint; cannot evaluate the main run.')
        write_json_atomic(directory/'training.json', result)
        write_csv(directory/'training.csv', result['epoch_metrics'])
        memory = dict(peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                      peak_cuda_reserved_mb=torch.cuda.max_memory_reserved()/1024**2)
        checkpoint = command._checkpoint_path(result)
        # Reload through the public factory: this also verifies family reconstruction.
        loaded = LibreYOLO(str(checkpoint), device='cuda', task='detect')
        raw = loaded.model.cuda().float().eval()
        parameters = sum(p.numel() for p in raw.parameters())
        inputs = torch.zeros(1, 3, request.image_size, request.image_size, device='cuda')
        with torch.inference_mode():
            sample = raw(inputs)
            validate_detection_outputs(sample, loaded.nb_classes)
            del sample
            for _ in range(10):
                raw(inputs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                raw(inputs)
            torch.cuda.synchronize()
            latency = (time.perf_counter()-start)*1000/50
        del raw, loaded, inputs
        gc.collect()
        torch.cuda.empty_cache()
        evaluation = BenchmarkObjectDetectionRequest(
            provider='libreyolo', model=model, model_path=str(checkpoint),
            dataset_path=str(request.dataset), output_path=str(directory/'evaluation'),
            split='val' if pilot else 'test', device='cuda', height=request.image_size,
            width=request.image_size, batch_size=request.batch_size, workers=0,
            confidence=.001, iou=.6, max_detections=100, plots=False, save_predictions=True)
        started = time.perf_counter()
        evaluation_result = command._benchmark(evaluation)
        torch.cuda.synchronize()
        native = normalize_detection_metrics(getattr(evaluation_result, 'metrics', evaluation_result))
        scored = self._score_predictions(request.dataset, evaluation.split, directory/'evaluation')
        if abs(native['map_50_95'] - scored['mAP50_95']) > 1e-6:
            raise MLXUserError('Cached prediction AP differs from native evaluation; inspect image/category mapping.')
        metrics = dict(native, precision=scored['precision'], recall=scored['recall'], f1=scored['f1'],
                       map_75=scored['mAP75'], per_class_AP=scored['per_class_AP'])
        audit = json.loads((request.dataset/'audit.json').read_text())
        # Retain the full best checkpoint, including its optimizer state. Only the
        # newly generated redundant terminal state is retired after successful eval.
        last = directory/'training/formal/weights/last.pt'
        if checkpoint.name == 'best.pt' and last.is_file() and not last.is_symlink():
            if last.resolve().is_relative_to(directory.resolve()) and checkpoint.is_file():
                write_json_atomic(directory/'checkpoint-retention.json', dict(
                    kept=str(checkpoint), kept_sha256=sha256_file(checkpoint),
                    retired=str(last), retired_sha256=sha256_file(last),
                    reason='Successful completed run; validation-selected full best state retained.'))
                last.unlink()
        return dict(model=model, seed=seed, status='completed', checkpoint=str(checkpoint),
                    checkpoint_size_mb=checkpoint.stat().st_size/1024**2,
                    model_initialization_seed=seed, data_seed=seed+10000,
                    parameters=parameters, trainable_params=parameters, training_seconds=seconds,
                    seconds_per_epoch=seconds/training.epochs, evaluation_seconds=time.perf_counter()-started,
                    images_per_second=audit['counts']['train']['images']*training.epochs/seconds,
                    inference_latency_ms=latency, inference_scope='FP32 CUDA forward, batch one, synthetic input; excludes preprocessing/NMS',
                    epochs=training.epochs, physical_batch_size=request.batch_size, effective_batch_size=8,
                    amp=True, precision_confidence=.25, precision_iou=.5, metrics=metrics, **memory)

    @staticmethod
    def _score_predictions(dataset, split, output):
        from libreyolo.data.yolo_coco_api import create_yolo_coco_api
        from mlx.modes.object_detection.zero_shot.scoring import ScoreTransferPredictions
        coco = create_yolo_coco_api(str(dataset/'data.yaml'), split=split)
        annotation = output/'annotations.json'
        images = list(coco.imgs.values())
        write_json_atomic(annotation, dict(images=images, annotations=list(coco.anns.values()), categories=list(coco.cats.values())))
        spec = dict(annotations=str(annotation), images=[dict(row, group='all', sequence='all') for row in images])
        return ScoreTransferPredictions(spec, output/'predictions.json', output).execute()['all']
