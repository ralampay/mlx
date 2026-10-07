"""CUDA profiling and strict reconstruction for new-taxonomy adapter studies."""

import gc
import json
import time

import torch

from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError
from .adapter_backend import VerifyFoundationCheckpoint, build_experimental_yolox, require_experiment_device


def build_transfer_model(request, method, seed, classes, *, registry=None):
    from mlx.modes.object_detection.libreyolo.adapter_targets import yolox_targets
    from mlx.modes.object_detection.feature_adapters import inject_adapters
    from libreyolo.models.yolox.transfer import reset_classifiers
    from mlx.core.random import seed_everything
    seed_everything(seed)
    model, info = VerifyFoundationCheckpoint(request.model, request.checkpoint).execute()
    reset_classifiers(model, len(classes), seed=seed)
    if method == "head-only":
        model.requires_grad_(False)
        model.head.requires_grad_(True)
    elif method != "full-finetune":
        inject_adapters(model, method, targets=yolox_targets(model,request.target,method, registry=registry), reduction=request.reduction,
                        rank=request.rank, alpha=request.alpha, train_head=True, registry=registry)
    return model, {**info, "nc": len(classes), "classes": dict(enumerate(classes))}


class CalibrateTransferBatch:
    """Conservative real-loss/AdamW probes, reserving memory for EMA and evaluation."""
    def __init__(self, request, classes, *, registry=None):
        self.registry = registry
        self.request, self.classes = request, classes

    def execute(self):
        request = self.request
        device = require_experiment_device(request.device)
        if device.type != "cuda":
            raise MLXUserError("Transfer calibration requires CUDA")
        records = json.loads((request.dataset / "manifest.json").read_text())["selected_images"]
        dense = max((r for r in records if r["split"] == "train"), key=lambda r: len(r["boxes"]))
        profiles = {}
        for method in request.methods:
            safe, probes = 0, []
            for batch in (1,2,4,8):
                model = optimizer = scaler = images = targets = loss = outputs = None
                try:
                    model, _ = build_transfer_model(request, method, 42, self.classes, registry=self.registry)
                    model.to(device).train()
                    for module in model.modules():
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
                            module.eval()
                    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=request.lr)
                    scaler = torch.amp.GradScaler("cuda", enabled=request.amp)
                    images = torch.rand(batch,3,request.image_size,request.image_size,device=device)*255
                    if len(dense["boxes"]) > request.max_labels:
                        raise MLXUserError("Training label capacity is smaller than the densest annotation")
                    targets = torch.zeros(batch,request.max_labels,5,device=device)
                    for index,(category,x,y,w,h) in enumerate(dense["boxes"]):
                        sx,sy = request.image_size/dense["size"][0],request.image_size/dense["size"][1]
                        targets[:,index] = torch.tensor([category,(x+w/2)*sx,(y+h/2)*sy,w*sx,h*sy],device=device)
                    torch.cuda.reset_peak_memory_stats(device)
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", enabled=request.amp):
                            outputs = model(images,targets)
                            loss = outputs["total_loss"]
                        if not torch.isfinite(loss):
                            raise MLXUserError("Nonfinite calibration loss")
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_reserved(device)
                    ema_bytes = sum(t.numel()*t.element_size() for t in model.state_dict().values())
                    free,total = torch.cuda.mem_get_info(device)
                    external = total-free-torch.cuda.memory_reserved(device)
                    fits = peak+ema_bytes+max(0,external) <= total*.75
                    probes.append({"batch":batch,"peak_reserved_bytes":peak,"ema_reserve_bytes":ema_bytes,"safe":fits})
                    if fits:
                        safe = batch
                    else:
                        break
                except torch.cuda.OutOfMemoryError:
                    probes.append({"batch":batch,"safe":False,"error":"cuda_oom"})
                    break
                finally:
                    model = optimizer = scaler = images = targets = loss = outputs = None
                    gc.collect()
                    torch.cuda.empty_cache()
            profiles[method] = {"safe_batch":safe,"probes":probes}
            if not safe:
                write_json_atomic(request.output / "calibration.json", {"status":"failed","profiles":profiles})
                raise MLXUserError(f"No batch size with 25% headroom for {method}")
        selected = min(p["safe_batch"] for p in profiles.values())
        result = {"status":"completed","profiles":profiles,"physical_batch_size":selected,
                  "gradient_accumulation":8//selected,"effective_batch_size":8}
        write_json_atomic(request.output / "calibration.json", result)
        return result


class CacheTransferPredictions:
    """Reload the saved model, then cache FP32 test predictions and synchronized timings."""
    def __init__(self, request, method, seed, classes, *, registry=None):
        self.registry = registry
        self.request,self.method,self.seed,self.classes = request,method,seed,classes

    def execute(self):
        from libreyolo.models.yolox.transfer import load_transfer_state_dict
        from libreyolo.utils.serialization import load_untrusted_torch_file
        request = self.request
        run = request.output / request.method_id(self.method) / f"seed-{self.seed}"
        if (run / "predictions.json").exists() and (run / "inference.json").exists():
            return
        model, info = build_transfer_model(request,self.method,self.seed,self.classes, registry=self.registry)
        compact = run / "adapter" / "checkpoint.pt"
        if compact.exists():
            payload = torch.load(compact,map_location="cpu",weights_only=True)
            if payload.get("format") != "yolox-taxonomy-transfer-v1":
                raise MLXUserError("Unexpected transfer checkpoint format")
            load_transfer_state_dict(model,payload["state"])
        else:
            config = json.loads((run / "config.json").read_text())
            payload = load_untrusted_torch_file(config.get("export_checkpoint_path",config["selected_checkpoint_path"]),map_location="cpu")
            model.load_state_dict(payload["model"],strict=True)
        del payload
        wrapper = build_experimental_yolox(model.float().eval(),info,request.device)
        from mlx.modes.object_detection.evaluation import normalize_detection_metrics
        reloaded = normalize_detection_metrics(wrapper.val(
            data=str(request.dataset / "data.yaml"), split="test", imgsz=request.image_size,
            batch=request.batch_size, device=request.device, workers=request.workers,
            conf=.001, iou=.6, half=False, verbose=False, save_json=False,
            max_det=request.max_detections,
            save_plots=False, save_dir=str(run / "reload-evaluation")))
        reference = json.loads((run / "metrics.json").read_text())
        differences = {key:abs(reference[key]-reloaded[native])
                       for key,native in (("mAP50","map_50"),("mAP50_95","map_50_95"))}
        if any(value > 1e-6 for value in differences.values()):
            raise MLXUserError(f"Reloaded transfer checkpoint changes evaluation: {differences}")
        coco = json.loads((request.dataset / "test.json").read_text())
        predictions,timings = [],[]
        for index,item in enumerate(coco["images"]):
            source = str(request.dataset / "images" / "test" / item["file_name"])
            if index == 0:
                wrapper.predict(source=source,conf=.001,iou=.6,imgsz=request.image_size,save=False,max_det=request.max_detections)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = wrapper.predict(source=source,conf=.001,iou=.6,imgsz=request.image_size,save=False,max_det=request.max_detections)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-started)*1000)
            result = output[0] if isinstance(output,list) else output
            if result.boxes is not None:
                for xyxy,cls,score in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist()):
                    x1,y1,x2,y2 = xyxy
                    predictions.append({"image_id":item["id"],"category_id":int(cls),"bbox":[x1,y1,x2-x1,y2-y1],"score":score})
        write_json_atomic(run / "predictions.json",predictions)
        write_json_atomic(run / "inference.json",{"precision":"float32","reload_verified":True,
            "reload_metric_absolute_differences":differences,
            "latency_ms":timings,"mean_end_to_end_ms":sum(timings)/len(timings),
            "includes":"image loading, preprocessing, CUDA forward, NMS, result construction"})
