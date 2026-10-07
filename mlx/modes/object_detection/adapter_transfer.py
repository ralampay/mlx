"""Supervised taxonomy-transfer study lifecycle and queued local execution."""

from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_experiment import AdapterExperimentRequest, RunAdapterExperiment
from mlx.modes.object_detection.adapter_report import GenerateAdapterReport
from mlx.modes.object_detection.adapter_queue import active_cuda_jobs


METHODS = ("head-only", "lora", "ssf", "convpass", "drax-hybrid", "full-finetune")


class PrepareTransferStudy:
    def __init__(self, output, dataset, checkpoint, repositories, *, conditions=None, baseline=None, settings=None):
        self.output,self.dataset,self.checkpoint = map(lambda p:Path(p).expanduser().resolve(),(output,dataset,checkpoint))
        self.repositories = repositories
        self.conditions = conditions
        self.baseline = Path(baseline).expanduser().resolve() if baseline else None
        self.settings = dict(settings or {})

    def execute(self):
        if self.output.exists():
            raise MLXUserError(f"Refusing to overwrite study: {self.output}")
        from mlx.modes.object_detection.adapter_data import load_prepared_adapter_dataset
        manifest = load_prepared_adapter_dataset(self.dataset,expected_classes=None)
        self.output.mkdir(parents=True)
        models = self.output / "models"
        models.mkdir()
        foundation = models / self.checkpoint.name
        shutil.copy2(self.checkpoint,foundation)
        if sha256_file(foundation) != sha256_file(self.checkpoint):
            raise MLXUserError("Foundation snapshot checksum mismatch")
        from mlx.core.repository_snapshot import SnapshotRepositories
        sources = SnapshotRepositories(self.repositories, self.output / "source").execute()
        config = {"output":str(self.output),"dataset":str(self.dataset),"checkpoint":str(foundation),
            "original_checkpoint":str(self.checkpoint),"checkpoint_sha256":sha256_file(foundation),
            "dataset_selection_sha256":manifest["selection_sha256"],"methods":list(METHODS),"seeds":[1,2,3,4,5],
            "epochs":50,"image_size":320,"lr":.0003,"device":"cuda","amp":True,
            "head_policy":"reset-classifiers","train_head":True,"rank":8,"reduction":8,
            "target":"neck","effective_batch_size":8,"sources":sources}
        config["dataset_files"] = {str(path.relative_to(self.dataset)):sha256_file(path)
                                   for path in sorted(self.dataset.rglob("*")) if path.is_file()}
        if self.conditions is not None:
            config["conditions"] = self.conditions
            config["methods"] = [item["method"] for item in self.conditions]
            config["required_physical_batch"] = 8
        allowed = {"seeds","epochs","image_size","required_physical_batch","max_labels","max_detections","report_gallery"}
        if set(self.settings) - allowed:
            raise MLXUserError("Unsupported transfer-study settings")
        config.update(self.settings)
        config["dataset_name"] = manifest["dataset"]
        if self.baseline:
            from mlx.modes.object_detection.transfer_baseline import VerifyTransferBaseline
            config["baseline"] = VerifyTransferBaseline(self.baseline, config).execute()
        write_json_atomic(self.output / "plan.json",config)
        write_json_atomic(self.output / "dataset.json",{k:v for k,v in manifest.items() if k != "selected_images"})
        return config


class RunQueuedTransferStudy:
    def __init__(self, config_path, *, resume=False, poll_seconds=30):
        self.path = Path(config_path)
        self.resume = resume
        self.poll_seconds = poll_seconds

    def execute(self):
        config = json.loads(self.path.read_text())
        output = Path(config["output"])
        with (output / "study.lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MLXUserError("This transfer study is already running") from exc
            state = output / "status.json"
            if state.exists() and not self.resume:
                raise MLXUserError("Existing queue state requires explicit --resume; no run will be overwritten")
            try:
                self._verify(config)
                while True:
                    jobs = active_cuda_jobs()
                    if not jobs:
                        break
                    self._status(output,"queued",waiting_for=jobs)
                    time.sleep(self.poll_seconds)
                self._verify(config)
                self._run(config)
            except Exception as exc:
                self._status(output,"failed",error=f"{type(exc).__name__}: {exc}")
                raise

    @staticmethod
    def _verify(config):
        if sha256_file(config["checkpoint"]) != config["checkpoint_sha256"]:
            raise MLXUserError("Foundation snapshot changed")
        manifest = json.loads((Path(config["dataset"]) / "manifest.json").read_text())
        if manifest["selection_sha256"] != config["dataset_selection_sha256"]:
            raise MLXUserError("Dataset split manifest changed")
        for name,digest in config["dataset_files"].items():
            if sha256_file(Path(config["dataset"]) / name) != digest:
                raise MLXUserError(f"Dataset file changed: {name}")
        for source in config["sources"].values():
            for name,digest in source["files"].items():
                if sha256_file(Path(source["snapshot"]) / name) != digest:
                    raise MLXUserError(f"Source snapshot changed: {name}")
        if config.get("baseline"):
            for name, digest in config["baseline"]["files"].items():
                if sha256_file(Path(config["baseline"]["root"]) / name) != digest:
                    raise MLXUserError(f"Baseline artifact changed: {name}")

    @staticmethod
    def _status(output,status,**details):
        write_json_atomic(output / "status.json",{"status":status,"pid":os.getpid(),
            "updated_at":datetime.now(timezone.utc).isoformat(),**details})

    def _run(self,config):
        import torch
        from mlx.modes.object_detection.libreyolo.adapter_backend import CollectAdapterEnvironment, require_experiment_device
        from mlx.modes.object_detection.libreyolo.transfer_backend import CalibrateTransferBatch, CacheTransferPredictions
        from mlx.modes.object_detection.transfer_report import GenerateTaxonomyTransferReport
        output = Path(config["output"])
        device = require_experiment_device(config["device"])
        if config.get("baseline"):
            import libreyolo
            environment = CollectAdapterEnvironment(output, device=str(device), amp=config["amp"],
                mlx_root=Path(__file__).resolve().parents[3],libreyolo_root=Path(libreyolo.__file__).resolve().parents[1]).execute()
            original_environment = json.loads((Path(config["baseline"]["root"]) / "environment.json").read_text())
            for key in ("gpu_name","gpu_vram_bytes","nvidia_driver","pytorch_version","pytorch_cuda_version","cudnn_version","amp"):
                if environment[key] != original_environment[key]:
                    raise MLXUserError(f"Baseline environment mismatch: {key}")
        classes = json.loads((Path(config["dataset"]) / "manifest.json").read_text())["classes"]
        request = AdapterExperimentRequest(model="yolox-l",checkpoint=Path(config["checkpoint"]),dataset=Path(config["dataset"]),
            output=output,methods=tuple(config["methods"]),seeds=tuple(config["seeds"]),epochs=config["epochs"],
            image_size=config["image_size"],device=config["device"],amp=config["amp"],lr=config["lr"],
            head_policy=config["head_policy"],train_head=config["train_head"],rank=config["rank"],
            reduction=config["reduction"],target=config["target"],
            max_labels=config.get("max_labels",50),max_detections=config.get("max_detections",300))
        conditions = config.get("conditions") or [
            {"id":method,"method":method,"rank":request.rank,"alpha":request.alpha}
            for method in request.methods]
        requests = [replace(request, methods=(item["method"],), condition_id=item["id"] if config.get("conditions") else None,
                            rank=item["rank"], alpha=item["alpha"]) for item in conditions]
        if len({item["id"] for item in conditions}) != len(conditions):
            raise MLXUserError("Condition IDs must be unique")
        for item in requests:
            item.validate()
        self._status(output,"calibrating")
        calibration_path = output / "calibration.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            if calibration.get("status") != "completed":
                raise MLXUserError("Incomplete calibration; inspect artifacts and use a new study")
        else:
            if config.get("conditions"):
                profiles = {}
                for item in requests:
                    profile = CalibrateTransferBatch(replace(item,output=output / "calibration" / item.condition_id),classes).execute()
                    profiles[item.condition_id] = profile
                chosen = min(p["physical_batch_size"] for p in profiles.values())
                required = config.get("required_physical_batch")
                if required and chosen < required:
                    raise MLXUserError("New conditions cannot safely retain baseline batch 8; study stopped")
                selected = required or chosen
                calibration = {"status":"completed","physical_batch_size":selected,"gradient_accumulation":8//selected,"effective_batch_size":8,"profiles":profiles}
                write_json_atomic(calibration_path,calibration)
            else:
                calibration = CalibrateTransferBatch(request,classes).execute()
        requests = [replace(item,batch_size=calibration["physical_batch_size"],gradient_accumulation=calibration["gradient_accumulation"]) for item in requests]
        self._status(output,"smoke-tests",physical_batch_size=requests[0].batch_size)
        self._declare_study(config,output / "smoke",conditions,(42,))
        smoke_rows = []
        for item in requests:
            smoke = replace(item,output=output / "smoke",epochs=1,seeds=(42,))
            smoke_rows.extend(RunAdapterExperiment(smoke).execute())
            CacheTransferPredictions(smoke,item.methods[0],42,classes).execute()
        timings = {row["method"]:row["seconds_per_epoch"] for row in smoke_rows}
        estimate = sum(timings.values())*request.epochs*len(request.seeds)
        write_json_atomic(output / "runtime-estimate.json",{"basis":"measured one-epoch CUDA smoke runs, includes training validation/checkpoint overhead",
            "seconds_per_epoch":timings,"estimated_training_hours":estimate/3600,
            "limitations":"Excludes final cached prediction/report overhead; warmup and I/O can change runtime"})
        self._declare_study(config,output,conditions,request.seeds)
        completed = 0
        total = len(requests)*len(request.seeds)
        for seed in request.seeds:
            # Rotate deterministic method order so one method is not always first.
            offset = (seed-1)%len(requests)
            ordered = requests[offset:]+requests[:offset]
            for item in ordered:
                method = item.methods[0]
                identifier = item.method_id(method)
                self._status(output,"training",method=identifier,seed=seed,completed_runs=completed,total_runs=total,
                    estimated_remaining_hours=sum(timings.values())*request.epochs*(total-completed)/len(requests)/3600)
                one = replace(item,seeds=(seed,))
                rows = RunAdapterExperiment(one).execute()
                timings[identifier] = rows[0]["seconds_per_epoch"]
                CacheTransferPredictions(one,method,seed,classes).execute()
                GenerateAdapterReport(output,comparison_method="drax-hybrid").execute()
                GenerateTaxonomyTransferReport(output,request.dataset,baseline=config.get("baseline",{}).get("root"),
                    gallery=config.get("report_gallery",True)).execute()
                completed += 1
                write_json_atomic(output / "runtime-estimate.json",{
                    "basis":"most recent measured CUDA run per method (smoke until full run available)",
                    "seconds_per_epoch":timings,"completed_runs":completed,
                    "estimated_remaining_training_hours":sum(timings.values())*request.epochs*(total-completed)/len(requests)/3600,
                    "excludes":"queue delay and cached prediction/report overhead"})
                gc.collect()
                torch.cuda.empty_cache()
        self._status(output,"completed",completed_runs=completed,total_runs=total)

    @staticmethod
    def _declare_study(config,output,conditions,seeds):
        output.mkdir(parents=True,exist_ok=True)
        study = {"research_question":"Can parameter-efficient adapters approach full fine-tuning on a shifted YOLOX-L target domain, and does DraxAdapter improve the performance-efficiency tradeoff?",
            "model":"yolox-l","foundation_checkpoint":config["checkpoint"],"foundation_sha256":config["checkpoint_sha256"],
            "dataset":config.get("dataset_name","NEU-DET"),"dataset_selection_sha256":config["dataset_selection_sha256"],
            "available_methods":[item["id"] for item in conditions],"planned_seeds":list(seeds)}
        RunAdapterExperiment._write_once(output / "study.json",study)
