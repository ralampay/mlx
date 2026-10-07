"""Local multi-dataset adapter pilots using existing, authoritative annotations."""

from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path

import yaml

from mlx.core.artifacts import sha256_file, write_json_atomic, write_csv
from mlx.core.exceptions import MLXUserError
from mlx.modes.object_detection.adapter_transfer import PrepareTransferStudy, RunQueuedTransferStudy


def pilot_conditions():
    return [{"id":name,"method":method,"rank":rank,"alpha":alpha} for name,method,rank,alpha in (
        ("head-only","head-only",8,1.), ("ssf","ssf",8,1.),
        ("convpass","convpass",8,1.), ("lora","lora",8,1.),
        ("lora-r100","lora",100,12.5), ("drax-spatial","drax-spatial",8,1.),
        ("drax-hybrid","drax-hybrid",8,1.), ("full-finetune","full-finetune",8,1.))]


class PrepareRemoteAdapterDataset:
    """Create a non-destructive adapter view; images and labels stay in their dataset."""
    def __init__(self, source, destination):
        self.source,self.destination = Path(source).resolve(),Path(destination).resolve()

    def execute(self):
        source,target = self.source,self.destination
        source_manifest = source / "manifest.json"
        original = json.loads(source_manifest.read_text())
        if target.exists():
            manifest = json.loads((target / "manifest.json").read_text())
            if manifest.get("source_manifest_sha256") != sha256_file(source_manifest):
                raise MLXUserError(f"Prepared remote dataset source changed: {source}")
            for filename,digest in manifest["source_receipts"].items():
                if sha256_file(source / filename) != digest:
                    raise MLXUserError(f"Prepared remote dataset file changed: {source / filename}")
            return manifest
        names = original["class_names"]
        records = []
        annotations = {}
        maximum = 0
        receipts = {"manifest.json":sha256_file(source_manifest)}
        members = {(m["split"],m["image"]):m for m in original["members"]}
        for split in ("train","val","test"):
            path = source / f"instances_{split}.json"
            coco = json.loads(path.read_text())
            receipts[path.name] = sha256_file(path)
            if {c["id"]:c["name"] for c in coco["categories"]} != dict(enumerate(names)):
                raise MLXUserError(f"COCO class order mismatch: {path}")
            boxes = defaultdict(list)
            for annotation in coco["annotations"]:
                boxes[annotation["image_id"]].append([annotation["category_id"],*annotation["bbox"]])
            for image in coco["images"]:
                name = image["file_name"]
                if Path(name).name != name:
                    raise MLXUserError(f"Expected flat prepared image filename: {name}")
                member = members[(split,name)]
                image_path = source / "images" / split / name
                label_path = source / "labels" / split / (Path(name).stem+".txt")
                if not label_path.is_file():
                    raise MLXUserError(f"Missing prepared labels: {label_path}")
                if sha256_file(image_path) != member["sha256"]:
                    raise MLXUserError(f"Image checksum mismatch: {image_path}")
                receipts[str(image_path.relative_to(source))] = member["sha256"]
                receipts[str(label_path.relative_to(source))] = sha256_file(label_path)
                maximum = max(maximum,len(boxes[image["id"]]))
                records.append({"stem":Path(name).stem,"split":split,"size":[image["width"],image["height"]],
                    "boxes":boxes[image["id"]],"pixel_sha256":member["pixel_sha256"],"scene":member.get("scene")})
            annotations[split] = coco
        target.mkdir(parents=True)
        # Individual links allow the existing snapshot validator to hash every file.
        for split,coco in annotations.items():
            for kind in ("images","labels"):
                (target / kind / split).mkdir(parents=True)
            for image in coco["images"]:
                name = image["file_name"]
                for kind,filename in (("images",name),("labels",Path(name).stem+".txt")):
                    link = target / kind / split / filename
                    original_path = source / kind / split / filename
                    link.symlink_to(os.path.relpath(original_path,link.parent))
            write_json_atomic(target / f"{split}.json",coco)
        (target / "data.yaml").write_text(yaml.safe_dump({"path":str(target),"train":"images/train",
            "val":"images/val","test":"images/test","nc":len(names),"names":names}))
        manifest = {"dataset":original["dataset"],"classes":names,"splits":original["splits"],
            "selected_images":records,"selection_sha256":sha256_file(source_manifest),
            "source_manifest_sha256":sha256_file(source_manifest),"source":str(source),"source_receipts":receipts,
            "max_labels":max(50,maximum),"split_policy":original["split_policy"],
            "overlap":{key:original.get(key) for key in ("cross_split_exact_duplicates","cross_split_source_scenes")},
            "limitations":["Original splits and documented cross-split overlap are preserved.",
                "320-pixel pilot inputs may limit small-object detection.",
                "Standard YOLOX preprocessing filters boxes no larger than one resized pixel; label capacity does not remove this geometric filter.",
                "Primary COCO AP uses maxDets=100; predictions retain up to 1000 detections per image."]}
        write_json_atomic(target / "manifest.json",manifest)
        return manifest


class PrepareRemoteAdapterPilot:
    def __init__(self, output, datasets_root, checkpoint, repositories):
        self.output,self.datasets_root,self.checkpoint = map(Path,(output,datasets_root,checkpoint))
        self.repositories = repositories

    def execute(self):
        if self.output.exists():
            raise MLXUserError(f"Pilot output already exists: {self.output}")
        self.output.mkdir(parents=True)
        studies = []
        for name in ("rsod","nwpu-vhr-10","ssdd","hrsid","dior"):
            data = self.datasets_root / name / "adapter-pilot-v1"
            manifest = PrepareRemoteAdapterDataset(self.datasets_root / name / "processed",data).execute()
            output = self.output / name
            PrepareTransferStudy(output,data,self.checkpoint,self.repositories,conditions=pilot_conditions(),settings={
                "seeds":[1],"epochs":50,"image_size":320,"required_physical_batch":None,
                "max_labels":manifest["max_labels"],"max_detections":1000,"report_gallery":False}).execute()
            studies.append({"dataset":name,"config":str(output / "plan.json"),"runs":8})
        config = {"output":str(self.output),"studies":studies,"total_runs":40,"seeds":[1],"epochs":50,
            "interpretation":"Exploratory single-seed pilot. No seed-level statistical significance or equivalence claims.",
            "foundation_checkpoint":str(self.checkpoint)}
        write_json_atomic(self.output / "pilot.json",config)
        return config


class RunRemoteAdapterPilot:
    def __init__(self, configuration, *, resume=False, study_runner=RunQueuedTransferStudy):
        self.configuration,self.resume = Path(configuration),resume
        self.study_runner = study_runner

    def execute(self):
        config = json.loads(self.configuration.read_text())
        output = Path(config["output"])
        with (output / "pilot.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MLXUserError("This remote-sensing pilot is already running") from exc
            return self._run(config, output)

    def _run(self, config, output):
        if (output / "status.json").exists() and not self.resume:
            raise MLXUserError("Existing pilot requires explicit --resume")
        completed = 0
        try:
            for study in config["studies"]:
                write_json_atomic(output / "status.json",{"status":"running","dataset":study["dataset"],
                    "completed_runs":completed,"total_runs":config["total_runs"],"child_status":str(Path(study["config"]).parent / "status.json")})
                self.study_runner(study["config"],resume=self.resume).execute()
                completed += study["runs"]
                self._aggregate(config)
            write_json_atomic(output / "status.json",{"status":"completed","completed_runs":completed,"total_runs":config["total_runs"]})
        except Exception as exc:
            write_json_atomic(output / "status.json",{"status":"failed","completed_runs":completed,"error":str(exc)})
            raise

    @staticmethod
    def _aggregate(config):
        rows = []
        for study in config["studies"]:
            path = Path(study["config"]).parent / "aggregate" / "transfer-results.json"
            if path.exists():
                rows.extend({**row,"dataset_id":study["dataset"]} for row in json.loads(path.read_text()))
        destination = Path(config["output"]) / "aggregate"
        write_json_atomic(destination / "results.json",{"runs":rows,"interpretation":config["interpretation"]})
        write_csv(destination / "results.csv",rows,fieldnames=sorted({key for row in rows for key in row}))
