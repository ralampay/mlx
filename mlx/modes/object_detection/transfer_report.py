"""Paired transfer statistics and complete, non-cherry-picked qualitative comparisons."""

from collections import defaultdict
import html
import json
from pathlib import Path
from statistics import mean

from mlx.core.artifacts import write_json_atomic, write_csv, sha256_file
from mlx.core.paired_superiority import AnalyzePairedSuperiority, holm_adjust
from mlx.modes.object_detection.zero_shot.gallery import GenerateTransferGallery
from mlx.modes.object_detection.zero_shot.scoring import ScoreTransferPredictions
from mlx.modes.object_detection.adapter_report import _summary
from mlx.core.exceptions import MLXUserError


class GenerateTaxonomyTransferReport:
    def __init__(self, output, dataset, *, baseline=None, gallery=True):
        self.output,self.dataset = Path(output),Path(dataset)
        self.baseline = Path(baseline) if baseline else None
        self.gallery = gallery

    def execute(self):
        ground_truth = json.loads((self.dataset / "test.json").read_text())
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        name = manifest.get("dataset","NEU-DET")
        images = [{**item,"path":str(self.dataset / "images" / "test" / item["file_name"]),
                   "group":(Path(item["file_name"]).stem.rsplit("_",1)[0] if name == "NEU-DET" else "all"),"sequence":"unknown"}
                  for item in ground_truth["images"]]
        dataset = {"name":name,"annotations":str(self.dataset / "test.json"),"images":images}
        rows,models = [],[]
        boxes = defaultdict(lambda:defaultdict(list))
        for annotation in ground_truth["annotations"]:
            boxes[annotation["image_id"]]["Ground truth"].append([*annotation["bbox"],annotation["category_id"],1.0,0])
        paths = sorted(self.output.glob("*/seed-*/predictions.json"))
        if self.baseline:
            paths += sorted(self.baseline.glob("*/seed-*/predictions.json"))
        known = set()
        for path in paths:
            run = path.parent
            method,seed = run.parent.name,int(run.name.removeprefix("seed-"))
            identifier = f"{method}/seed-{seed}"
            if identifier in known:
                raise MLXUserError(f"Duplicate comparison condition: {identifier}")
            known.add(identifier)
            metrics = json.loads((run / "metrics.json").read_text())
            if metrics.get("status") != "completed":
                continue
            scores_path = run / "scores.json"
            if self.baseline and run.is_relative_to(self.baseline) and not scores_path.exists():
                raise MLXUserError(f"Baseline has no cached scores: {run}")
            scores = (json.loads(scores_path.read_text()) if scores_path.exists()
                      else ScoreTransferPredictions(dataset,path,run).execute())
            inference = json.loads((run / "inference.json").read_text())
            rows.append({**metrics,**scores["all"],"end_to_end_inference_ms":inference["mean_end_to_end_ms"]})
            checkpoint = run / "adapter" / "checkpoint.pt"
            if not checkpoint.exists():
                checkpoint = Path(metrics.get("export_checkpoint_path",metrics["selected_checkpoint_path"]))
            models.append({"id":identifier,"method":method,"seed":seed,"checkpoint":str(checkpoint),
                           "sha256":sha256_file(checkpoint)})
            if self.gallery:
                for prediction in json.loads(path.read_text()):
                    if prediction["score"] >= .25:
                        boxes[prediction["image_id"]][identifier].append([*prediction["bbox"],prediction["category_id"],prediction["score"],0])
        aggregate = self.output / "aggregate"
        write_json_atomic(aggregate / "transfer-results.json",rows)
        write_csv(aggregate / "transfer-results.csv",rows,fieldnames=sorted({k for row in rows for k in row}))
        candidate = {row["seed"]:row for row in rows if row["method"] == "drax-hybrid"}
        comparisons = {}
        for method in ("head-only","lora","ssf","convpass","full-finetune"):
            baseline = {row["seed"]:row for row in rows if row["method"] == method}
            seeds = sorted(candidate.keys() & baseline.keys())
            if len(seeds) >= 2:
                comparisons[method] = AnalyzePairedSuperiority([candidate[s]["mAP50_95"] for s in seeds],
                                                               [baseline[s]["mAP50_95"] for s in seeds]).execute()
        if len(comparisons) == 5 and all(r["n"] == 5 for r in comparisons.values()):
            for row,p in zip(comparisons.values(),holm_adjust(r["p"] for r in comparisons.values())):
                row["holm_adjusted_p"] = p
        write_json_atomic(aggregate / "paired-superiority.json",comparisons)
        if self.baseline:
            comparisons = self._combined_report(rows, aggregate)
        write_json_atomic(self.output / "models" / "manifest.json",models)
        if not self.gallery:
            return {"completed_runs":len(rows),"test_images":len(images),"gallery":"deferred; predictions saved"}
        names = {c["id"]:c["name"] for c in ground_truth["categories"]}
        gallery = self.output / "gallery"
        directory = gallery / "neu-det"
        directory.mkdir(parents=True,exist_ok=True)
        links = []
        for item in images:
            panels = (("Ground truth","drax-hybrid/seed-1","lora-r100/seed-1","drax-spatial/seed-1")
                      if self.baseline else ("Ground truth","head-only/seed-1","lora/seed-1","drax-hybrid/seed-1"))
            GenerateTransferGallery._image_page(directory,item,boxes[item["id"]],names,models,panel_labels=panels)
            links.append(f'<a href="neu-det/{item["id"]}.html"><img loading="lazy" width="400" src="neu-det/{item["id"]}-comparison.jpg">{html.escape(item["file_name"])}</a>')
        summary = (aggregate / "summary.md").read_text() if (aggregate / "summary.md").exists() else ""
        (gallery / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>NEU-DET: all test images</h1>'
            '<p>'+html.escape(" / ".join(panels))+'. Select any completed method and seed within each image. '
            'Blank panels indicate a model not completed yet, not zero detections. Fixed confidence 0.25. Includes wins and failures.</p>'+"\n".join(links))
        (aggregate / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>NEU-DET transfer study</h1>'
            '<a href="../gallery/index.html">All-image comparison</a> · <a href="transfer-results.csv">Detailed metrics</a>'
            '<pre>'+html.escape(summary)+'</pre><h2>Seed-paired comparisons</h2><pre>'+html.escape(json.dumps(comparisons,indent=2))+'</pre>'
            '<p>Exploratory paired t intervals; Holm correction is computed only after every declared comparison has five pairs. '
            'Exact sign-flip p-values are also recorded. Five seeds have limited power. Non-significance does not establish equivalence. '
            'NEU acquisition sequences are unavailable; exact-pixel duplicate grouping cannot exclude near-duplicate leakage.</p>')
        return {"completed_runs":len(rows),"test_images":len(images)}

    @staticmethod
    def _combined_report(rows, aggregate):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["method"]].append(row)
        metrics = ("mAP50", "mAP50_95", "precision", "recall", "training_seconds",
                   "peak_cuda_memory_mb", "end_to_end_inference_ms")
        summaries = {method:{key:_summary([row[key] for row in runs]) for key in metrics}
                     for method,runs in grouped.items()}
        pairs = (("drax-hybrid","lora-r100"),("drax-hybrid","drax-spatial"),
                 ("lora-r100","lora"),("drax-spatial","full-finetune"))
        paired = {}
        for candidate, baseline in pairs:
            a = {row["seed"]:row for row in grouped[candidate]}
            b = {row["seed"]:row for row in grouped[baseline]}
            seeds = sorted(a.keys() & b.keys())
            if len(seeds) >= 2:
                paired[f"{candidate}_minus_{baseline}"] = AnalyzePairedSuperiority(
                    [a[s]["mAP50_95"] for s in seeds], [b[s]["mAP50_95"] for s in seeds]).execute()
        if len(paired) == 4 and all(value["n"] == 5 for value in paired.values()):
            for value,p in zip(paired.values(),holm_adjust(value["p"] for value in paired.values())):
                value["holm_adjusted_p"] = p
        write_json_atomic(aggregate / "paired-superiority.json", paired)
        write_json_atomic(aggregate / "results.json", {"runs":rows,"method_summaries":summaries,
            "paired_differences":paired,"evaluation":"cached-prediction COCO; AP in [0,1]"})
        write_csv(aggregate / "results.csv", rows,fieldnames=sorted({key for row in rows for key in row}))
        lines = ["# NEU-DET follow-up ablation", "", f"Completed comparison rows: {len(rows)}/40.", "",
            "Original runs are reused read-only. This follow-up was informed by prior test results; it is exploratory, not independent confirmation.", "",
            "| Condition | Seeds | mAP50 | mAP50–95 ± SD | Trainable M | Training min | CUDA MiB | End-to-end ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method in sorted(summaries):
            runs, summary = grouped[method], summaries[method]
            ap = summary["mAP50_95"]
            lines.append(f"| {method} | {len(runs)} | {summary['mAP50']['mean']*100:.2f} | {ap['mean']*100:.2f} ± {ap['standard_deviation']*100:.2f} | "
                         f"{mean(r['trainable_params'] for r in runs)/1e6:.3f} | {summary['training_seconds']['mean']/60:.2f} | "
                         f"{summary['peak_cuda_memory_mb']['mean']:.1f} | {summary['end_to_end_inference_ms']['mean']:.2f} |")
        lines += ["", "Paired statistics: `paired-superiority.json`. Holm correction uses the four prespecified comparisons after all five pairs complete.",
                  "Exact sign-flip tests are included; five seeds have limited power. Non-significance does not establish equivalence.",
                  "LoRA rank 100 uses alpha 12.5 to preserve alpha/rank=1/8; it is parameter-matched, not compute-matched.",
                  "All conditions train the full 7,552,801-parameter head. CUDA figures measure allocated training-epoch memory, not whole-device usage.",
                  "Native validator metrics remain in per-run metrics.json; these tables consistently use cached-prediction COCO evaluation."]
        (aggregate / "summary.md").write_text("\n".join(lines)+"\n")
        return paired
