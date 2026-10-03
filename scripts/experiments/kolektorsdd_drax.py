"""Reproducible small-model DRAX comparison on IVD-SEG KolektorSDD."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from itertools import product
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from mlx.core.random import seed_everything
from mlx.modes.segmentation.data import SegmentationDataset
from mlx.modes.segmentation.metrics import per_image_metrics
from mlx.modes.segmentation.models import build_segmentation_model
from mlx.modes.segmentation.train import TrainSegmentationModel
from mlx.modes.segmentation.utils import load_checkpoint_bundle


DATA_ROOT = Path.home() / "Desktop/datasets/semantic-segmentation/IVD-SEG"
RUN_ROOT = Path.home() / "Desktop/experiments/semantic-segmentation/kolektorsdd-drax-small"
ARCHIVE_URL = "https://zenodo.org/api/records/12704122/files/KolektorSDD-SEG.zip/content"
ARCHIVE_MD5 = "e552158c568773f634c0dd2d2eeab882"
MODELS = (
    "unet-compact",
    "unet-mobilenet_v3_large",
    "unet-drax_mobilenet_v3_large-average",
    "unet-mobilenet_v3_large-skip-conv",
    "unet-mobilenet_v3_large-skip-drax",
    "unet-mobilenet_v3_large-skip-drax-balanced",
)
CONTRASTS = (
    (MODELS[4], MODELS[2], "skip_vs_final"),
    (MODELS[4], MODELS[3], "drax_vs_conv"),
    (MODELS[5], MODELS[4], "balanced_vs_original"),
    (MODELS[5], MODELS[1], "balanced_vs_plain"),
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class PrepareKolektorSDD:
    def __init__(self, data_root: Path = DATA_ROOT, run_root: Path = RUN_ROOT) -> None:
        self.data_root = data_root
        self.run_root = run_root

    def execute(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        archive = self.data_root / "KolektorSDD-SEG.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(archive)
        if hashlib.md5(archive.read_bytes()).hexdigest() != ARCHIVE_MD5:
            raise ValueError(f"KolektorSDD archive checksum mismatch: {archive}")

        source = self.data_root / "KolektorSDD-SEG"
        records = []
        with zipfile.ZipFile(archive) as bundle:
            for split in ("train", "val", "test"):
                images = {
                    Path(name).stem: name for name in bundle.namelist()
                    if name.startswith(f"imgs/{split}/") and name.endswith(".png")
                }
                masks = {
                    Path(name).stem: name for name in bundle.namelist()
                    if name.startswith(f"annotations/{split}/") and name.endswith(".png")
                }
                if not images or images.keys() != masks.keys():
                    raise ValueError(f"Unpaired or missing {split} images in KolektorSDD archive.")
                for stem in sorted(images):
                    image_bytes, mask_bytes = bundle.read(images[stem]), bundle.read(masks[stem])
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image is None or mask is None or image.shape[:2] != (256, 256) or mask.shape != (256, 256):
                        raise ValueError(f"Invalid KolektorSDD pair: {split}/{stem}")
                    if not np.isin(mask, (0, 255)).all():
                        raise ValueError(f"Unexpected mask values: {split}/{stem}")
                    name = f"{split}_{stem}.png"
                    image_path, mask_path = source / "images" / name, source / "masks" / name
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image_bytes)
                    mask_path.write_bytes(mask_bytes)
                    records.append({
                        "id": name, "source_split": split,
                        "foreground_pixels": int(np.count_nonzero(mask)),
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    })
        if len(records) != 52 or len({row["id"] for row in records}) != 52:
            raise ValueError("Expected exactly 52 unique KolektorSDD pairs.")
        write_json(source / "manifest.json", records)
        self._prepare_folds(records, source)

    def _prepare_folds(self, records: list[dict], source: Path) -> None:
        ordered = sorted(records, key=lambda row: row["id"])
        areas = np.asarray([row["foreground_pixels"] for row in ordered])
        bins = np.digitize(areas, np.quantile(areas, (1 / 3, 2 / 3)))
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        manifests = []
        for fold, (development, held_out) in enumerate(splitter.split(areas, bins)):
            development_bins = bins[development]
            train, validation = train_test_split(
                development, test_size=5, stratify=development_bins, random_state=42 + fold
            )
            split_ids = {
                "train": [ordered[i]["id"] for i in sorted(train)],
                "val": [ordered[i]["id"] for i in sorted(validation)],
                "test": [ordered[i]["id"] for i in sorted(held_out)],
            }
            fold_root = self.run_root / "folds" / f"fold-{fold}"
            for split, names in split_ids.items():
                for name in names:
                    for kind in ("images", "masks"):
                        target = fold_root / split / kind / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source / kind / name, target)
            write_json(fold_root / "split.json", split_ids)
            manifests.append({"fold": fold, **split_ids})
        write_json(self.run_root / "folds.json", manifests)


class RunKolektorComparison:
    def __init__(self, run_root: Path = RUN_ROOT, *, epochs: int = 30, batch_size: int = 4,
                 device: str = "cuda", selected_fold: int | None = None,
                 selected_model: str | None = None) -> None:
        self.run_root = run_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.selected_fold = selected_fold
        self.selected_model = selected_model

    def execute(self) -> None:
        if not (self.run_root / "folds.json").exists():
            raise ValueError("Prepare KolektorSDD and its fold manifests first.")
        protocol = {
            "models": MODELS, "epochs": self.epochs, "batch_size": self.batch_size,
            "device": self.device, "loss": "cross-entropy-dice",
            "loss_config": {"foreground_weight": 20.0}, "lr": 1e-3,
            "input_size": (256, 256), "outer_folds": 3, "split_seed": 42,
            "pretrained": False,
        }
        protocol_path = self.run_root / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError(f"Experiment protocol differs from previous runs: {protocol_path}")
        write_json(protocol_path, protocol)
        for fold in range(3):
            if self.selected_fold is not None and fold != self.selected_fold:
                continue
            for model_name in MODELS:
                if self.selected_model is not None and model_name != self.selected_model:
                    continue
                self._run_one(fold, model_name)

    def _run_one(self, fold: int, model_name: str) -> None:
        output = self.run_root / "runs" / f"fold-{fold}" / model_name
        result_path = output / "result.json"
        if result_path.exists():
            print(f"Skipping completed fold={fold} model={model_name}", flush=True)
            return
        print(f"Starting fold={fold} model={model_name}", flush=True)
        seed_everything(1000 + fold)
        config = {
            "model": model_name, "dataset_path": str(self.run_root / "folds" / f"fold-{fold}"),
            "output_path": str(output), "device": self.device,
            "input_size": (256, 256), "transform": "resize", "num_classes": 2,
            "colored": True, "pretrained": False, "class_names": "background,foreground",
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": 1e-3,
            "loss": "cross-entropy-dice", "loss_config": {"foreground_weight": 20.0},
            "random_seed": 1000 + fold,
        }
        last = output / f"{model_name}.last.pth"
        if last.exists():
            config["model_path"] = str(last)
        try:
            TrainSegmentationModel(config).execute()
            checkpoint = output / f"{model_name}.best-dice.pth"
            if not checkpoint.exists():
                raise RuntimeError("Best-Dice checkpoint was not written.")
            rows, latency = self._evaluate(config, checkpoint, fold)
            write_csv(output / "image_metrics.csv", rows)
            result = {
                "fold": fold, "model": model_name, "checkpoint": str(checkpoint),
                "parameters": sum(p.numel() for p in build_segmentation_model(model_name, config, num_classes=2).parameters()),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "cpu_latency_ms_median": latency,
                "mean_foreground_dice": float(np.nanmean([row["foreground_dice"] for row in rows])),
                "mean_foreground_iou": float(np.nanmean([row["foreground_iou"] for row in rows])),
            }
            write_json(result_path, result)
            (output / "failure.json").unlink(missing_ok=True)
            print(f"Completed fold={fold} model={model_name} Dice={result['mean_foreground_dice']:.4f}", flush=True)
        except Exception as exc:
            write_json(output / "failure.json", {"fold": fold, "model": model_name, "error": repr(exc)})
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _evaluate(self, config: dict, checkpoint: Path, fold: int) -> tuple[list[dict], float]:
        model, _ = load_checkpoint_bundle({**config, "model_path": str(checkpoint), "device": "cpu"})
        model.eval()
        dataset = SegmentationDataset(
            config["dataset_path"], split="test", input_size=(256, 256),
            num_classes=2, colored=True,
        )
        rows = []
        with torch.inference_mode():
            for index, (image_path, _) in enumerate(dataset.samples):
                image, target = dataset[index]
                prediction = model(image.unsqueeze(0)).argmax(dim=1)[0].numpy()
                metrics = per_image_metrics(
                    target.numpy(), prediction, class_names=["background", "foreground"],
                    boundary_tolerance=2,
                )
                rows.append({"id": image_path.name, "fold": fold, **metrics})
            torch.set_num_threads(4)
            sample = torch.zeros(1, 3, 256, 256)
            for _ in range(10):
                model(sample)
            times = []
            for _ in range(30):
                start = time.perf_counter()
                model(sample)
                times.append((time.perf_counter() - start) * 1000)
        return rows, float(np.median(times))


class AnalyzeKolektorComparison:
    def __init__(self, run_root: Path = RUN_ROOT) -> None:
        self.run_root = run_root

    def execute(self) -> None:
        results = []
        predictions = {}
        predicted_areas = {}
        for fold in range(3):
            for model in MODELS:
                path = self.run_root / "runs" / f"fold-{fold}" / model
                result_path = path / "result.json"
                if not result_path.exists():
                    raise ValueError(f"Missing completed experiment: fold={fold} model={model}")
                results.append(json.loads(result_path.read_text()))
                with (path / "image_metrics.csv").open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        predictions[(model, row["id"])] = float(row["foreground_dice"])
                        predicted_areas[(model, row["id"])] = float(row["foreground_predicted_support"]) / (256 * 256)
        write_csv(self.run_root / "summary.csv", results)
        ids = sorted({image_id for model, image_id in predictions if model == MODELS[0]})
        if len(ids) != 52 or any((model, image_id) not in predictions for model in MODELS for image_id in ids):
            raise ValueError("Expected 52 paired held-out predictions for every model.")
        cpu_latency = self._measure_cpu_latency()
        model_rows = []
        for model in MODELS:
            runs = [row for row in results if row["model"] == model]
            model_rows.append({
                "model": model,
                "mean_held_out_dice": float(np.mean([predictions[(model, name)] for name in ids])),
                "mean_predicted_foreground_fraction": float(np.mean([
                    predicted_areas[(model, name)] for name in ids
                ])),
                "cpu_latency_ms_median": cpu_latency[model]["median_ms"],
                "cpu_latency_ms_p95": cpu_latency[model]["p95_ms"],
                "parameters": runs[0]["parameters"],
                "mean_checkpoint_bytes": float(np.mean([row["checkpoint_bytes"] for row in runs])),
            })
        write_csv(self.run_root / "models.csv", model_rows)
        plain_latency = next(row["cpu_latency_ms_median"] for row in model_rows if row["model"] == MODELS[1])
        eligible = [row for row in model_rows if row["parameters"] < 10_000_000
                    and row["cpu_latency_ms_median"] <= 1.25 * plain_latency]
        selected = max(eligible, key=lambda row: row["mean_held_out_dice"]) if eligible else None
        rng = np.random.default_rng(42)
        contrasts = []
        for candidate, reference, label in CONTRASTS:
            differences = np.asarray([predictions[(candidate, name)] - predictions[(reference, name)] for name in ids])
            signs = rng.choice((-1, 1), size=(100000, len(ids)))
            p = (1 + np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >= abs(differences.mean()))) / 100001
            draws = rng.integers(0, len(ids), size=(20000, len(ids)))
            low, high = np.quantile(differences[draws].mean(axis=1), (0.025, 0.975))
            fold_effects = {}
            for fold in range(3):
                fold_ids = set(json.loads((self.run_root / "folds" / f"fold-{fold}" / "split.json").read_text())["test"])
                fold_effects[f"fold_{fold}_difference"] = float(np.mean([
                    predictions[(candidate, name)] - predictions[(reference, name)]
                    for name in ids if name in fold_ids
                ]))
            effects = np.asarray([fold_effects[f"fold_{fold}_difference"] for fold in range(3)])
            null_effects = np.asarray([
                np.mean(np.asarray(signs) * effects)
                for signs in product((-1, 1), repeat=3)
            ])
            fold_p = float(np.mean(np.abs(null_effects) >= abs(effects.mean()) - 1e-12))
            contrasts.append({"contrast": label, "candidate": candidate, "reference": reference,
                              "mean_dice_difference": float(differences.mean()),
                              "ci_low": float(low), "ci_high": float(high), "p_unadjusted": float(p),
                              "p_fold_exact": fold_p,
                              **fold_effects})
        ordered = sorted(range(len(contrasts)), key=lambda i: contrasts[i]["p_unadjusted"])
        adjusted = 0.0
        for rank, index in enumerate(ordered):
            adjusted = max(adjusted, min(1.0, (len(contrasts) - rank) * contrasts[index]["p_unadjusted"]))
            contrasts[index]["p_holm"] = adjusted
        write_csv(self.run_root / "contrasts.csv", contrasts)
        source_manifest = json.loads((DATA_ROOT / "KolektorSDD-SEG" / "manifest.json").read_text())
        true_foreground = sum(row["foreground_pixels"] for row in source_manifest) / (len(source_manifest) * 256 * 256)
        protocol = json.loads((self.run_root / "protocol.json").read_text())
        write_json(self.run_root / "report.json", {"runs": results, "models": model_rows,
            "selected_model": selected, "contrasts": contrasts,
            "true_foreground_fraction": true_foreground,
            "interpretation": "Image-level paired tests are exploratory and may overstate evidence because images in each fold share a fitted model. The fold-level exact test is a sensitivity check."})
        lines = [
            "# KolektorSDD small DRAX comparison", "",
            "Scratch-trained three-fold study: 52 held-out image predictions per model; "
            f"{protocol['epochs']} epochs per fold. Image-level p-values are exploratory and may "
            "overstate evidence because images within a fold share one trained "
            "model. Exact fold-level sign-flip p-values provide a sensitivity "
            "check across three outer folds whose training sets overlap. Predicted foreground coverage "
            "exposes degenerate all-background or all-foreground masks; a small "
            "Dice advantage from such a mask is not useful localization.", "",
            f"Actual foreground coverage: {100 * true_foreground:.2f}%.", "",
            "| Model | Mean held-out Dice | Predicted foreground | CPU p50 / p95 (ms) | Parameters | Checkpoint (MiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(model_rows, key=lambda item: item["mean_held_out_dice"], reverse=True):
            lines.append(f"| {row['model']} | {row['mean_held_out_dice']:.4f} | "
                         f"{100 * row['mean_predicted_foreground_fraction']:.1f}% | "
                         f"{row['cpu_latency_ms_median']:.2f} / {row['cpu_latency_ms_p95']:.2f} | "
                         f"{row['parameters']:,} | {row['mean_checkpoint_bytes'] / 2**20:.1f} |")
        lines += ["", "| Contrast | Dice difference | 95% interval | Image Holm p | Fold exact p |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for row in contrasts:
            lines.append(f"| {row['contrast']} | {row['mean_dice_difference']:+.4f} | "
                         f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                         f"{row['p_holm']:.4g} | {row['p_fold_exact']:.3f} |")
        lines += ["", f"Selected under the CPU and parameter limits: "
                  f"{selected['model'] if selected else 'none'}.", "",
                  "See `summary.csv`, `models.csv`, and `contrasts.csv` for run and fold details.", ""]
        (self.run_root / "REPORT.md").write_text("\n".join(lines))

    def _measure_cpu_latency(self) -> dict[str, dict[str, float]]:
        cache = self.run_root / "cpu_latency.json"
        if cache.exists():
            measurements = json.loads(cache.read_text())
            if set(measurements) == set(MODELS):
                return measurements
        torch.set_num_threads(4)
        sample = torch.zeros(1, 3, 256, 256)
        rows = {}
        with torch.inference_mode():
            for model_name in MODELS:
                checkpoint = self.run_root / "runs" / "fold-0" / model_name / f"{model_name}.best-dice.pth"
                model, _ = load_checkpoint_bundle({"model_path": str(checkpoint), "device": "cpu"})
                model.eval()
                for _ in range(20):
                    model(sample)
                timings = []
                for _ in range(100):
                    start = time.perf_counter()
                    model(sample)
                    timings.append((time.perf_counter() - start) * 1000)
                rows[model_name] = {
                    "median_ms": float(np.median(timings)),
                    "p95_ms": float(np.percentile(timings, 95)),
                }
                del model
                gc.collect()
        write_json(cache, rows)
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.action == "prepare":
        PrepareKolektorSDD().execute()
    elif args.action == "run":
        RunKolektorComparison(epochs=args.epochs, batch_size=args.batch_size,
                              device=args.device, selected_fold=args.fold,
                              selected_model=args.model).execute()
    else:
        AnalyzeKolektorComparison().execute()


if __name__ == "__main__":
    main()
