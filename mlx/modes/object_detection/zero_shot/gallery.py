"""Offline, all-image comparative bounding-box gallery with balanced highlights."""

from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from mlx.core.artifacts import write_json_atomic
from .data import read_json

COLORS = ("#ff5050", "#50dd50", "#5599ff", "#ffcc22", "#ff55ee", "#22dddd")


def select_highlights(images, differences):
    result = []
    for group in sorted({i["group"] for i in images}):
        available = [
            i
            for i in images
            if i["group"] == group and differences.get(str(i["id"])) is not None
        ]
        for category in ("win", "loss", "near-tie"):
            candidates = [
                i
                for i in available
                if (
                    differences[str(i["id"])] > 0.02
                    if category == "win"
                    else (
                        differences[str(i["id"])] < -0.02
                        if category == "loss"
                        else abs(differences[str(i["id"])]) <= 0.02
                    )
                )
            ]
            candidates.sort(
                key=lambda i: (
                    (
                        (-differences[str(i["id"])])
                        if category == "win"
                        else (
                            differences[str(i["id"])]
                            if category == "loss"
                            else abs(differences[str(i["id"])])
                        )
                    ),
                    i["id"],
                )
            )
            chosen = []
            for item in candidates:
                same = [c for c in chosen if c["sequence"] == item["sequence"]]
                if item["sequence"].startswith("video"):
                    if any(abs(c["frame"] - item["frame"]) < 30 for c in same):
                        continue
                elif len(same) >= 2:
                    continue
                chosen.append(item)
                result.append(
                    {
                        "image_id": item["id"],
                        "group": group,
                        "category": category,
                        "mean_seed_f1_difference": differences[str(item["id"])],
                    }
                )
                if len(chosen) == 5:
                    break
    return result


class GenerateTransferGallery:
    def __init__(self, output):
        self.output = Path(output)

    def execute(self, datasets, models):
        root = self.output / "gallery"
        root.mkdir(exist_ok=True)
        entries, highlights = [], []
        for dataset in datasets:
            directory = root / dataset["name"]
            directory.mkdir(exist_ok=True)
            gt = read_json(dataset["annotations"])
            names = {c["id"]: c["name"] for c in gt["categories"]}
            boxes = defaultdict(lambda: defaultdict(list))
            for a in gt["annotations"]:
                boxes[a["image_id"]]["Ground truth"].append(
                    [*a["bbox"], a["category_id"], 1.0, a.get("iscrowd", 0)]
                )
            image_counts = {}
            for model in models:
                run = self.output / "runs" / dataset["name"] / model["id"]
                for prediction in read_json(run / "predictions.json"):
                    if prediction["score"] >= 0.25:
                        boxes[prediction["image_id"]][model["id"]].append(
                            [
                                *prediction["bbox"],
                                prediction["category_id"],
                                prediction["score"],
                                0,
                            ]
                        )
                if model["method"] in {"lora", "drax-hybrid"}:
                    image_counts[model["id"]] = read_json(run / "per-image.json")
            differences = {}
            seeds = [m["seed"] for m in models if m["method"] == "lora"]
            for item in dataset["images"]:
                key = str(item["id"])
                pairs = [
                    (
                        image_counts[f"drax-hybrid/seed-{s}"][key]["f1"],
                        image_counts[f"lora/seed-{s}"][key]["f1"],
                    )
                    for s in seeds
                ]
                # Undefined empty-frame F1 is not assigned an invented ranking score.
                differences[key] = (
                    sum(a - b for a, b in pairs) / len(pairs)
                    if all(a is not None and b is not None for a, b in pairs)
                    else None
                )
            selected = select_highlights(dataset["images"], differences)
            highlights.extend({"dataset": dataset["name"], **r} for r in selected)
            for item in dataset["images"]:
                self._image_page(directory, item, boxes[item["id"]], names, models)
                entries.append(
                    {
                        "dataset": dataset["name"],
                        "id": item["id"],
                        "group": item["group"],
                        "sequence": item["sequence"],
                        "file": item["file_name"],
                        "url": f"{dataset['name']}/{item['id']}.html",
                        "panel": f"{dataset['name']}/{item['id']}-comparison.jpg",
                    }
                )
        write_json_atomic(root / "images.json", entries)
        write_json_atomic(root / "highlights.json", highlights)
        links = {(e["dataset"], e["id"]): e for e in entries}
        selected_html = "".join(
            f'<li><a href="{links[(r["dataset"], r["image_id"])]["url"]}">{html.escape(r["dataset"])} {html.escape(r["group"])} {r["category"]}: image {r["image_id"]}, ΔF1 {r["mean_seed_f1_difference"]:+.3f}</a></li>'
            for r in highlights
        )
        page = f"""<!doctype html><meta charset="utf-8"><title>Zero-shot comparative gallery</title>
<style>body{{font:16px system-ui;margin:2em;background:#15171b;color:#eee}}a{{color:#7cf}}img{{max-width:600px;width:95%}}#items{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr))}}input{{padding:12px;width:60%}}</style>
<h1>Complete zero-shot comparison: {len(entries)} images</h1>
<p><a href="../aggregate/summary.md">Report</a> · <a href="../aggregate/results.csv">All metrics</a> · <a href="../models/manifest.json">Model snapshots</a></p>
<p>Panels: ground truth / foundation / LoRA seed 1 / Drax hybrid seed 1. Open an image to select any method and seed, zoom, or download predictions. Solid boxes are detections; crowd GT is labelled CROWD. Threshold 0.25.</p>
<details><summary>Balanced wins, losses and near-ties (mean-five-seed F1 ranking)</summary><ul>{selected_html}</ul></details>
<p><input id="q" placeholder="Search dataset, weather, sequence or filename"><button onclick="offset=Math.max(0,offset-40);render()">Previous</button><button onclick="offset+=40;render()">Next</button><span id="count"></span></p><div id="items"></div>
<script>const entries={json.dumps(entries).replace('<', chr(92)+'u003c')}; let offset=0;
function render(){{let filtered=entries.filter(e=>JSON.stringify(e).toLowerCase().includes(q.value.toLowerCase()));if(offset>=filtered.length)offset=0;count.textContent=` ${{offset+1}}–${{Math.min(offset+40,filtered.length)}} / ${{filtered.length}}`;items.innerHTML=filtered.slice(offset,offset+40).map(e=>`<figure><a href="${{e.url}}"><img loading="lazy" src="${{e.panel}}"></a><figcaption>${{e.dataset}} / ${{e.group}} / ${{e.file}}</figcaption></figure>`).join('')}}q.oninput=()=>{{offset=0;render()}};render();</script>"""
        (root / "index.html").write_text(page)
        return {"images": len(entries), "highlights": len(highlights)}

    @staticmethod
    def _image_page(directory, item, boxes, names, models):
        identifier = item["id"]
        with Image.open(item["path"]) as original:
            image = original.convert("RGB")
            image.thumbnail((1280, 1280))
        image.save(directory / f"{identifier}.jpg", quality=88)
        panel = Image.new("RGB", (1280, 800), "#15171b")
        for n, label in enumerate(
            ("Ground truth", "frozen", "lora/seed-1", "drax-hybrid/seed-1")
        ):
            tile = image.copy()
            tile.thumbnail((640, 360))
            draw = ImageDraw.Draw(tile)
            sx, sy = tile.width / item["width"], tile.height / item["height"]
            for x, y, w, h, category, score, crowd in boxes[label]:
                color = COLORS[category % len(COLORS)]
                draw.rectangle(
                    (x * sx, y * sy, (x + w) * sx, (y + h) * sy), outline=color, width=2
                )
                draw.text(
                    (x * sx, max(0, y * sy - 11)),
                    f"{names[category]} {score:.2f}" + (" CROWD" if crowd else ""),
                    fill=color,
                    stroke_width=1,
                    stroke_fill="black",
                )
            left, top = (n % 2) * 640, (n // 2) * 400
            panel.paste(tile, (left, top + 25))
            ImageDraw.Draw(panel).text((left + 8, top + 5), label, fill="white")
        panel.save(directory / f"{identifier}-comparison.jpg", quality=88)
        for model in models:
            boxes.setdefault(model["id"], [])
        data = json.dumps(
            {
                "boxes": boxes,
                "names": names,
                "colors": COLORS,
                "width": item["width"],
                "height": item["height"],
            }
        ).replace("<", "\\u003c")
        options = "".join(f"<option>{html.escape(k)}</option>" for k in boxes)
        page = f"""<!doctype html><meta charset="utf-8"><title>Image {identifier}</title>
<style>body{{font:16px system-ui;background:#15171b;color:white;margin:1em}}a{{color:#7cf}}canvas{{width:100%;height:auto}}.views{{display:flex;gap:12px}}.views>div{{width:50%;overflow:auto}}select{{padding:8px}}</style>
<p><a href="../index.html">All images</a> · {html.escape(item['file_name'])} · confidence ≥0.25 · <a href="{identifier}-comparison.jpg">Download fixed comparison</a> · <a href="../../models/manifest.json">Model hashes</a></p>
<p>Zoom: <input type="range" id="zoom" min="100" max="300" value="100"> <button id="download">Download displayed-threshold predictions (all models)</button></p>
<div class="views"><div><select id="left">{options}</select><canvas id="a"></canvas></div><div><select id="right">{options}</select><canvas id="b"></canvas></div></div>
<script>const data={data};const image=new Image(); image.src='{identifier}.jpg';
function draw(canvas,label){{canvas.width=image.width;canvas.height=image.height;let c=canvas.getContext('2d');c.drawImage(image,0,0);let sx=canvas.width/data.width,sy=canvas.height/data.height;c.font='14px sans-serif';c.lineWidth=2;for(let [x,y,w,h,k,s,crowd] of data.boxes[label]){{c.strokeStyle=data.colors[k%6];c.fillStyle=c.strokeStyle;c.setLineDash(crowd?[5,4]:[]);c.strokeRect(x*sx,y*sy,w*sx,h*sy);c.fillText(data.names[k]+' '+s.toFixed(2)+(crowd?' CROWD':''),x*sx,Math.max(14,y*sy-3))}}canvas.style.width=zoom.value+'%'}}
function render(){{draw(a,left.value);draw(b,right.value)}}left.value='lora/seed-1';right.value='drax-hybrid/seed-1';image.onload=render;left.onchange=right.onchange=zoom.oninput=render;
download.onclick=()=>{{let link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(data)],{{type:'application/json'}}));link.download='{identifier}-boxes.json';link.click();URL.revokeObjectURL(link.href)}};</script>"""
        (directory / f"{identifier}.html").write_text(page)
