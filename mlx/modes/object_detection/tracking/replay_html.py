from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def render_tracking_replay_html(payload: Mapping[str, Any]) -> str:
    """Build a self-contained browser replay with no video or network dependency."""

    embedded_data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    embedded_data = embedded_data.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__MLX_REPLAY_DATA__", embedded_data)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLX Tracking Replay</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --panel: rgba(14, 30, 50, 0.92);
      --panel-border: rgba(148, 163, 184, 0.2);
      --text: #e5edf7;
      --muted: #91a4ba;
      --accent: #38bdf8;
      --ground-truth: #4ade80;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 15%, #11365a 0, transparent 32%),
        radial-gradient(circle at 85% 0%, #312e81 0, transparent 28%),
        var(--bg);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    main { width: min(1180px, calc(100% - 32px)); margin: 28px auto; }
    header { display: flex; gap: 18px; align-items: end; justify-content: space-between; }
    h1 { margin: 0; font-size: clamp(24px, 4vw, 38px); letter-spacing: -0.04em; }
    .subtitle { color: var(--muted); margin: 5px 0 0; }
    .badge {
      border: 1px solid var(--panel-border); border-radius: 999px;
      padding: 7px 11px; color: var(--accent); background: var(--panel);
    }
    .metrics {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px; margin: 18px 0;
    }
    .metric, .stage, .controls {
      border: 1px solid var(--panel-border); background: var(--panel);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
    }
    .metric { padding: 12px 14px; border-radius: 12px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { font-size: 20px; font-variant-numeric: tabular-nums; }
    .stage { border-radius: 16px; padding: 12px; }
    canvas {
      display: block; width: 100%; height: auto; border-radius: 10px;
      background: #07101c;
    }
    .controls {
      margin-top: 12px; border-radius: 14px; padding: 12px;
      display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center;
    }
    button, select {
      border: 1px solid var(--panel-border); border-radius: 9px;
      background: #11243a; color: var(--text); padding: 8px 12px; cursor: pointer;
    }
    button:hover, select:hover { border-color: var(--accent); }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .toggles { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px; color: var(--muted); }
    .toggles label { cursor: pointer; }
    .toggles input { accent-color: var(--accent); }
    .readout { min-width: 140px; text-align: right; font-variant-numeric: tabular-nums; }
    footer { color: var(--muted); margin: 14px 2px; font-size: 12px; }
    @media (max-width: 650px) {
      header { align-items: start; flex-direction: column; }
      .controls { grid-template-columns: auto 1fr; }
      .readout { grid-column: 1 / -1; text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>MLX Tracking Replay</h1>
        <p class="subtitle" id="subtitle">Portable 2D MOT visualization</p>
      </div>
      <div class="badge" id="schema"></div>
    </header>
    <section class="metrics" id="metrics"></section>
    <section class="stage">
      <canvas id="scene" aria-label="Tracking replay canvas"></canvas>
      <div class="toggles">
        <label><input id="showPredictions" type="checkbox" checked> Predictions</label>
        <label><input id="showGroundTruth" type="checkbox" checked> Ground truth</label>
        <label><input id="showTrails" type="checkbox" checked> Motion trails</label>
        <span>Prediction: solid · Ground truth: dashed green</span>
      </div>
    </section>
    <section class="controls">
      <button id="play" type="button">▶ Play</button>
      <input id="frame" type="range" min="1" step="1" value="1">
      <div class="readout"><span id="frameLabel"></span> · <select id="speed">
        <option value="0.25">0.25×</option>
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
        <option value="4">4×</option>
      </select></div>
    </section>
    <footer>Space: play/pause · Left/right arrows: step · No source video required</footer>
  </main>
  <script id="mlx-replay-data" type="application/json">__MLX_REPLAY_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("mlx-replay-data").textContent);
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("frame");
    const playButton = document.getElementById("play");
    const speed = document.getElementById("speed");
    const frameLabel = document.getElementById("frameLabel");
    const canvasWidth = data.canvas.width;
    const canvasHeight = data.canvas.height;
    const frameCount = data.frame_count;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    slider.max = frameCount;
    document.getElementById("schema").textContent = data.schema_version;
    const runLabel = [data.run.provider, data.run.tracker].filter(Boolean).join(" · ");
    document.getElementById("subtitle").textContent = runLabel || "Portable 2D MOT visualization";

    const byFrame = records => {
      const grouped = new Map();
      for (const record of records || []) {
        if (!grouped.has(record.frame_id)) grouped.set(record.frame_id, []);
        grouped.get(record.frame_id).push(record);
      }
      return grouped;
    };
    const predictions = byFrame(data.predictions.records);
    const groundTruth = byFrame(data.ground_truth ? data.ground_truth.records : []);

    const metricLabels = {
      mota: "MOTA", idf1: "IDF1", motp: "Mean IoU", precision: "Precision",
      recall: "Recall", id_switches: "ID switches", predictions: "Predictions",
      ground_truth_objects: "GT objects"
    };
    const metricOrder = ["mota", "idf1", "motp", "precision", "recall", "id_switches"];
    const metricRoot = document.getElementById("metrics");
    const metricValues = data.metrics || {
      predictions: data.predictions.record_count,
      ground_truth_objects: data.ground_truth ? data.ground_truth.record_count : 0
    };
    const keys = data.metrics ? metricOrder : ["predictions", "ground_truth_objects"];
    for (const key of keys) {
      const value = metricValues[key];
      if (value === undefined || value === null) continue;
      const card = document.createElement("div");
      card.className = "metric";
      const label = document.createElement("span");
      label.textContent = metricLabels[key] || key;
      const strong = document.createElement("strong");
      strong.textContent = typeof value === "number" && !Number.isInteger(value)
        ? value.toFixed(3) : String(value);
      card.append(label, strong);
      metricRoot.append(card);
    }

    const colorForId = id => `hsl(${(Number(id) * 137.508) % 360} 78% 62%)`;
    const center = row => [row.left + row.width / 2, row.top + row.height / 2];
    const maxTrail = Math.max(1, Math.round(data.fps * 2));

    function drawGrid() {
      ctx.fillStyle = "#07101c";
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.10)";
      ctx.lineWidth = 1;
      const step = Math.max(40, Math.round(Math.min(canvasWidth, canvasHeight) / 8));
      for (let x = 0; x <= canvasWidth; x += step) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvasHeight); ctx.stroke();
      }
      for (let y = 0; y <= canvasHeight; y += step) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasWidth, y); ctx.stroke();
      }
    }

    function drawTrail(trackId, frame, color) {
      const points = [];
      for (let candidate = Math.max(1, frame - maxTrail); candidate <= frame; candidate++) {
        const row = (predictions.get(candidate) || []).find(item => item.track_id === trackId);
        if (row) points.push(center(row));
      }
      if (points.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.38;
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.stroke();
      ctx.restore();
    }

    function drawBox(row, color, dashed, prefix) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 2 : 3;
      ctx.setLineDash(dashed ? [8, 5] : []);
      ctx.strokeRect(row.left, row.top, row.width, row.height);
      const text = `${prefix} ${row.track_id}${dashed ? "" : ` · ${row.confidence.toFixed(2)}`}`;
      ctx.font = "bold 12px ui-sans-serif, system-ui, sans-serif";
      const textWidth = ctx.measureText(text).width;
      const labelY = Math.max(4, row.top - 20);
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.fillRect(row.left, labelY, textWidth + 10, 19);
      ctx.fillStyle = "#07101c";
      ctx.fillText(text, row.left + 5, labelY + 14);
      ctx.restore();
    }

    function render(frame) {
      drawGrid();
      const predicted = predictions.get(frame) || [];
      const expected = groundTruth.get(frame) || [];
      if (document.getElementById("showTrails").checked) {
        for (const row of predicted) drawTrail(row.track_id, frame, colorForId(row.track_id));
      }
      if (document.getElementById("showGroundTruth").checked) {
        for (const row of expected) drawBox(row, "#4ade80", true, "GT");
      }
      if (document.getElementById("showPredictions").checked) {
        for (const row of predicted) drawBox(row, colorForId(row.track_id), false, "ID");
      }
      ctx.fillStyle = "rgba(7, 16, 28, 0.82)";
      ctx.fillRect(8, 8, 238, 28);
      ctx.fillStyle = "#e5edf7";
      ctx.font = "13px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(`Frame ${frame} · ${predicted.length} tracks · ${expected.length} GT`, 16, 27);
      frameLabel.textContent = `${frame} / ${frameCount}`;
    }

    let timer = null;
    function pause() {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
      playButton.textContent = "▶ Play";
    }
    function play() {
      pause();
      playButton.textContent = "❚❚ Pause";
      const delay = Math.max(10, 1000 / (data.fps * Number(speed.value)));
      timer = window.setInterval(() => {
        const next = Number(slider.value) + 1;
        if (next > frameCount) { pause(); return; }
        slider.value = next;
        render(next);
      }, delay);
    }
    playButton.addEventListener("click", () => timer === null ? play() : pause());
    slider.addEventListener("input", () => { pause(); render(Number(slider.value)); });
    speed.addEventListener("change", () => { if (timer !== null) play(); });
    for (const id of ["showPredictions", "showGroundTruth", "showTrails"]) {
      document.getElementById(id).addEventListener("change", () => render(Number(slider.value)));
    }
    window.addEventListener("keydown", event => {
      if (event.code === "Space") { event.preventDefault(); timer === null ? play() : pause(); }
      if (event.code === "ArrowRight" || event.code === "ArrowLeft") {
        pause();
        const delta = event.code === "ArrowRight" ? 1 : -1;
        slider.value = Math.min(frameCount, Math.max(1, Number(slider.value) + delta));
        render(Number(slider.value));
      }
    });
    render(1);
  </script>
</body>
</html>
"""
