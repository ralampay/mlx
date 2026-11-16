# Temporal synthetic data

The `time_series` mode trains generative VAEs on numeric CSV time series, samples
independent sequences, and measures synthetic quality. See
[ARCHITECTURE.md](../ARCHITECTURE.md#temporal-synthetic-data) for ownership and extension contracts.

## Input and training

A CSV contains a time column and one or more numeric feature columns:

```csv
timestamp,temperature,pressure
2026-01-01T00:00:00Z,20.1,101.2
2026-01-01T00:01:00Z,20.4,101.3
```

This is a format illustration; training requires enough rows for all partitions.
Select the measured time series with `--feature-columns` and its ordering with
`--time-column`. Unselected columns are ignored. Numeric timestamps retain their
units; ISO-8601 timestamps use seconds (naive timestamps mean UTC).

```bash
python -m mlx --mode time_series --action ls-models

python -m mlx --mode time_series --action train \
  --input measurements.csv --time-column timestamp \
  --feature-columns temperature,pressure \
  --window-size 128 --stride 32 --model timevae \
  --latent-dim 16 --hidden-dims 64,128 \
  --epochs 100 --batch-size 64 --lr 0.001 --kl-weight 1.0 \
  --seed 42 --device cpu --output runs/timevae
```

Use a single feature name for univariate generation. Add `--series-id-column sensor_id`
for independent series; each must have the same sampling interval. Rows are sorted
within series. Duplicate times, missing/non-finite values, irregular sampling, and
insufficient data are errors, not silently repaired. Features cannot use the reserved
export names `synthetic_series_id`, `step`, or `elapsed_time`.

Each series is split chronologically at floor(70% × rows) and floor(85% × rows)
before window creation. Every partition needs at least `window-size` observations.
For length 128, roughly 860 or more rows per series are needed; the exact check uses
the integer boundaries. Only training observations fit per-feature standardization.
Constant training features use scale 1. Training windows may overlap; validation
and benchmark windows do not. Trailing incomplete windows are discarded.

Models:

- `vanilla-vae`: dense encoder and decoder operating on flattened windows.
- `conv-vae`: strided temporal convolution encoder and transposed-convolution decoder.
- `timevae` (default): convolutional VAE plus additive level, polynomial trend,
  and optional periodic components. This is an independent TimeVAE-inspired PyTorch
  implementation, not a verified reproduction of the paper's benchmark results.

All share the flags in the example. `hidden-dims` describes dense widths or
convolution channels. The objective is mean reconstruction MSE plus `kl-weight`
times mean KL over the batch and latent dimensions. Decoder output lengths are
aligned to the requested window size. Generation uses a standard-normal latent prior.

For TimeVAE, pass `--model-config timevae.json`:

```json
{"trend_degree": 2, "seasonal_periods": [24, 48]}
```

Trend degree defaults to 2 (0 disables it; maximum 10). Seasonal periods default to
none, must be distinct integers >= 2, and cannot exceed the window size. They are
measured in samples, not timestamp units. Level and residual components are always
present. Other initial models accept no model-specific options. `--config` retains
its existing AWS meaning; it is not the temporal-model configuration flag.

Training writes `best.pt`, `history.json`, and `manifest.json`. History reports
training and validation reconstruction, KL and total losses separately. Validation
uses fixed sampled noise across epochs; the lowest validation total selects the
checkpoint. No test data select model weights.

## Synthesis

```bash
python -m mlx --mode time_series --action synthesize \
  --model-path runs/timevae/best.pt --num-sequences 1000 \
  --seed 123 --batch-size 64 --device cpu --output synthetic/timevae
```

The checkpoint is self-contained; the original CSV is unnecessary. Outputs are
`synthetic.csv` and a provenance `manifest.json`. CSV columns are new
`synthetic_series_id`, zero-based `step`, `elapsed_time`, and feature columns in their
original order and units. Each ID is a separate sequence of the trained length.
Do not concatenate these into one continuous trajectory. Generation is unconditional,
not a one-to-one reconstruction or a continuation of the input dataset. To match a
windowed dataset's size, set `--num-sequences` to its window count.

CPU and CUDA are supported; CPU is the default. Seed, device, model, and batch size
must stay the same for reproducibility in the same runtime. Cross-platform bitwise
reproducibility is not promised. Output directories must not exist; failed workflows
remove their staged artifacts. Training, synthesis and reports never overwrite
existing directories.

## Quality benchmark

```bash
python -m mlx --mode time_series --action benchmark \
  --model-path runs/timevae/best.pt --input measurements.csv \
  --synthetic-data synthetic/timevae/synthetic.csv \
  --bootstrap-samples 100 --max-metric-windows 512 --seed 42 \
  --output reports/timevae
```

Use the original CSV: its digest and preprocessing must match the checkpoint.
The synthetic CSV may come from another generator if it follows the export schema.
The report records hashes of all inputs. Final fidelity compares generated windows
with the untouched real test partition; the real-versus-real reference compares
validation with test and may expose distribution drift.

- **Marginal fidelity:** Wasserstein distance and KS statistic per feature.
- **Variation:** standard-deviation ratios, within-window variation,
  between-window mean variation, and first-difference variation.
- **Temporal structure:** autocorrelation error (lags 1–20, bounded by window length),
  cross-feature correlation error, and normalized power-spectrum L1 distance.
- **Utility:** identical ridge predictors (`alpha=1`, previous observation predicts
  the next) trained on real or synthetic windows and evaluated on real test windows.
  Training-window counts are matched. MAE/RMSE and synthetic-minus-real gaps are reported.
- **Copying/diversity:** exact duplicate fraction, nearest-training-window RMSE, and
  nearest-other-synthetic-window RMSE. Held-out-real-to-training distances provide context.

Metrics use training-standardized values, while descriptive mean/std/quantiles and
trajectory plots use original units. Fidelity distances should be small and variation
ratios near one, but there is no universal passing threshold or quality percentage.
Undefined metrics (for example constant-feature autocorrelation) are JSON `null`.
Copying diagnostics are not a privacy guarantee; reconstruction loss is not synthetic
quality. Use validation for exploratory model selection and reserve final test reports
for the chosen recipe to avoid tuning to the test set.

Metric window counts are bounded by `--max-metric-windows` and recorded in the report.
Real test windows use a seeded contiguous span (which can omit some series when
capped); other partitions use seeded samples. Copying references sample the actual
possibly-overlapping training windows, while utility uses non-overlapping windows.
The cap can miss duplicates or nearest matches outside the evaluated sample.
Bootstrap intervals resample series clusters, or blocks of chronological evaluation
windows for a single series, and independently resample generated sequences. These
intervals describe sampling variation, not variation across model-training seeds.
For seed sensitivity, train separate runs with different `--seed` values and compare
reports. Very small datasets yield weak or degenerate intervals.

Outputs: `metrics.json`, standalone `report.html`, and per-feature PNG plots.
Add `--format json` to any action for machine-readable CLI results.

## Research references

The initial model families follow [TimeVAE (2021)](https://arxiv.org/abs/2111.08095),
whose [author repository](https://github.com/abudesai/timeVAE) includes dense and
convolutional baselines. Future model candidates include
[KoVAE (ICLR 2024)](https://github.com/azencot-group/KoVAE),
[TimeVQVAE (AISTATS 2023)](https://github.com/ML4ITS/TimeVQVAE), and
[AEQ-RVAE-ST (2025, revised 2026)](https://arxiv.org/abs/2505.05020).
These models are not yet registered. Conditional generation, missing-value modeling,
irregular sampling, and variable-length sequences are outside the initial implementation.
