# Autoencoder retrieval experiments

The `text-embedding --action benchmark-autoencoders` workflow compares original GGUF embeddings
with corpus-adapted autoencoder bottlenecks. See [ARCHITECTURE.md](../ARCHITECTURE.md) for command
boundaries and extension ownership.

Install the optional dependencies into the Python environment used to run MLX:

```bash
python -m pip install -e '.[retrieval-benchmark]'
```

The experiment runs on CPU. The provided launcher has explicit settings and can be run from any
working directory:

```bash
MLX_ROOT=/path/to/mlx
"$MLX_ROOT/scripts/test-autoencoders.sh"
```

The tracked launcher is [scripts/test-autoencoders.sh](../scripts/test-autoencoders.sh). Set
`MLX_ROOT`, `PYTHON_BIN`, `MODEL_PATH`, `DATASET_ROOT`, or `EXPERIMENT_OUTPUT` to override its defaults.
Additional CLI arguments are forwarded last, for example `--epochs 10 --output /path/to/pilot`.
Use a new output directory for changed settings; do not mix pilot and confirmatory runs.
The default output is `~/Desktop/experiments/gemma-ae-retrieval-v1`.

For the separate 384/512-dimensional experiment, run:

```bash
MLX_ROOT=/path/to/mlx
"$MLX_ROOT/scripts/test-autoencoders-384-512.sh"
```

This launcher uses hidden width 512 and writes to
`~/Desktop/experiments/gemma-ae-retrieval-384-512-v1`. All other protocol settings
remain the same (200 training runs and six statistical comparisons). The wider hidden
layer is required by the 512-dimensional bottleneck, so cross-run comparisons with
the 128/256 experiment also reflect a change in hidden-layer capacity.

## Dataset suite

`laptop-ae-v1` keeps full local `nfcorpus` and `scifact`, and downloads pinned revisions of
`zeta-alpha-ai/NanoArguAna`, `NanoFiQA2018`, `NanoSCIDOCS`, `NanoTouche2020`, `NanoQuoraRetrieval`,
`NanoDBPedia`, `NanoHotpotQA`, and `NanoMSMARCO` from Hugging Face. Preparation can also run separately:

```bash
python -m mlx --mode text-embedding --action prepare-datasets \
  --dataset-path "$HOME/Desktop/datasets/retrieval" --suite laptop-ae-v1
```

Directories use `nano-arguana`, `nano-fiqa2018`, `nano-scidocs`, `nano-touche2020`, `nano-quora`,
`nano-dbpedia`, `nano-hotpotqa`, and `nano-msmarco`. Each contains `corpus.jsonl`, `queries.jsonl`,
`qrels/test.tsv`, and `source_manifest.json`. Existing NFCorpus and SciFact files are not modified.

NanoBEIR uses positive query/document pairs without graded scores. Preparation assigns score 1,
retains all IDs and text, and explicitly maps the source's `train` storage split to evaluation
qrels. It does not train on those queries or qrels. Empty document strings present in NanoFiQA (27) and NanoSCIDOCS (344) are
preserved and embedded with the document prompt, not dropped or replaced with invented text.
This is an explicit experiment loader policy; ordinary BEIR loading remains strict.

The ten collections contain 46,228 documents and 1,022 judged evaluation queries. Additional
unjudged queries in the existing full datasets remain exported but are excluded from metrics.
Each dataset has its own ID namespace. Nano results must not be presented as full-BEIR results.
Sources and methodology: [NanoBEIR collection](https://huggingface.co/collections/zeta-alpha-ai/nanobeir),
[BEIR](https://github.com/beir-cellar/beir).

## Protocol and options

The launcher uses original normalized embeddings and `simple` autoencoders with `mse` and
`mse-similarity` (weight 1), dimensions 128/256, seeds 42–46, hidden width 256, batch size 64,
50 epochs, learning rate 0.001, and a seeded 80/20 document split. That is 200 training runs.
The two losses use identical initialization, splits, batch order, and singleton-merging policies
for each dataset/dimension/seed. Validation objective selects the checkpoint, never retrieval
performance. Both corpus and queries are encoded with that checkpoint and L2-normalized.
There are no relevance-supervised losses or native-truncation controls.

`--prompt-format embeddinggemma` formats queries as `task: search result | query: ...`, and documents
as `title: TITLE | text: BODY` (missing titles become `none`). `--pooling auto` uses GGUF metadata;
`--context-length 2048` configures llama.cpp context and batch capacity. Long inputs use the
binding's token-prefix truncation. There is no alternate pooling retry. Model identity, resolved
pooling, context length, and runtime versions are recorded. See the
[EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card).

`--vector-store exact` exhaustively scores cosine similarity, sorting ties by document ID.
`--exclude-self-matches` removes a corpus ID equal to the query ID before evaluation. This is
opt-in for standalone benchmarks and enabled for the suite. Original and compressed runs use
the same ranking policy. Existing metric definitions remain unchanged, including exponential
gain for graded nDCG and the existing MAP denominator convention.

Important flags:

| Flag | Experiment default |
|---|---|
| `--suite` | `laptop-ae-v1` |
| `--download-datasets` | Off; launcher enables it |
| `--autoencoder-model` | `simple` |
| `--losses` | `mse,mse-similarity` |
| `--bottleneck-dims` | `128,256` |
| `--seeds` | `42,43,44,45,46` |
| `--embedding-batch-size` | `1`; separate from training batch size |
| `--similarity-weight` | `1.0` |
| `--primary-metric` | `ndcg@10` |
| `--noninferiority-margin` | `0.01` absolute nDCG units |
| `--alpha` | `0.05` |
| `--resume` | Off; launcher enables it |

The initial ablation supports MSE and MSE-similarity, normalized vectors, exact search, and
nDCG@10 as its primary metric. Training runs sequentially and embedding runs once per dataset.

## Statistics and outputs

The primary hypothesis is non-inferiority relative to original embeddings: mean delta > -0.01.
Paired query differences are averaged over queries and seeds **within each dataset**. The ten
dataset means have equal weight in a one-sided t-test; query and seed repetitions do not increase
the dataset sample size. Holm correction applies jointly across all loss/dimension comparisons.
This assumes approximately normal, sufficiently independent dataset-level differences. It is
conditional on the selected mixed full/Nano suite and its corpus-adapted training protocol.

The secondary family tests MSE-similarity superiority over MSE at each dimension, with its own
Holm correction. Recall@100, MRR@10, MAP@100, and other requested cutoffs are descriptive.
Confidence intervals are unadjusted and labeled; conclusions use adjusted p-values. Degenerate
variance is explicitly inconclusive. A nonsignificant result is not proof of equivalence, nor
proof of degradation. Small collections cannot guarantee a decisive 0.01-margin result.

Read `comparison/report.md` and `comparison/statistics.json` for conclusions, `dataset_metrics.csv`
for paired dataset deltas and seed variability, `seed_metrics.csv` for compressed metrics/timings/vector
sizes, `baseline_metrics.csv` for corresponding original-vector measurements, and `paired_query_differences.csv` for auditable primary-metric differences. Per-cell
benchmark directories retain all requested metrics and rankings, and training directories retain
checkpoints and histories. Raw vector sizes exclude checkpoint and CSV storage overhead.

`experiment.json` fixes dataset/model hashes, configuration, versions, and relevant source hashes.
Stages publish atomically and record content hashes and durations. `--resume` reuses verified
complete stages and reruns interrupted stages from their beginning, including interrupted training.
Changed completed files or configuration require a new output directory. Failures stop execution,
leave a contextual `failure.json`, and cannot produce a complete-suite conclusion.
Run only one process against an experiment output directory at a time.

## Validation

```bash
python -m pytest tests/test_retrieval_experiment.py tests/test_text_embedding.py \
  tests/test_autoencoder.py tests/test_autoencoder_similarity.py

MLX_RETRIEVAL_SMOKE_MODEL=/path/to/embeddinggemma-300m-Q4_0.gguf \
  python -m pytest tests/test_retrieval_experiment.py::test_real_gguf_retrieval_smoke -q
```

The opt-in smoke test executes a tiny two-dataset experiment with the actual GGUF, both losses,
training, cached transforms, exact retrieval, and statistics. It is not scientific evidence for
the compression hypothesis.

## v2: structured models and reconstruction regularizers

The v2 launchers implement a pilot followed by a frozen confirmation experiment. The existing
`test-autoencoders-384-512.sh` remains the original two-loss experiment. Architecture and extension
contracts are documented in [ARCHITECTURE.md](../ARCHITECTURE.md#configured-autoencoder-retrieval-experiments-v2).

### Run the experiment

From this checkout, with the same Python environment and GGUF as the previous experiment:

```bash
bash scripts/test-autoencoders-v2-pilot.sh --dry-run
bash scripts/test-autoencoders-v2-pilot.sh

python -m mlx --mode text-embedding --action select-autoencoder-settings \
  --input "$HOME/Desktop/experiments/gemma-ae-v2-pilot" \
  --output "$HOME/Desktop/experiments/gemma-ae-v2-selection"

bash scripts/test-autoencoders-v2-384-512.sh --dry-run
bash scripts/test-autoencoders-v2-384-512.sh
```

The confirmation config is generated by selection; the confirmation launcher requires it to exist.
A dry run verifies inputs and reports counts without creating experiment outputs or training.
`--format json` returns machine-readable counts. Both launchers enable verified stage resume.
Use a new output directory after changing settings, code, or completed artifacts.

Overrides: `MLX_ROOT`, `PYTHON_BIN`, `MODEL_PATH`, `DATASET_ROOT`, `EXPERIMENT_CONFIG`,
`EXPERIMENT_OUTPUT`, and `EMBEDDING_SOURCE`. The default source is
`~/Desktop/experiments/gemma-ae-retrieval-384-512-v1`; only its original embeddings are reused,
never its trained checkpoints. Source stages remain read-only. Model/dataset hashes, content,
embedding settings and llama.cpp binding version must agree. Set `EMBEDDING_SOURCE=''` explicitly
to generate original embeddings in a new experiment instead. No dataset downloads happen by default.
Additional CLI options are forwarded last. Runtime training options override recipe defaults;
`--losses`, `--seeds`, `--bottleneck-dims`, `--autoencoder-model`, and `--similarity-weight`
cannot override an explicit recipe. Variants belong in the JSON recipe.

### Methods

| Variant | Architecture | Training objective |
|---|---|---|
| `mse` | Existing simple GELU MLP | MSE |
| `mse-similarity` | Existing simple GELU MLP | MSE plus batch sample-similarity matching; reference only |
| `spectral-mse` | Same MLP, spectral normalization on decoder linear layers | MSE |
| `least-volume` | Spectral MLP | MSE plus calibrated Least Volume |
| `covariance` | Existing simple MLP | MSE plus calibrated mean squared off-diagonal feature covariance |
| `ordered` | 512-wide MLP with supported prefixes | MSE through a sampled prefix; deterministic all-prefix validation |
| `pca` | Centered, unwhitened full-SVD PCA | Fit only on training-document rows |
| `truncate` | First 384/512 original coordinates | No fitting |

Sources: [Least Volume, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a1d20cc72a21ef971d7e49a90d8fa56f-Abstract-Conference.html),
[latent redundancy, 2024](https://doi.org/10.1016/j.patrec.2024.01.013), and
[ordered Re-Bottleneck, 2025](https://arxiv.org/abs/2507.07867).
These experiments adapt those ideas to frozen text embeddings; they are not reproductions of the
papers' retrieval results. The covariance penalty and loss calibration are explicit adaptations.
The ordered method uses reconstruction only, without Re-Bottleneck's discriminator or contrastive variant.

For a latent batch `Z` of shape `B × d`, the new penalties are:

- Least Volume: `exp(mean(log(std(Z, correction=1, dim=0) + 0.001)))`.
- Covariance: center columns, form `C = Zc.T @ Zc / (B-1)`, then average `C[i,j]**2` for `i != j`.

The total is `MSE + rho * scale * penalty`. Compute `scale = initial_MSE / initial_penalty`
once on the first at most 1,024 training rows in original corpus order, with the initialized model
in evaluation mode; persist it and never recalibrate on validation/test data. Nonfinite/nonpositive
calibration terms fail. A zero coefficient bypasses calibration and equals MSE. Covariance needs at
least two latent coordinates; both penalties need two batch rows. Trailing singleton batches are
merged consistently across methods. At batch size 64 the covariance matrix has rank at most 63;
this regularizer does not imply independent features or guaranteed preserved rankings.

Spectral parametrizations use five power iterations. This is not a claim of an exactly 1-Lipschitz
GELU decoder. The spectral-MSE control isolates architecture effects from the volume penalty.
Ordered training samples one prefix uniformly from `{128,256,384,512}` per batch using a separate
seeded generator; validation averages all four reconstruction losses. Export 384/512 prefixes from
one checkpoint. All methods normalize final corpus and query vectors before exact cosine search.

### Pilot and confirmation

The tracked pilot recipe uses NFCorpus, NanoQuora, and NanoFiQA2018, seeds 40/41, dimensions
384/512, and `rho` values 0.01, 0.1, and 1 for each new penalty. It runs **102 neural fits** and
**no retrieval evaluations**. Both stages retain hidden width 512, batch size 64, Adam at 0.001,
50 epochs, CPU, and the same seeded 80/20 corpus partition. Checkpoints use the method's own total
validation objective; reconstruction MSE is logged separately for selection.

Selection averages reconstruction ratios equally across datasets, dimensions, and seeds, relative
to spectral MSE for Least Volume and simple MSE for covariance. It chooses one global coefficient
per penalty; scores within 1e-8 tie in favor of the smaller coefficient. Incomplete/nonfinite pilots
cannot be selected. Boundary winners do not trigger automatic grid expansion. Preserve the original
pilot config file and artifacts for selection verification. A second selection needs a new output directory.

The confirmation uses all ten datasets and seeds 42–46. Five fixed-width AE variants each run 100
fits; ordered AE runs 50: **550 neural fits**. PCA adds **50 fits**, sharing each 512-component fit
across both dimensions. There are **730 retrieval evaluations**: 600 AE, 100 PCA, 20 truncation,
and 10 original-vector baselines. Hyperparameters target reconstruction, not retrieval; test queries
and relevance judgments are excluded from all model training and hyperparameter selection.

Primary analysis tests all eight compressed methods at both widths against original embeddings:
16 non-inferiority tests, margin 0.01 nDCG@10, Holm-adjusted together at alpha 0.05. A separate
8-test secondary family compares Least Volume versus spectral MSE, covariance versus MSE, ordered
versus MSE, and MSE-similarity versus MSE at both dimensions. Query/seed differences are averaged
within datasets; the sample size is ten datasets, not the number of fits or queries. Other retrieval
metrics are descriptive. Confidence intervals are unadjusted and degenerate cases inconclusive.

`comparison/report.md` and `statistics.json` contain conclusions. `training_runs.csv` counts shared
training once and reports checkpoint/PCA bytes separately. `seed_metrics.csv` records evaluation
times and vector bytes; `dataset_metrics.csv`, `baseline_metrics.csv`, and
`paired_query_differences.csv` support audit. `cells.json` maps evaluations back to training runs.
Checkpoints and training manifests retain loss calibration, formula version, validation components,
seed, split hash, and architecture configuration. An ordered checkpoint also supports standalone
export with `--mode autoencoder --action embed --output-dim 384` and the usual input/output flags.

### Validation

```bash
python -m pytest tests/test_autoencoder_v2.py tests/test_autoencoder.py \
  tests/test_autoencoder_similarity.py tests/test_text_embedding.py tests/test_retrieval_experiment.py -q

MLX_RETRIEVAL_SMOKE_MODEL="$HOME/workspace/ai_models/embedding_models/embeddinggemma-300m-Q4_0.gguf" \
  python -m pytest tests/test_autoencoder_v2.py::test_real_gguf_configured_smoke -q
```

The smoke test is a tiny integration check, not a scientific experiment.

## Orthogonal tied AE experiment

The `orthogonal-tied` model encodes with one orthonormal-row matrix and decodes with its
transpose. Training-only uncentered SVD initializes the model; signed QR restores the
constraint after every Adam step. The initialized checkpoint can win at epoch zero.
MSE alone is compared with `mse-cosine` (per-vector cosine error weighted by `1 / input_width`).
Neither objective requires pairwise training samples. Centered PCA, uncentered SVD, native
truncation, and original Gemma embeddings are controls. See [architecture](../ARCHITECTURE.md)
for extension contracts and process ownership.

The frozen recipe runs ten datasets, seeds 42–46, 384/512 dimensions, 50 epochs, batch 64,
and learning rate 0.001: 200 AE fits, 50 PCA fits, 50 SVD fits, and 430 retrieval evaluations.
This is exploratory because these datasets have already informed candidate selection.
The non-inferiority margin stays at 0.01 nDCG@10. Reports distinguish Holm-adjusted
non-inferiority from two-sided difference tests; nonsignificance is not equivalence.

```bash
# Foreground (also resumes verified completed stages)
bash scripts/test-autoencoders-orthogonal-384-512.sh

# Detached background job
bash scripts/start-autoencoders-orthogonal-384-512.sh
```

Both accept `PYTHON_BIN`, `MODEL_PATH`, `DATASET_ROOT`, `EMBEDDING_SOURCE`,
`EXPERIMENT_CONFIG`, and `EXPERIMENT_OUTPUT` environment overrides. Use `EXPERIMENT_OUTPUT`
instead of a trailing `--output`, so process locking and the experiment destination agree.
The repository is resolved from the script location. CPU thread defaults are one; explicit
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` settings are respected and
recorded by the background launcher. Cached Gemma embeddings are reused by default.

Default output: `$HOME/Desktop/experiments/gemma-ae-orthogonal-384-512-v1`.
Launch metadata lives beside it in a `.launches` directory. Each detached launch prints
its supervisor PID, timestamped log, and `status.json` path. Monitor with `tail -f` on that
log and inspect status for `child_pid`, `exit_code`, and `finished_at`. Send SIGTERM to the
supervisor PID for graceful interruption; rerun the same launcher to resume. An abrupt
SIGKILL may leave the child running; inspect the recorded child PID before restarting.
Never delete lock files to bypass an active job. Stale PID files do not block restarts.
The launcher is POSIX-only and requires Python, `fcntl`, Bash, and `nohup`.

Completed results are in `comparison/report.md`, `statistics.json`, `dataset_metrics.csv`,
`seed_metrics.csv`, and `training_runs.csv`. Do not edit completed stage files, because
resume verifies their hashes. A running job is not a completed statistical result.
