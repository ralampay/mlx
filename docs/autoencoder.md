# Vector Autoencoders

The `autoencoder` mode trains reconstruction models over fixed-width numeric vectors and exports
their bottleneck representations. It consumes MLX-compatible CSV files whose `embedding` column is
a non-empty JSON array. All other columns and row order are preserved during export.

## Train

```bash
python -m mlx --mode autoencoder --action train \
  --model simple \
  --input ./artifacts/scifact/corpus_embeddings.csv \
  --output ./artifacts/autoencoder-128 \
  --input-dim 384 --hidden-dim 256 --bottleneck-dim 128 \
  --loss mse --epochs 50 --batch-size 64 --val-ratio 0.2 --seed 42
```

`--input-dim` is optional and otherwise inferred; when supplied it is a strict assertion. The
starter `simple` model is a symmetric GELU MLP with a linear bottleneck and linear reconstruction
output. Reconstruction width always equals input width. The bottleneck must be smaller than the
input, and the hidden layer must be at least as wide as the bottleneck.

Rows are deterministically split into training and validation partitions. Adam with learning rate
`1e-3` is the initial optimizer. The output directory contains `autoencoder.pth`, `training.csv`,
optional `training_history.png`, `autoencoder_manifest.json`, and `run_metadata.json`.

Available losses are shown with:

```bash
python -m mlx --mode autoencoder --action ls-loss-functions
```

Built-ins are `mse`, `mae`, `smooth-l1`, and `mse-similarity`. Smooth L1 accepts a positive
`beta` through a JSON configuration file:

```json
{"beta": 0.5}
```

```bash
python -m mlx --mode autoencoder --action train ... \
  --loss smooth-l1 --loss-config ./smooth-l1.json
```

## Similarity-preserving reconstruction

`mse-similarity` adds a latent-space geometry term to reconstruction MSE. For a batch
of original vectors `X`, encoder outputs `Z`, and decoded vectors `X_hat`, the objective is:

```text
MSE(X_hat, X) + similarity_weight * mean_{i != j}[(cos(X_i, X_j) - cos(Z_i, Z_j))^2]
```

It uses ordinary embedding CSVs, without query/document pairs or relevance labels. Original
similarities are detached targets. Feature dimensions can differ; each similarity matrix is
batch-by-batch. Normalization applies only inside the similarity term and does not change
reconstruction targets or the saved input-normalization contract. Zero vectors use an epsilon
norm floor (`1e-8`), giving zero similarity; they carry no meaningful angular target.

```bash
python -m mlx --mode autoencoder --action train \
  --model simple --input ./artifacts/scifact/corpus_embeddings.csv \
  --output ./artifacts/autoencoder-similarity-128 \
  --hidden-dim 256 --bottleneck-dim 128 --loss mse-similarity \
  --epochs 50 --batch-size 64 --val-ratio 0.2 --seed 42
```

The default `similarity_weight` is `1.0`. To change it, pass `--loss-config` with a JSON
file such as `{"similarity_weight": 0.1}`. The weight must be finite and nonnegative;
unknown configuration keys are rejected. Zero weight reduces to ordinary MSE with the
existing batching behavior. The effective configuration is saved in checkpoint/manifest metadata.

For positive weight, batch size and both split sizes must be at least two. If needed, provide
more vectors or adjust `--val-ratio`. A trailing singleton is merged into the preceding batch
in both training and validation, retaining every sample; this can exceed `--batch-size` by one.
Training shuffles with the configured seed, while validation order is fixed. Memory usage for
similarity matrices is quadratic in batch size. Half-precision similarity calculations are
promoted to float32, including under autocast.

`train_loss` and `val_loss` record the combined objective, averaged across batches by sample
count. Best-checkpoint selection uses that validation objective, not a retrieval metric.
Compare original embeddings, MSE-only compression, and similarity-preserving compression at the
same bottleneck dimension using held-out retrieval evaluation. Batch composition affects the
geometry term, so keep batch size and seed fixed for comparisons. Better retrieval is not guaranteed.

This is a cosine-matrix adaptation of
[similarity-preserving knowledge distillation (Tung and Mori, 2019)](https://arxiv.org/abs/1907.09682)
and [relational knowledge distillation (Park et al., 2019)](https://arxiv.org/abs/1904.05068),
combined with autoencoder reconstruction. It is not an exact reproduction of either paper's loss.

## Normalization

If the input is a canonical `corpus_embeddings.csv` or `query_embeddings.csv`, MLX reads its
sibling `embedding_manifest.json`. An already-normalized source is recorded as such. For generic
embedding CSVs, use `--normalize-inputs` to apply L2 normalization before reconstruction training.
The effective choice is stored in the checkpoint and applied automatically whenever it encodes
new vectors. `--no-normalize-inputs` is rejected when the source manifest says vectors are already
normalized because normalization cannot be reversed.

## Export bottleneck vectors

```bash
python -m mlx --mode autoencoder --action embed \
  --model simple \
  --model-path ./artifacts/autoencoder-128/autoencoder.pth \
  --input ./artifacts/scifact/query_embeddings.csv \
  --output ./artifacts/scifact/query_embeddings_ae128.csv
```

The checkpoint is authoritative. An explicit `--model` must match it. The result preserves all
source columns, replaces `embedding` with the bottleneck vector, and writes
`query_embeddings_ae128.manifest.json`. `--normalize-embeddings` optionally L2-normalizes the final
bottleneck representation.

For a complete retrieval artifact, pass the checkpoint to `text-embedding --adapter`. That command
transforms corpus and query vectors identically, writes their latent CSVs, indexes the latent corpus
in Chroma, and records adapter SHA-256 and dimension provenance in `embedding_manifest.json`.

## Custom models

An external model reference uses `package.module:DefinitionClass`. The class has a no-argument
constructor and provides `name`, `description`, and `build(config)`. The returned `torch.nn.Module`
must expose integer `input_dimensions` and `bottleneck_dimensions`, reconstruct its input shape in
`forward`, and implement `encode` and `decode`.

```python
class MyAutoencoderDefinition:
    name = "my-autoencoder"
    description = "A custom vector reconstruction model."

    def build(self, config):
        return MyAutoencoder(
            input_dimensions=config["input_dimensions"],
            bottleneck_dimensions=config["bottleneck_dimensions"],
        )
```

Use it as `--model my_package.autoencoders:MyAutoencoderDefinition`. Additional JSON-object options
may be supplied with `--autoencoder-config`; dimension keys remain controlled by dedicated flags.
Python applications may instead create an immutable `AutoencoderRegistry` with `register(...)` and
inject it into `TrainAutoencoder`.

## Custom losses

A custom loss definition also has a no-argument constructor and provides `name`, `description`, and
`build(config)`. It must return a scalar-producing `torch.nn.Module`:

```python
class WeightedReconstructionLossDefinition:
    name = "weighted-reconstruction"
    description = "Feature-weighted squared reconstruction loss."

    def build(self, config):
        return WeightedReconstructionLoss(weights=config["weights"])
```

Select it with
`--loss my_package.losses:WeightedReconstructionLossDefinition --loss-config ./weights.json`.
Built modules may opt into latent-aware training with `requires_latent = True` and a
`forward(reconstruction, target, *, latent)` signature. The trainer then calls `encode()` once
and decodes that same latent tensor; such models must implement their reconstruction path as
`decode(encode(inputs))`. Existing two-argument losses continue using `model(inputs)`.
A module may declare `minimum_batch_size = 2` to request singleton-merging batches (default: 1;
only 1 and 2 are supported). Definitions may provide a `default_config` mapping, merged with
explicit options before construction and serialization. These optional capabilities work for
registry-injected and import-path definitions without loss-name checks in the trainer.

Applications can also extend and inject `ReconstructionLossRegistry`. Import-path extensions are
explicit trusted Python code and must remain importable whenever their checkpoints are loaded.
## Checkpoint trust

Checkpoints use restricted tensor loading. Built-in models load without extra flags. A checkpoint
that names external architecture code requires `--trust-checkpoint-code` for both autoencoder
embedding and text-embedding `--adapter`, or an exact reference in a caller-injected registry.
Only authorize code you trust: the flag permits Python imports and is not a sandbox. It never
enables unrestricted pickle loading. See the [autoencoder extension tutorial](tutorials/adding-an-autoencoder.md).
