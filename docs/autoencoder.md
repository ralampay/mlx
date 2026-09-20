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

Built-ins are `mse`, `mae`, and `smooth-l1`. Smooth L1 accepts a positive `beta` through a JSON
configuration file:

```json
{"beta": 0.5}
```

```bash
python -m mlx --mode autoencoder --action train ... \
  --loss smooth-l1 --loss-config ./smooth-l1.json
```

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
Applications can also extend and inject `ReconstructionLossRegistry`. Import-path extensions are
explicit trusted Python code and must remain importable whenever their checkpoints are loaded.
