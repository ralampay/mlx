# Text Embedding and Retrieval Benchmarking

The `text_embedding` mode creates reproducible embedding artifacts from BEIR datasets and
benchmarks retrieval against their relevance judgments. The CLI also accepts `text-embedding`;
`nlp` remains an alias for compatibility with the earlier CSV embedding command.

## Installation and model requirements

Install the optional llama.cpp and Chroma integration:

```bash
python -m pip install ".[text-embedding]"
```

The model must be a local `.gguf` file that supports llama.cpp sequence embeddings. MLX loads it
with `Llama(model_path=..., embedding=True)`. Token-level output is rejected because retrieval
requires one fixed-width vector per query or document. No GPU is assumed.

## BEIR dataset layout

SciFact, NFCorpus, and ArguAna use the same input contract:

```text
dataset/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    └── test.tsv
```

Corpus lines contain `_id`, `title`, and `text`; query lines contain `_id` and `text`. The qrels
file contains tab-separated query ID, corpus ID, and integer relevance, with an optional header.
MLX rejects malformed JSONL, duplicate IDs, and judgments that reference unknown IDs. A document
with a title is embedded as `title + "\n" + text`; otherwise only `text` is used.

Place each downloaded dataset at a descriptive path, for example `./datasets/scifact`,
`./datasets/nfcorpus`, or `./datasets/arguana`, and pass that directory through `--input`.

## Create embeddings

```bash
python -m mlx \
  --mode text-embedding \
  --action embed \
  --model ./models/multilingual-e5-small-q8_0.gguf \
  --input ./datasets/scifact \
  --output ./artifacts/scifact-e5-small \
  --query-prefix "query: " \
  --document-prefix "passage: "
```

Prefixes default to empty strings. They are configurable because E5-style models commonly require
`query: ` and `passage: ` while other models do not. Use `--normalize-embeddings` to apply L2
normalization after the provider returns vectors. The native output is preserved by default.
`--batch-size` controls llama.cpp embedding batches. All choices are recorded in the manifest.

The output contains `corpus_embeddings.csv`, `query_embeddings.csv`, persistent `vector_store/`,
`dataset_manifest.json`, `embedding_manifest.json`, copied `qrels.tsv`, and `run_metadata.json`.
Each CSV stores its vector as one JSON array, which makes vectors easy to load and transform for
future dimensionality-reduction experiments. `representation` defaults to `original` and can be
overridden without encoding assumptions about PCA or autoencoders.

Chroma receives the vectors already produced by llama.cpp and is never configured with its own
embedding function. It persists a cosine collection and document metadata under `vector_store/`.

### Autoencoder representations

A trained vector autoencoder may be inserted after GGUF embedding with `--adapter`:

```bash
python -m mlx --mode text-embedding --action embed \
  --model ./models/multilingual-e5-small-q8_0.gguf \
  --adapter ./artifacts/autoencoder-128/autoencoder.pth \
  --input ./datasets/scifact --output ./artifacts/scifact-e5-ae128
```

For checkpoints naming custom Python architecture code, add `--trust-checkpoint-code` only if
you trust that code. Built-in architectures do not require this flag.

The checkpoint controls any L2 preprocessing expected by the autoencoder. MLX encodes both corpus
and query vectors through the bottleneck, then applies the optional final
`--normalize-embeddings` step. Exported CSVs and Chroma therefore use the same latent vectors.
The embedding manifest records source and final dimensions plus the adapter architecture, hash,
preprocessing contract, and loss provenance. The default representation is
`autoencoder-<bottleneck-dimension>` unless `--representation` is explicit.

## Benchmark retrieval

```bash
python -m mlx \
  --mode text-embedding \
  --action benchmark \
  --input ./artifacts/scifact-e5-small \
  --output ./results/scifact-e5-small
```

Benchmarking loads query vectors and copied qrels from the embedding artifacts, opens the existing
vector store, and does not load or execute the GGUF model. The default retrieval depth is 100 and
metric cutoffs are 1, 5, 10, 20, and 100. Override them with `--top-k` and a comma-separated
`--k-values`; the maximum cutoff cannot exceed the retrieval depth.

Only exported queries present in the selected qrels are evaluated. Unjudged queries are excluded
from aggregate metrics and failure lists; `excluded_queries` records their count. This matters for
BEIR exports containing queries from multiple splits. Judged queries with only nonpositive labels
remain in the cohort with zero relevance-based metrics. Corrupt manifests, incompatible cosine
indexes, and invalid rankings fail with actionable errors.

The result directory contains `metrics.json`, one-row `metrics.csv`, `query_metrics.csv`,
`rankings.jsonl`, `failures.csv`, `benchmark_manifest.json`, `run_metadata.json`, and `report.md`.
The per-query and ranking artifacts support later statistical analysis; failures contain queries
with no relevant result in the configured retrieval depth.

Metrics use positive qrels as relevant documents. Precision@K divides relevant hits by K;
Recall@K divides hits by all relevant documents; MRR@K uses the first relevant rank; MAP@K averages
precision at relevant ranks with `min(relevant_count, K)` as its denominator; DCG uses graded gain
`2^relevance - 1`; nDCG divides by the ideal graded ranking. Chroma cosine distances are converted
inside the adapter to best-first, higher-is-better similarity scores.

## Extension points

Embedding commands depend only on `TextEmbeddingProvider`; llama.cpp is one adapter. Vector
indexing and retrieval depend only on `VectorStore` and immutable registry-selected factories.
A future `PgVectorStore` can implement `add`, `query`, and `close`, normalize its results to
best-first higher-is-better scores, and register one lazy factory. The embedding and benchmark
commands, BEIR loader, metrics, and artifact writers require no changes.

Future PCA tooling can implement the same core vector-transform contract used by autoencoder
checkpoints without entering dataset parsing, metrics, or vector-store adapters.
