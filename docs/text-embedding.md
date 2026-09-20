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
with `Llama(model_path=..., embedding=True)` by default. Explicit pooling adds a
`pooling_type` override. Token-level output is rejected because retrieval
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
  --pooling mean \
  --prompt-format e5
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
  --pooling mean --prompt-format e5 \
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

## Pooling and model-specific formatting

Sequence pooling reduces token representations to one fixed-width vector for a query or document.
Some valid GGUF embedding models have missing or incompatible pooling metadata; an explicit
`--pooling` can let llama.cpp load them with the architecture's intended pooling method.

| Option | Meaning |
| ------ | ------- |
| `auto` | Let GGUF/llama.cpp determine pooling |
| `mean` | Mean pooling over token representations |
| `cls` | Use CLS/token-specific pooling |
| `last` | Use the final token representation |
| `none` | No sequence pooling |

`auto` is the default and omits the constructor's pooling override, preserving existing model
loading. MLX never retries with mean or any other method. Choose the method intended by the
original model architecture: changing pooling changes the representation and experimental results.
`none` is passed to llama.cpp, but token-level output is rejected by this retrieval workflow.
MLX never manually averages token embeddings.

`--prompt-format` accepts `auto`, `none`, or `e5`. Both `auto` (the default) and `none` leave the
text unchanged unless you supply the existing custom prefixes. Automatic family detection is
not attempted: filenames and architecture identifiers are not reliable evidence of prompt conventions.
For multilingual-e5-small, multilingual-e5-base, and multilingual-e5-large, use `e5` to prepend
`query: ` to queries and `passage: ` to documents, including non-English inputs. The document
prefix precedes the combined title/body. Other models receive no E5 prefixes automatically.
E5 formatting cannot be combined with nonempty `--query-prefix` or `--document-prefix`; choose
one formatting method to prevent double prefixes. Input text is not inspected or stripped for
existing prefixes, so pass unprefixed BEIR text when using the preset.

```bash
python -m mlx --mode text-embedding --action embed \
  --model /home/ralampay/workspace/ai_models/embedding_models/multilingual-e5-small-q8_0.gguf \
  --pooling mean --prompt-format e5 \
  --input ~/Desktop/datasets/retrieval/scifact \
  --output ~/Desktop/experiments/e5-scifact

# Existing Gemma invocation: automatic pooling and unchanged text formatting.
python -m mlx --mode text-embedding --action embed \
  --model /path/to/working-gemma-model.gguf \
  --input ~/Desktop/datasets/retrieval/scifact \
  --output ~/Desktop/experiments/gemma-scifact

python -m mlx --mode text-embedding --action benchmark \
  --input ~/Desktop/experiments/e5-scifact \
  --output ~/Desktop/experiments/e5-scifact-results
```

### Reproducibility and compatibility

Embedding and run manifests include `embedding_configuration` with `model_path`, `model_filename`,
`pooling_requested`, `pooling_effective`, `embedding_dimension`, `context_length`,
`prompt_format_requested`, `prompt_format_effective`, and `llama_cpp_python_version`.
The full model path records provenance; it is not required to reload the portable artifacts.
`embedding_dimension` is the source model dimension; existing `embedding.dimensions` describes
final vectors after any representation transform. Existing hashes, normalization, actual prefixes,
and adapter provenance remain recorded.

Effective pooling comes from the runtime library accessor, not the requested option. If unavailable,
auto records `model/default` and explicit pooling records `unknown`. Unavailable context length or
package version is null. Third-party providers without runtime metadata record unknown pooling.
Benchmark JSON manifests/summaries retain the saved configuration; CSV and Markdown reports show
requested/effective pooling and prompt format. Old manifests remain readable with unknown provenance.
Benchmarking never reloads the model or applies a new pooling choice.

The inspected llama-cpp-python 0.3.35 API supports `pooling_type`, named pooling constants,
`pooling_type()` for resolved pooling, and `n_ctx()` for runtime context length. Dependencies remain
unpinned; no historical minimum version is asserted. Older/incompatible builds missing a requested
constant fail with upgrade guidance. GGUF architecture support still depends on the installed
llama.cpp build; explicit pooling cannot repair every model-loading failure. Add `--verbose` to
retain the chained Python traceback on stderr alongside the original underlying error.
The new nondefault options apply to BEIR embedding, not the retained legacy CSV compatibility route.
