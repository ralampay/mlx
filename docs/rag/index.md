# RAG Utilities

## Overview

The `rag` module orchestrates ingestion, vectorization, and query workflows around a single collection. Every command is executed via `mlx --module rag` plus an `--action` flag. The module currently targets a Chroma-compatible backend (local or remote) but tags each record with the originating platform so you can filter results per backend.

## Environment Requirements

Ensure the following environment variables are populated, preferably via `.env` (you can copy `.env.dist` and edit the values):

- `DB_ADAPTER`: Vector database adapter (`chromadb` is supported out of the box).
- `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`: Connection details when `DB_ADAPTER=chromadb`.
- `HUGGINGFACE_TOKEN`: Needed when using Hugging Face hosted embeddings or generators.
- `LOCAL_LLM_MODEL`: Path to a GGUF weights file used by offline modes.
- `LOCAL_LLM_GENERATION_MODEL`: Optional gguf for generation; falls back to `LOCAL_LLM_MODEL` when unset.

## Vectorization Summary

Vectorization scans the dataset directory, chunks PDF/TXT sources, and reports dataset statistics without writing to the database:

```bash
mlx --module rag \
    --action vectorization-summary \
    --chunk-size 800 \
    --chunk-overlap 100 \
    --dataset-path ./datasets/rag \
    --table-name demo_collection \
    --file-limit 50 \
    --local
```

- `--dataset-path`: Directory with the documents you want to ingest (supports `.txt`, `.pdf`, etc.).
- `--table-name`: Target collection/Chroma table for storing vectors (required for every command).
- `--chunk-size`, `--chunk-overlap`: Control document chunking.
- `--file-limit`: Cap the number of files processed in this preview pass.
- `--local`: Use the local GGUF model listed in `LOCAL_LLM_MODEL` instead of hosted embeddings.

To evaluate cloud embeddings before ingesting vectors, specify `--platform openai` (or `--platform huggingface`) and pass the corresponding `--model`:

```bash
mlx --module rag \
    --action vectorization-summary \
    --platform openai \
    --model text-embedding-3-large \
    --dataset-path ./datasets/rag \
    --table-name demo_collection
```

```bash
mlx --module rag \
    --action vectorization-summary \
    --platform huggingface \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset-path ./datasets/rag \
    --table-name demo_collection
```

## Batch Insert

Once you are happy with chunking, write the vectors to the collection:

```bash
mlx --module rag \
    --action batch-insert \
    --chunk-size 800 \
    --chunk-overlap 100 \
    --dataset-path ./datasets/rag \
    --table-name demo_collection \
    --file-limit 50 \
    --local
```

Switching to hosted embeddings uses the same pattern as the summary command—just add `--platform`/`--model`. The process displays per-file progress and logs the destination collection plus model metadata inside the insertion summary.

## Querying

Querying loads the top-k nearest vectors from the named table and optionally forwards the context to a generator:

```bash
mlx --module rag \
    --action query \
    --platform openai \
    --model gpt-4o-mini \
    --table-name demo_collection \
    --top-k 5
```

- When `--platform openai` is used, embeddings and final responses are produced by OpenAI APIs.
- `--local` directs the pipeline to load embeddings from the local GGUF model while still calling OpenAI for response generation.
- Use `--model-generator` to separate the embedding model (`--model`) from the generator (hosts like OpenAI/Hugging Face).
- The Hugging Face example below uses `mistralai/Mistral-7B-Instruct-v0.2` for both embeddings and generation, relying on `HUGGINGFACE_TOKEN`:

```bash
mlx --module rag \
    --action query \
    --platform huggingface \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --model-generator mistralai/Mistral-7B-Instruct-v0.2 \
    --table-name demo_collection \
    --top-k 5
```

For fully-local responses skip the hosted platform flags and rely on `LOCAL_LLM_GENERATION_MODEL` or `LOCAL_LLM_MODEL`.

## Cleanup

Clear a collection before re-running ingestion with:

```bash
mlx --module rag \
    --action delete-all \
    --table-name demo_collection
```

This currently works for `chromadb` tables.

> **Note:** Every `rag` command requires `--table-name` to select the collection that backs the workflow.
