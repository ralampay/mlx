#!/usr/bin/env bash
set -euo pipefail
MLX_ROOT="${MLX_ROOT:-$HOME/workspace/mlx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$HOME/workspace/ai_models/embedding_models/embeddinggemma-300m-Q4_0.gguf}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/Desktop/datasets/retrieval}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG or use a pilot/confirmation launcher}"
EXPERIMENT_OUTPUT="${EXPERIMENT_OUTPUT:?Set EXPERIMENT_OUTPUT or use a pilot/confirmation launcher}"
EMBEDDING_SOURCE="${EMBEDDING_SOURCE-$HOME/Desktop/experiments/gemma-ae-retrieval-384-512-v1}"
source_options=()
if [[ -n "$EMBEDDING_SOURCE" ]]; then
  source_options=(--embedding-source "$EMBEDDING_SOURCE")
fi
cd -- "$MLX_ROOT"
exec "$PYTHON_BIN" -m mlx \
  --mode text-embedding --action benchmark-autoencoders \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --dataset-path "$DATASET_ROOT" --model "$MODEL_PATH" \
  --embedding-backend llama-cpp --prompt-format embeddinggemma \
  --pooling auto --context-length 2048 --embedding-batch-size 1 \
  --normalize-embeddings --vector-store exact --exclude-self-matches \
  --primary-metric ndcg@10 --noninferiority-margin 0.01 --alpha 0.05 \
  --top-k 100 --k-values 10,100 --metrics ndcg,recall,mrr,map \
  --output "$EXPERIMENT_OUTPUT" --resume \
  "${source_options[@]}" "$@"
