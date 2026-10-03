#!/usr/bin/env bash
set -euo pipefail

# Override these environment variables to use a different checkout, Python, or output.
MLX_ROOT="${MLX_ROOT:-$HOME/workspace/mlx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$HOME/workspace/ai_models/embedding_models/embeddinggemma-300m-Q4_0.gguf}"
EXPERIMENT_OUTPUT="${EXPERIMENT_OUTPUT:-$HOME/Desktop/experiments/gemma-ae-retrieval-v1}"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename -- "$SCRIPT_DIRECTORY")" == bin ]]; then
  DATASET_ROOT="${DATASET_ROOT:-$(dirname -- "$SCRIPT_DIRECTORY")}"
else
  DATASET_ROOT="${DATASET_ROOT:-$HOME/Desktop/datasets/retrieval}"
fi
cd -- "$MLX_ROOT"

exec "$PYTHON_BIN" -m mlx \
  --mode text-embedding \
  --action benchmark-autoencoders \
  --suite laptop-ae-v1 \
  --dataset-path "$DATASET_ROOT" \
  --download-datasets \
  --model "$MODEL_PATH" \
  --embedding-backend llama-cpp \
  --prompt-format embeddinggemma \
  --pooling auto \
  --context-length 2048 \
  --embedding-batch-size 1 \
  --normalize-embeddings \
  --vector-store exact \
  --exclude-self-matches \
  --autoencoder-model simple \
  --losses mse,mse-similarity \
  --similarity-weight 1.0 \
  --bottleneck-dims 128,256 \
  --hidden-dim 256 \
  --seeds 42,43,44,45,46 \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --val-ratio 0.2 \
  --primary-metric ndcg@10 \
  --noninferiority-margin 0.01 \
  --alpha 0.05 \
  --top-k 100 \
  --k-values 10,100 \
  --metrics ndcg,recall,mrr,map \
  --device cpu \
  --output "$EXPERIMENT_OUTPUT" \
  --resume \
  "$@"
