#!/usr/bin/env bash
set -euo pipefail

python -m mlx \
    --mode object_detection \
    --platform local \
    --action train \
    --provider libreyolo \
    --model yolox-m \
    --dataset-s3-uri s3://mlx-object-detection-datasets/foundational-urban.zip \
    --output ./runs/foundational-urban-yolox-m \
    --run-name foundational-urban-yolox-m \
    --device 0 \
    --epochs 50 \
    --batch-size 8 \
    --height 640 \
    --width 640 \
    --pretrained \
    --optimizer auto \
    --amp \
    --use-best \
    --random-seed 42 \
    --validate-after-training \
    --validation-split val
