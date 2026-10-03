#!/usr/bin/env python3
"""Run the frozen local-generator stage after a retrieval benchmark completes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx.modes.text_embedding.rag_evaluation import EvaluateRagReduction
from mlx.modes.text_embedding.rag_generation import LlamaCppRagGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    generator = LlamaCppRagGenerator(args.generator, max_tokens=protocol["generation_max_tokens"],
                                     seed=protocol["generation_seed"],
                                     temperature=protocol["generation_temperature"])
    result = EvaluateRagReduction(
        args.datasets, args.benchmark, args.output, generator=generator,
        generator_model_path=args.generator, seed=protocol["generation_seed"],
        top_k=protocol["rag_top_k"],
        passage_char_limit=protocol["context_char_limit_per_passage"],
    ).execute()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
