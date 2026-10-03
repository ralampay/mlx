#!/usr/bin/env python3
"""Summarize completed Gemma and Qwen RAG reduction evaluations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx.modes.text_embedding.rag_statistics import AnalyzeRagReduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = AnalyzeRagReduction(
        {"gemma": args.gemma, "qwen": args.qwen}, args.output,
        answer_margin=protocol["noninferiority_margin_answer_accuracy_absolute"],
    ).execute()
    print(json.dumps({"models": result["models"], "report": str(args.output / "report.md")}))


if __name__ == "__main__":
    main()
