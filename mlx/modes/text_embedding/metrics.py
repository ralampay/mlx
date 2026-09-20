from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    _validate_k(k)
    return sum(value > 0 for value in relevances[:k]) / k


def recall_at_k(relevances: Sequence[int], total_relevant: int, k: int) -> float:
    _validate_k(k)
    if total_relevant <= 0:
        return 0.0
    return sum(value > 0 for value in relevances[:k]) / total_relevant


def reciprocal_rank_at_k(relevances: Sequence[int], k: int) -> float:
    _validate_k(k)
    for rank, relevance in enumerate(relevances[:k], start=1):
        if relevance > 0:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    relevances: Sequence[int], total_relevant: int, k: int
) -> float:
    _validate_k(k)
    denominator = min(total_relevant, k)
    if denominator <= 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, relevance in enumerate(relevances[:k], start=1):
        if relevance > 0:
            hits += 1
            total += hits / rank
    return total / denominator


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    _validate_k(k)
    return sum(
        (2.0 ** float(relevance) - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(
    relevances: Sequence[int], ideal_relevances: Sequence[int], k: int
) -> float:
    ideal = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    return dcg_at_k(relevances, k) / ideal if ideal else 0.0


@dataclass(frozen=True)
class QueryMetricResult:
    values: Mapping[str, float]
    relevant_count: int
    retrieved_relevant: Mapping[int, int]
    best_relevant_rank: int | None


def compute_query_metrics(
    retrieved_document_ids: Sequence[str],
    qrels: Mapping[str, int],
    k_values: Sequence[int],
) -> QueryMetricResult:
    relevances = [max(0, int(qrels.get(identifier, 0))) for identifier in retrieved_document_ids]
    ideal = [int(value) for value in qrels.values() if value > 0]
    relevant_count = len(ideal)
    values: dict[str, float] = {}
    retrieved_relevant: dict[int, int] = {}
    for k in k_values:
        retrieved_relevant[k] = sum(value > 0 for value in relevances[:k])
        values[f"precision@{k}"] = precision_at_k(relevances, k)
        values[f"recall@{k}"] = recall_at_k(relevances, relevant_count, k)
        values[f"mrr@{k}"] = reciprocal_rank_at_k(relevances, k)
        values[f"map@{k}"] = average_precision_at_k(relevances, relevant_count, k)
        values[f"ndcg@{k}"] = ndcg_at_k(relevances, ideal, k)
    best_rank = next(
        (rank for rank, value in enumerate(relevances, start=1) if value > 0),
        None,
    )
    return QueryMetricResult(values, relevant_count, retrieved_relevant, best_rank)


def aggregate_query_metrics(results: Sequence[QueryMetricResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].values.keys()
    return {
        key: sum(float(result.values[key]) for result in results) / len(results)
        for key in keys
    }


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


__all__ = [
    "QueryMetricResult",
    "aggregate_query_metrics",
    "average_precision_at_k",
    "compute_query_metrics",
    "dcg_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
