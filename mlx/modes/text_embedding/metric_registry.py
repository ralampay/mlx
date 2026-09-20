"""Retrieval metric selection; formulas remain in metrics.py."""
from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping
from mlx.core.exceptions import MLXUserError
from mlx.core.extensions import load_reference


@dataclass(frozen=True)
class RetrievalMetricContext:
    relevances: tuple[int, ...]
    ideal_relevances: tuple[int, ...]
    relevant_count: int


def precision(context, k):
    from .metrics import precision_at_k
    return precision_at_k(context.relevances, k)


def recall(context, k):
    from .metrics import recall_at_k
    return recall_at_k(context.relevances, context.relevant_count, k)


def mrr(context, k):
    from .metrics import reciprocal_rank_at_k
    return reciprocal_rank_at_k(context.relevances, k)


def average_precision(context, k):
    from .metrics import average_precision_at_k
    return average_precision_at_k(context.relevances, context.relevant_count, k)


def ndcg(context, k):
    from .metrics import ndcg_at_k
    return ndcg_at_k(context.relevances, context.ideal_relevances, k)


def hit_rate(context, k):
    return float(any(value > 0 for value in context.relevances[:k]))


DEFAULT_METRICS = ("precision", "recall", "mrr", "map", "ndcg")


@dataclass(frozen=True)
class RetrievalMetricRegistry:
    entries: Mapping[str, str | Callable] = field(default_factory=lambda: {
        "precision": precision, "recall": recall, "mrr": mrr,
        "map": average_precision, "ndcg": ndcg, "hit-rate": hit_rate,
    })

    def __post_init__(self):
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def register(self, name: str, metric) -> "RetrievalMetricRegistry":
        if not name.strip() or "@" in name:
            raise ValueError("Metric names must be non-empty and exclude '@'.")
        return RetrievalMetricRegistry({**self.entries, name.strip().lower(): metric})

    def resolve(self, name: str):
        reference = self.entries.get(name)
        if reference is None:
            raise MLXUserError(f"Unsupported retrieval metric '{name}'. Available: {', '.join(sorted(self.entries))}.")
        metric = load_reference(reference, kind="retrieval metric") if isinstance(reference, str) else reference
        if not callable(metric):
            raise MLXUserError(f"Retrieval metric '{name}' must be callable.")
        return metric


DEFAULT_METRIC_REGISTRY = RetrievalMetricRegistry()
