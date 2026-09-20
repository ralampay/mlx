# Adding a retrieval metric

## Contract and implementation

Retrieval metric functions accept a `RetrievalMetricContext` plus cutoff `k` and return a
finite scalar. Context contains ranked relevances, ideal relevances, and relevant-document count.
It contains no Chroma objects or terminal formatting.

This tiny custom hit metric asks whether any relevant result appears in the first k positions:

```python
from mlx.modes.text_embedding.metric_registry import RetrievalMetricRegistry
from mlx.modes.text_embedding.metrics import compute_query_metrics

def found_relevant(context, k):
    return float(any(value > 0 for value in context.relevances[:k]))

registry = RetrievalMetricRegistry({}).register("found", found_relevant)
result = compute_query_metrics(["wrong", "right"], {"right": 1}, (1, 2),
                               metric_names=("found",), registry=registry)
assert result.values == {"found@1": 0., "found@2": 1.}
assert tuple(registry.entries) == ("found",)
```

## Registration, configuration, and execution

Place a permanent implementation in the text-embedding metric package or an external module;
register its callable/import reference in that mode's registry. Pass `metric_registry=registry`
and request `metrics=("found",)` to `BenchmarkTextEmbeddingCommand`.
Built-ins are discoverable using `--action ls-metrics` and selected with `--metrics`.
A Python-only registration does not modify a future CLI process.

## Files and expansion

Only the metric, its registration, and focused tests change. The example above is a
hand-checkable unit test. Add more relevance logic if necessary; do not edit vector stores,
embedding providers, reporters, or unrelated intrinsic algorithm metrics.
