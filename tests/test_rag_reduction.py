import csv
import json

from mlx.core.artifacts import write_json_atomic
from mlx.modes.text_embedding.rag_datasets import PrepareRagEvaluationDatasets
from mlx.modes.text_embedding.rag_evaluation import EvaluateRagReduction, _normalize_answer
from mlx.modes.text_embedding.rag_statistics import AnalyzeRagReduction, VARIANTS


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class Generator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        assert "Paris" in prompt
        return "Paris."


def test_answer_normalization_preserves_decimal_and_ignores_terminal_period():
    assert _normalize_answer("The Paris.") == "paris"
    assert _normalize_answer("127.40%") == "127.4%"
    assert _normalize_answer("94.0") == _normalize_answer("94")


def test_rag_answers_are_paired_and_generation_is_cached(tmp_path):
    datasets = tmp_path / "datasets"
    root = datasets / "fixture"
    _jsonl(root / "corpus.jsonl", [
        {"_id": "d1", "title": "France", "text": "Paris is its capital."},
        {"_id": "d2", "title": "Other", "text": "Unrelated."},
    ])
    _jsonl(root / "queries.jsonl", [{"_id": "q1", "text": "Capital of France?"}])
    _jsonl(root / "answers.jsonl", [{"query_id": "q1", "answers": ["Paris"]}])
    (root / "qrels").mkdir()
    (root / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    write_json_atomic(root / "generation_queries.json", ["q1"])
    benchmark = tmp_path / "benchmark"
    write_json_atomic(benchmark / "experiment.json", {"experiment": {"datasets": ["fixture"]}})
    write_json_atomic(benchmark / "comparison/cells.json", {})
    write_json_atomic(benchmark / "comparison/stage.json", {})
    rows = [{"query_id": "q1", "results": [{"document_id": "d1", "rank": 1, "relevance": 1}]}]
    paths = {"full": "baseline/rankings.jsonl",
             **{name: f"runs/{name}-512/seed-42/eval-512/benchmark/rankings.jsonl"
                for name in VARIANTS if name not in ("full", "truncate")},
             "truncate": "runs/truncate-512/eval-512/benchmark/rankings.jsonl"}
    for path in paths.values():
        _jsonl(benchmark / "fixture" / path, rows)
    model = tmp_path / "generator.gguf"
    model.write_bytes(b"model")
    generator = Generator()
    command = EvaluateRagReduction(datasets, benchmark, tmp_path / "result", generator=generator,
                                   generator_model_path=model, top_k=1)
    result = command.execute()
    assert result["rows"] == result["retrieval_rows"] == 6
    assert result["unique_generations"] == generator.calls == 1
    command.execute()
    assert generator.calls == 1
    with (tmp_path / "result/query_results.csv").open() as source:
        assert all(int(row["supported_exact"]) == 1 for row in csv.DictReader(source))
    analysis = AnalyzeRagReduction({"gemma": tmp_path / "result"}, tmp_path / "statistics",
                                   bootstrap_draws=100).execute()
    assert analysis["top_k_overlap"][0]["same_top_k_fraction"] == 1


def test_rag_preparation_rejects_invalid_sample_counts(tmp_path):
    try:
        PrepareRagEvaluationDatasets(tmp_path, tmp_path / "out", questions_per_dataset=1).execute()
    except Exception as exc:
        assert "sample counts" in str(exc)
    else:
        raise AssertionError("Expected invalid sample size")
