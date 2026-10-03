"""End-to-end answer evaluation from completed, fixed RAG retrieval rankings."""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from mlx.core.artifacts import sha256_file, write_csv, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.data import BeirDatasetLoader


def _normalize_answer(value):
    value = value.lower().strip().replace(",", "")
    value = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", value)
    value = "".join(character for character in value if character not in string.punctuation or character in ".-%")
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    normalized = " ".join(value.split())
    if re.fullmatch(r"-?\d+(?:\.\d+)?%?", normalized):
        suffix = "%" if normalized.endswith("%") else ""
        try:
            return format(Decimal(normalized.removesuffix("%")).normalize(), "f") + suffix
        except InvalidOperation:
            pass
    return normalized


def _token_f1(prediction, reference):
    left, right = _normalize_answer(prediction).split(), _normalize_answer(reference).split()
    if not left or not right:
        return float(left == right)
    overlap = sum((Counter(left) & Counter(right)).values())
    return 2 * overlap / (len(left) + len(right))


def _read_jsonl(path):
    try:
        with Path(path).open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]
    except (OSError, ValueError) as exc:
        raise MLXUserError(f"Unable to read RAG artifact {path}: {exc}") from exc


class EvaluateRagReduction:
    """Evaluate a fixed generator on each reducer's top-k evidence, with resume cache."""

    def __init__(self, dataset_root, benchmark_root, output_path, *, generator,
                 generator_model_path, seed=42, top_k=3, passage_char_limit=2400):
        self.dataset_root = Path(dataset_root).expanduser()
        self.benchmark_root = Path(benchmark_root).expanduser()
        self.output = Path(output_path).expanduser()
        self.generator = generator
        self.generator_model_path = Path(generator_model_path).expanduser()
        self.seed = seed
        self.top_k = top_k
        self.passage_char_limit = passage_char_limit

    def execute(self):
        if (type(self.seed) is not int or type(self.top_k) is not int or self.top_k < 1
                or type(self.passage_char_limit) is not int or self.passage_char_limit < 1):
            raise MLXUserError("RAG seed, retrieval depth, and passage length must be positive integers.")
        if not (self.benchmark_root / "comparison/stage.json").is_file():
            raise MLXUserError("RAG evaluation requires a completed benchmark comparison stage.")
        identity = self._identity()
        self.output.mkdir(parents=True, exist_ok=True)
        identity_path = self.output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise MLXUserError("RAG evaluation identity changed; use a new output directory.")
        else:
            if any(self.output.iterdir()):
                raise MLXUserError("RAG output directory is nonempty without an identity manifest.")
            write_json_atomic(identity_path, identity)
        cache = self._load_cache()
        rows, retrieval_rows = [], []
        try:
            for name in identity["datasets"]:
                generated, retrieved = self._dataset(name, cache)
                rows.extend(generated)
                retrieval_rows.extend(retrieved)
        finally:
            close = getattr(self.generator, "close", None)
            if callable(close):
                close()
        write_csv(self.output / "query_results.csv", rows)
        write_csv(self.output / "retrieval_results.csv", retrieval_rows)
        write_json_atomic(self.output / "summary.json", self._summary(rows, len(cache)))
        return {"rows": len(rows), "retrieval_rows": len(retrieval_rows),
                "unique_generations": len(cache),
                "summary": str(self.output / "summary.json")}

    def _identity(self):
        experiment = json.loads((self.benchmark_root / "experiment.json").read_text())
        datasets = experiment["experiment"]["datasets"]
        sources = {}
        for name in datasets:
            root = self.dataset_root / name
            sources[name] = {key: sha256_file(root / key) for key in (
                "corpus.jsonl", "queries.jsonl", "qrels/test.tsv", "answers.jsonl", "generation_queries.json")}
        return {"schema_version": 1, "datasets": datasets, "source_hashes": sources,
                "benchmark_comparison_sha256": sha256_file(self.benchmark_root / "comparison/cells.json"),
                "generator_sha256": sha256_file(self.generator_model_path), "seed": self.seed,
                "generator_config": getattr(self.generator, "provenance", None),
                "prompt_version": 1, "top_k": self.top_k,
                "passage_char_limit": self.passage_char_limit}

    def _load_cache(self):
        cache = {}
        path = self.output / "generation_cache.jsonl"
        if path.exists():
            for row in _read_jsonl(path):
                key, answer = row.get("prompt_sha256"), row.get("answer")
                if not isinstance(key, str) or not isinstance(answer, str):
                    raise MLXUserError(f"Invalid generation cache row in {path}.")
                if key in cache and cache[key] != answer:
                    raise MLXUserError(f"Conflicting cached answer in {path}.")
                cache[key] = answer
        return cache

    def _dataset(self, name, cache):
        root = self.dataset_root / name
        dataset = BeirDatasetLoader(allow_empty_documents=True).load(root)
        docs = {doc.id: doc for doc in dataset.corpus}
        queries = {query.id: query.text for query in dataset.queries}
        gold = {}
        for qrel in dataset.qrels:
            if qrel.relevance > 0:
                gold.setdefault(qrel.query_id, set()).add(qrel.document_id)
        answers = {row["query_id"]: row["answers"] for row in _read_jsonl(root / "answers.jsonl")}
        generation_ids = json.loads((root / "generation_queries.json").read_text())
        if len(generation_ids) != len(set(generation_ids)) or any(q not in queries for q in generation_ids):
            raise MLXUserError(f"Invalid frozen generation queries in {root}.")
        rankings = self._rankings(name, set(queries))
        rows, retrieval_rows = [], []
        for variant, by_query in rankings.items():
            for query_id in queries:
                hits = by_query[query_id][:self.top_k]
                hit_ids = [item["document_id"] for item in hits]
                if any(doc_id not in docs for doc_id in hit_ids):
                    raise MLXUserError(f"Unknown ranked document in {name}/{variant}/{query_id}.")
                evidence = gold[query_id]
                recall = len(evidence.intersection(hit_ids)) / len(evidence)
                retrieval_rows.append({"dataset": name, "variant": variant, "query_id": query_id,
                                       "evidence_recall": recall, "all_evidence": int(recall == 1),
                                       "retrieved_ids": json.dumps(hit_ids)})
                if query_id not in generation_ids:
                    continue
                prompt = self._prompt(queries[query_id], [docs[doc_id] for doc_id in hit_ids])
                key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if key not in cache:
                    answer = self.generator.generate(prompt)
                    with (self.output / "generation_cache.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps({"prompt_sha256": key, "answer": answer}, ensure_ascii=False) + "\n")
                    cache[key] = answer
                answer = cache[key]
                references = answers[query_id]
                exact = max(int(_normalize_answer(answer) == _normalize_answer(ref)) for ref in references)
                f1 = max(_token_f1(answer, ref) for ref in references)
                rows.append({"dataset": name, "variant": variant, "query_id": query_id,
                             "answer": answer, "references": json.dumps(references, ensure_ascii=False),
                             "exact_match": exact, "token_f1": f1, "evidence_recall": recall,
                             "all_evidence": int(recall == 1), "supported_exact": int(exact and recall == 1),
                             "retrieved_ids": json.dumps(hit_ids), "prompt_sha256": key})
        return rows, retrieval_rows

    def _rankings(self, name, query_ids):
        root = self.benchmark_root / name
        paths = {"full": root / "baseline/rankings.jsonl",
                 "orthogonal-mse": root / f"runs/orthogonal-mse-512/seed-{self.seed}/eval-512/benchmark/rankings.jsonl",
                 "vanilla-mse": root / f"runs/vanilla-mse-512/seed-{self.seed}/eval-512/benchmark/rankings.jsonl",
                 "pca": root / f"runs/pca-512/seed-{self.seed}/eval-512/benchmark/rankings.jsonl",
                 "svd": root / f"runs/svd-512/seed-{self.seed}/eval-512/benchmark/rankings.jsonl",
                 "truncate": root / "runs/truncate-512/eval-512/benchmark/rankings.jsonl"}
        result = {}
        for variant, path in paths.items():
            values = _read_jsonl(path)
            try:
                by_query = {row["query_id"]: row["results"] for row in values}
            except (TypeError, KeyError) as exc:
                raise MLXUserError(f"Invalid rankings for {name}/{variant}.") from exc
            if len(by_query) != len(values) or set(by_query) != query_ids:
                raise MLXUserError(f"Incomplete or duplicate rankings for {name}/{variant}.")
            result[variant] = by_query
        return result

    def _prompt(self, question, documents):
        source_text = "\n\n".join(
            f"[{index}] {doc.title}\n{doc.text[:self.passage_char_limit]}"
            for index, doc in enumerate(documents, start=1)
        )
        return f"Sources:\n{source_text}\n\nQuestion: {question}\nAnswer:"

    @staticmethod
    def _summary(rows, generations):
        grouped = {}
        for row in rows:
            grouped.setdefault((row["dataset"], row["variant"]), []).append(row)
        return {"unique_generations": generations, "dataset_variants": [
            {"dataset": dataset, "variant": variant, "questions": len(items),
             **{metric: sum(float(item[metric]) for item in items) / len(items)
                for metric in ("exact_match", "token_f1", "evidence_recall", "all_evidence", "supported_exact")}}
            for (dataset, variant), items in sorted(grouped.items())]}
