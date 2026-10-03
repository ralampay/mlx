"""Prepare fixed answer-and-evidence QA suites in the BEIR retrieval layout."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from mlx.core.artifacts import sha256_file, write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.modes.text_embedding.data import BeirDatasetLoader
from mlx.modes.text_embedding.retrieval_datasets import dataset_hashes


def _rank(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parquet(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except (ImportError, OSError, ValueError) as exc:
        raise MLXUserError(f"Unable to read RAG dataset {path}: {exc}") from exc


def _add_document(corpus, identifier, title, text):
    if not text.strip():
        return
    record = {"_id": identifier, "title": title, "text": text}
    if identifier in corpus and corpus[identifier] != record:
        raise MLXUserError(f"Conflicting RAG document ID: {identifier}")
    corpus[identifier] = record


class PrepareRagEvaluationDatasets:
    """Convert pinned QA sources to reusable evidence retrieval and answer labels."""

    def __init__(self, sources, output_root, *, questions_per_dataset=60,
                 generation_questions=30, distractor_examples=120):
        self.sources = Path(sources).expanduser()
        self.root = Path(output_root).expanduser()
        self.questions = questions_per_dataset
        self.generation_questions = generation_questions
        self.distractors = distractor_examples

    def execute(self):
        if (type(self.questions) is not int or self.questions < 2
                or type(self.generation_questions) is not int
                or not 1 <= self.generation_questions <= self.questions
                or type(self.distractors) is not int or self.distractors < 0):
            raise MLXUserError("RAG dataset sample counts must be positive and generation questions cannot exceed test questions.")
        plans = (
            ("rag-squad-v2", self.sources / "squad_v2/squad_v2/validation-00000-of-00001.parquet", self._squad),
            ("rag-hotpotqa", self.sources / "hotpotqa/distractor/validation-00000-of-00001.parquet", self._hotpot),
            ("rag-finqa", self.sources / "finqa/test.json", self._finqa),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        prepared = []
        for name, source, converter in plans:
            target = self.root / name
            if target.exists():
                self._verify(target, source)
            else:
                self._prepare(target, source, converter)
            prepared.append(str(target))
        return {"datasets": prepared}

    def _prepare(self, target, source, converter):
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=self.root))
        try:
            corpus, questions, evidence = converter(source)
            if len(questions) != self.questions:
                raise MLXUserError(f"{target.name} has only {len(questions)} eligible questions; need {self.questions}.")
            (temporary / "qrels").mkdir()
            with (temporary / "corpus.jsonl").open("w", encoding="utf-8") as output:
                for doc in sorted(corpus.values(), key=lambda item: item["_id"]):
                    output.write(json.dumps(doc, ensure_ascii=False) + "\n")
            with (temporary / "queries.jsonl").open("w", encoding="utf-8") as output, (
                temporary / "answers.jsonl"
            ).open("w", encoding="utf-8") as labels:
                for item in questions:
                    output.write(json.dumps({"_id": item["id"], "text": item["question"]}, ensure_ascii=False) + "\n")
                    labels.write(json.dumps({"query_id": item["id"], "answers": item["answers"]}, ensure_ascii=False) + "\n")
            with (temporary / "qrels/test.tsv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(("query-id", "corpus-id", "score"))
                for item in questions:
                    for doc_id in sorted(evidence[item["id"]]):
                        writer.writerow((item["id"], doc_id, 1))
            write_json_atomic(temporary / "generation_queries.json", [
                item["id"] for item in questions[:self.generation_questions]
            ])
            dataset = BeirDatasetLoader(allow_empty_documents=True).load(temporary)
            if len(dataset.corpus) != len(corpus) or len(dataset.queries) != len(questions):
                raise MLXUserError(f"Converted {target.name} failed corpus/query validation.")
            sources = [source]
            if target.name == "rag-finqa":
                sources.append(self.sources / "finqa/dev.json")
            write_json_atomic(temporary / "source_manifest.json", {
                "schema_version": 1, "selection": "SHA256(query ID) ascending; no answer-based model selection",
                "questions": self.questions, "generation_questions": self.generation_questions,
                "distractor_examples": self.distractors, "documents": len(corpus),
                "source_sha256": {str(path.relative_to(self.sources)): sha256_file(path) for path in sources},
                "hashes": {**dataset_hashes(temporary),
                           "answers.jsonl": sha256_file(temporary / "answers.jsonl"),
                           "generation_queries.json": sha256_file(temporary / "generation_queries.json")},
            })
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _verify(self, target, source):
        try:
            manifest = json.loads((target / "source_manifest.json").read_text())
            sources = [source]
            if target.name == "rag-finqa":
                sources.append(self.sources / "finqa/dev.json")
            expected_sources = {str(path.relative_to(self.sources)): sha256_file(path) for path in sources}
            expected_hashes = {**dataset_hashes(target),
                               "answers.jsonl": sha256_file(target / "answers.jsonl"),
                               "generation_queries.json": sha256_file(target / "generation_queries.json")}
            if (manifest["source_sha256"] != expected_sources or manifest["hashes"] != expected_hashes
                    or manifest["questions"] != self.questions
                    or manifest["generation_questions"] != self.generation_questions):
                raise ValueError("source or output hash mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MLXUserError(f"Existing RAG dataset differs from its frozen source or settings: {target}") from exc

    def _squad(self, source):
        rows = _parquet(source)
        eligible = sorted((row for row in rows if row["answers"]["text"]), key=lambda row: _rank(row["id"]))
        selected = eligible[:self.questions]
        corpus, questions, evidence = {}, [], {}
        for row in rows:
            identifier = "squad-" + _rank(row["context"])[:20]
            _add_document(corpus, identifier, row["title"], row["context"])
        for row in selected:
            identifier = "squad-" + _rank(row["context"])[:20]
            questions.append({"id": row["id"], "question": row["question"],
                              "answers": list(dict.fromkeys(row["answers"]["text"]))})
            evidence[row["id"]] = {identifier}
        return corpus, questions, evidence

    def _hotpot(self, source):
        rows = sorted(_parquet(source), key=lambda row: _rank(row["id"]))
        selected = rows[:self.questions]
        selected_ids = {row["id"] for row in selected}
        corpus, questions, evidence = {}, [], {}
        for row in selected + rows[self.questions:self.questions + self.distractors]:
            titles = row["context"]["title"]
            passages = row["context"]["sentences"]
            ids = {}
            for title, sentences in zip(titles, passages, strict=True):
                body = " ".join(sentences)
                identifier = "hotpot-" + _rank(title + "\n" + body)[:20]
                _add_document(corpus, identifier, title, body)
                ids[title] = identifier
            if row["id"] not in selected_ids:
                continue
            required = set(row["supporting_facts"]["title"])
            if not required <= ids.keys():
                raise MLXUserError(f"Missing HotpotQA supporting passage for {row['id']}.")
            questions.append({"id": row["id"], "question": row["question"], "answers": [row["answer"]]})
            evidence[row["id"]] = {ids[title] for title in required}
        return corpus, questions, evidence

    def _finqa(self, source):
        try:
            test = json.loads(source.read_text(encoding="utf-8"))
            dev = json.loads((self.sources / "finqa/dev.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MLXUserError(f"Unable to read FinQA sources: {exc}") from exc
        eligible = sorted((row for row in test if row["qa"].get("answer") not in (None, "")
                           and row["qa"].get("gold_inds")), key=lambda row: _rank(row["id"]))
        selected = eligible[:self.questions]
        selected_ids = {row["id"] for row in selected}
        distractors = sorted(dev, key=lambda row: _rank(row["id"]))[:self.distractors]
        corpus, questions, evidence = {}, [], {}
        for row in selected + distractors:
            prefix = "finqa-" + _rank(row["filename"])[:16]
            title = row["filename"]
            passages = row["pre_text"] + row["post_text"]
            for index, passage in enumerate(passages):
                _add_document(corpus, f"{prefix}-text_{index}", title, passage)
            headers = row["table"][0]
            for index, cells in enumerate(row["table"]):
                body = " | ".join(f"{header}: {value}" for header, value in zip(headers, cells, strict=True))
                _add_document(corpus, f"{prefix}-table_{index}", title, body)
            if row["id"] not in selected_ids:
                continue
            ids = {f"{prefix}-{key}" for key in row["qa"]["gold_inds"]}
            if not ids <= corpus.keys():
                raise MLXUserError(f"Missing FinQA evidence row for {row['id']}.")
            questions.append({"id": row["id"], "question": row["qa"]["question"],
                              "answers": [str(row["qa"]["answer"])]})
            evidence[row["id"]] = ids
        return corpus, questions, evidence
