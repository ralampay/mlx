from __future__ import annotations

from pathlib import Path

import pandas as pandas
import pytest

from mlx.cli import _build_config, build_parser
from mlx.core.exceptions import MLXUserError
from mlx.modes.nlp import embedding
from mlx.modes.nlp.embedding import EmbedCsv


class FakeLlama:
    instances = []

    def __init__(self, *, model_path: str, embedding: bool) -> None:
        self.model_path = model_path
        self.embedding = embedding
        self.seen = []
        self.__class__.instances.append(self)

    def embed(self, content: str) -> list[float]:
        self.seen.append(content)
        return [float(len(content)), 0.5]


@pytest.fixture(autouse=True)
def fake_llama(monkeypatch):
    FakeLlama.instances.clear()
    monkeypatch.setattr(embedding, "Llama", FakeLlama)


def _write_inputs(tmp_path: Path, csv_text: str = "content\nhello\nworld\n") -> tuple[Path, Path]:
    model_path = tmp_path / "model.gguf"
    input_path = tmp_path / "input.csv"
    model_path.write_bytes(b"fake model")
    input_path.write_text(csv_text, encoding="utf-8")
    return model_path, input_path


def test_cli_parses_nlp_embedding_options():
    namespace = build_parser().parse_args(
        [
            "--mode",
            "nlp",
            "--action",
            "embed",
            "--model-file",
            "model.gguf",
            "--input-file",
            "input.csv",
            "--output-file",
            "output.csv",
            "--column_name",
            "body",
        ]
    )

    config = _build_config(namespace)

    assert config["model_file"] == "model.gguf"
    assert config["input_file"] == "input.csv"
    assert config["output_file"] == "output.csv"
    assert config["column_name"] == "body"


def test_embed_csv_writes_default_output(tmp_path):
    model_path, input_path = _write_inputs(tmp_path)

    output_path = EmbedCsv(str(model_path), str(input_path)).execute()

    assert output_path == tmp_path / "input-model-output.csv"
    assert FakeLlama.instances[0].embedding is True
    assert FakeLlama.instances[0].seen == ["hello", "world"]
    output = pandas.read_csv(output_path)
    assert output.columns.tolist() == ["content", "embeddings"]
    assert output.to_dict("records") == [
        {"content": "hello", "embeddings": "[5.0,0.5]"},
        {"content": "world", "embeddings": "[5.0,0.5]"},
    ]


def test_custom_input_column_is_exported_as_content(tmp_path):
    model_path, input_path = _write_inputs(tmp_path, "body\nhello\n")

    output_path = EmbedCsv(
        str(model_path), str(input_path), column_name="body"
    ).execute()

    output = pandas.read_csv(output_path)
    assert output.columns.tolist() == ["content", "embeddings"]
    assert output.to_dict("records") == [
        {"content": "hello", "embeddings": "[5.0,0.5]"}
    ]


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("body\nhello\n", "does not contain column 'content'"),
        ("content\n", "contains no data rows"),
        ('content\nhello\n""\n', "empty, null, or non-string"),
        ("content\n123\n", "empty, null, or non-string"),
    ],
)
def test_invalid_csv_content_fails_before_model_load(tmp_path, csv_text, message):
    model_path, input_path = _write_inputs(tmp_path, csv_text)

    with pytest.raises(MLXUserError, match=message):
        EmbedCsv(str(model_path), str(input_path)).execute()

    assert FakeLlama.instances == []


def test_existing_output_is_not_replaced(tmp_path):
    model_path, input_path = _write_inputs(tmp_path)
    output_path = tmp_path / "result.csv"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(MLXUserError, match="already exists"):
        EmbedCsv(str(model_path), str(input_path), str(output_path)).execute()

    assert output_path.read_text(encoding="utf-8") == "existing"
    assert FakeLlama.instances == []


def test_token_level_embeddings_are_rejected(tmp_path, monkeypatch):
    model_path, input_path = _write_inputs(tmp_path)
    monkeypatch.setattr(FakeLlama, "embed", lambda self, content: [[1.0], [2.0]])

    with pytest.raises(MLXUserError, match="token-level embeddings"):
        EmbedCsv(str(model_path), str(input_path)).execute()

    assert not (tmp_path / "input-model-output.csv").exists()


def test_inconsistent_embedding_dimensions_are_rejected(tmp_path, monkeypatch):
    model_path, input_path = _write_inputs(tmp_path)
    results = iter(([1.0, 2.0], [3.0]))
    monkeypatch.setattr(FakeLlama, "embed", lambda self, content: next(results))

    with pytest.raises(MLXUserError, match="dimension changed"):
        EmbedCsv(str(model_path), str(input_path)).execute()

    assert not (tmp_path / "input-model-output.csv").exists()
