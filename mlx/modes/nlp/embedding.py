from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import Any, Optional

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from mlx.core.exceptions import MLXUserError
from mlx.core.ui import console

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in installations missing extras
    pd = None

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - exercised in installations missing extras
    Llama = None


class EmbedCsv:
    def __init__(
        self,
        model_file: Optional[str],
        input_file: Optional[str],
        output_file: Optional[str] = None,
        column_name: str = "content",
    ) -> None:
        self.model_file = model_file
        self.input_file = input_file
        self.output_file = output_file
        self.column_name = column_name

    def execute(self) -> Path:
        model_path, input_path, output_path = self._validate_paths()
        dataframe = self._read_input(input_path)
        contents = self._validate_contents(dataframe)

        console.print(f"[cyan]Rows to embed:[/cyan] {len(contents)}")
        model = self._load_model(model_path)
        embeddings = self._create_embeddings(model, contents)
        self._write_output(output_path, contents, embeddings)
        self._print_result(output_path, len(contents), len(embeddings[0]))
        return output_path

    def _validate_paths(self) -> tuple[Path, Path, Path]:
        if not self.model_file:
            raise MLXUserError("NLP embed requires --model-file pointing to a GGUF embedding model.")
        if not self.input_file:
            raise MLXUserError("NLP embed requires --input-file pointing to a CSV file.")

        model_path = Path(self.model_file).expanduser()
        input_path = Path(self.input_file).expanduser()
        if not model_path.is_file():
            raise MLXUserError(f"Embedding model file not found: {model_path}")
        if not input_path.is_file():
            raise MLXUserError(f"Input CSV file not found: {input_path}")

        output_path = self._resolve_output_path(input_path, model_path)
        if output_path.exists():
            raise MLXUserError(
                f"Output file already exists: {output_path}. Choose a different --output-file or remove it first."
            )
        if not output_path.parent.is_dir():
            raise MLXUserError(f"Output directory does not exist: {output_path.parent}")
        return model_path, input_path, output_path

    def _resolve_output_path(self, input_path: Path, model_path: Path) -> Path:
        if self.output_file:
            return Path(self.output_file).expanduser()
        filename = f"{input_path.stem}-{model_path.stem}-output.csv"
        return input_path.parent / filename

    def _read_input(self, input_path: Path):
        if pd is None:
            raise MLXUserError("NLP embed requires pandas. Install the project dependencies and try again.")
        try:
            return pd.read_csv(input_path)
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
            raise MLXUserError(
                f"Unable to read input CSV '{input_path}': {exc}. Check that it is a valid, readable CSV file."
            ) from exc

    def _validate_contents(self, dataframe) -> list[str]:
        if self.column_name not in dataframe.columns:
            available = ", ".join(map(str, dataframe.columns)) or "none"
            raise MLXUserError(
                f"Input CSV does not contain column '{self.column_name}'. Available columns: {available}."
            )
        if dataframe.empty:
            raise MLXUserError("Input CSV contains no data rows to embed.")

        contents: list[str] = []
        invalid_rows: list[int] = []
        for position, value in enumerate(dataframe[self.column_name].tolist(), start=2):
            if not isinstance(value, str) or not value.strip():
                invalid_rows.append(position)
            else:
                contents.append(value)
        if invalid_rows:
            rendered = ", ".join(map(str, invalid_rows[:10]))
            suffix = "..." if len(invalid_rows) > 10 else ""
            raise MLXUserError(
                f"Column '{self.column_name}' has empty, null, or non-string values at CSV rows {rendered}{suffix}. "
                "Provide non-empty text for every row."
            )
        return contents

    def _load_model(self, model_path: Path):
        if Llama is None:
            raise MLXUserError(
                "NLP embed requires llama-cpp-python. Install the project dependencies and try again."
            )
        try:
            return Llama(model_path=str(model_path), embedding=True)
        except Exception as exc:
            raise MLXUserError(
                f"Unable to load embedding model '{model_path}': {exc}. Check that it is a compatible GGUF embedding model."
            ) from exc

    def _create_embeddings(self, model: Any, contents: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        expected_size: Optional[int] = None
        progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        with progress:
            task_id = progress.add_task("Embedding", total=len(contents))
            for csv_row, content in enumerate(contents, start=2):
                try:
                    result = model.embed(content)
                except Exception as exc:
                    raise MLXUserError(
                        f"Embedding failed at CSV row {csv_row}: {exc}. Check the text length and model compatibility."
                    ) from exc
                vector = self._validate_embedding(result, csv_row)
                if expected_size is None:
                    expected_size = len(vector)
                elif len(vector) != expected_size:
                    raise MLXUserError(
                        f"Embedding dimension changed at CSV row {csv_row}: expected {expected_size}, got {len(vector)}."
                    )
                embeddings.append(vector)
                progress.advance(task_id)
        return embeddings

    @staticmethod
    def _validate_embedding(result: Any, csv_row: int) -> list[float]:
        if not isinstance(result, (list, tuple)) or not result:
            raise MLXUserError(f"Model returned no embedding vector at CSV row {csv_row}.")
        if any(isinstance(value, (list, tuple)) for value in result):
            raise MLXUserError(
                f"Model returned token-level embeddings at CSV row {csv_row}; use a GGUF model with sequence pooling."
            )
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in result):
            raise MLXUserError(f"Model returned a non-numeric embedding at CSV row {csv_row}.")
        return [float(value) for value in result]

    def _write_output(
        self,
        output_path: Path,
        contents: list[str],
        embeddings: list[list[float]],
    ) -> None:
        output = pd.DataFrame(
            {
                "content": contents,
                "embeddings": [
                    json.dumps(vector, separators=(",", ":")) for vector in embeddings
                ],
            }
        )
        try:
            output.to_csv(output_path, index=False)
        except (OSError, ValueError) as exc:
            raise MLXUserError(
                f"Unable to write output CSV '{output_path}': {exc}. Check the destination permissions."
            ) from exc

    @staticmethod
    def _print_result(output_path: Path, row_count: int, dimensions: int) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Embedding export complete[/bold green]\n"
                f"[cyan]File:[/cyan] {output_path.resolve()}\n"
                f"[cyan]Rows:[/cyan] {row_count}\n"
                f"[cyan]Dimensions:[/cyan] {dimensions}",
                border_style="green",
            )
        )
