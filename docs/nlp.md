# Legacy NLP CSV embedding

Package: `mlx.modes.nlp`. CLI alias: `nlp`, routed through `text_embedding`.

[All module guides](README.md) · [Text embedding and retrieval](text-embedding.md)

## Contents

- [When to use this module](#when-to-use-this-module)
- [Install and run](#install-and-run)
- [Input and output contract](#input-and-output-contract)
- [Python API](#python-api)
- [Moving to retrieval workflows](#moving-to-retrieval-workflows)

## When to use this module

This compatibility module embeds one text column from a CSV using a local GGUF embedding
model. It preserves the historical `EmbedCsvCommand` and `EmbedCsv` Python entrypoints.
It does not build a retrieval index or evaluate relevance judgments.

The CLI chooses this path when `--input-file` is supplied and `--input` is absent.
Using `--mode nlp` with `--input` selects the modern BEIR workflow instead. The alias alone
does not select legacy CSV behavior.

## Install and run

From an installed checkout, add pandas and llama.cpp support:

```bash
python -m pip install -e ".[nlp]"
python -m mlx --mode nlp --action embed \
  --model-file ./models/embedding-model.gguf \
  --input-file ./documents.csv --column-name content \
  --output-file ./documents-embedded.csv
```

The model must produce one fixed-width, finite sequence vector per input string. Token-level
embedding arrays are rejected. Missing files, invalid text, and changing vector widths produce
`MLXUserError` with row or path context. This mode does not require a GPU.

## Input and output contract

Input is a CSV with a header and at least one data row. Every value in the selected column
must be a non-empty string; the default column name is `content`.

```csv
id,content
1,A short example document.
2,A second document.
```

Output has exactly two columns: `content` and `embeddings`. Each `embeddings` cell is a JSON
array. Input row order and selected text values are retained, but other columns such as `id`
are not copied. The output text column is named `content` even when a different input column
is selected.

The destination parent directory must exist, and an existing output file is rejected. If
`--output-file` is omitted, the file is written beside the input with the name
`<input-stem>-<model-stem>-output.csv`. A successful command returns its output path.

Non-default `--pooling` and `--prompt-format` options are rejected by the legacy CLI path;
use the BEIR workflow for those controls.

## Python API

Construct a request and command without loading the model:

```python
from mlx.modes.nlp import EmbedCsvCommand, EmbedCsvRequest

request = EmbedCsvRequest(
    model_file="./models/embedding-model.gguf",
    input_file="./documents.csv",
    output_file="./documents-embedded.csv",
)
command = EmbedCsvCommand(request)
```

Call `command.execute()` after supplying valid files. It returns `EmbedCsvResult` with an
`output_path`. Tests can inject `model_factory`, a callable receiving the model path and
returning an object with `embed(content)`. Path validation still applies with an injected model.
The historical `EmbedCsv(...).execute()` API returns the path directly.

## Moving to retrieval workflows

Use [text embedding](text-embedding.md) for separate corpora, queries, relevance judgments,
vector-store artifacts, and retrieval metrics. Its files have their own IDs and manifests.
Legacy CSV output is not a complete BEIR embedding artifact. It also uses the plural
`embeddings` column; [autoencoder input](autoencoder.md) expects a singular `embedding`
column, so conversion must be explicit.

See [the architecture](../ARCHITECTURE.md#text-embedding-and-retrieval) for compatibility ownership.
