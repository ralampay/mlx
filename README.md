# MLX (Machine Learning eXecutor)

A CLI that wraps chat, retrieval, and computer-vision workflows behind a consistent interface.

The terminal experience uses the Python `rich` text UI library for interactive prompts, status panels, tables, and runtime summaries.

## Usage

All commands share a common signature. Pick the `--module` you want to run, select a `--platform`, and supply any module-specific arguments. Example:

```bash
mlx --module chat --platform openai --model gpt-4o-mini
```

or when running from source:

```bash
python -m mlx --module chat --platform openai --model gpt-4o-mini
```

The sections below detail the built-in modules, their supported platforms, and the key parameters you can tweak.

## Modules

* [Chat](./docs/chat/index.md)
* [Object Detection](./docs/object_detection/index.md)
* [RAG Utilities](./docs/rag/index.md)
* [One-Shot Image Classification (Torch platform)](./docs/image_classification/index.md)

Run `mlx --help` to explore every available module action and configuration flag.

## Environment Setup

Copy the provided template and populate the values required for your workspace:

```bash
cp .env.dist .env
```

- `LOCAL_LLM_MODEL`: Filesystem path to the local model weights that offline modules will use.
- `LOCAL_LLM_GENERATION_MODEL`: Optional path to a text-generative GGUF used for RAG query responses (falls back to `LOCAL_LLM_MODEL` when unset).
- `OPENAI_API_KEY`: API key used by OpenAI-powered modules.
- `HUGGINGFACE_TOKEN`: Access token for downloading models or datasets from Hugging Face.
- `DB_ADAPTER`: Target vector database adapter (`chromadb` by default; alternatively `postgres` if you wire up a Postgres-backed store).
- `DB_HOST`, `DB_PORT`: Hostname and port for the ChromaDB server when `DB_ADAPTER=chromadb`.
- `DB_USERNAME`, `DB_PASSWORD`: Credentials for authenticated ChromaDB deployments (password is masked in the CLI).
- `ROBOFLOW_API_KEY`: Needed when building or downloading datasets from Roboflow for the one-shot and object-detection workflows.

The CLI loads `.env` automatically on startup. You can confirm the current values (masked where appropriate) with:

```bash
mlx --module system --action ls-env
```

Set any additional variables you rely on (for example `ROBOFLOW_API_KEY`) in the same `.env` file or through your preferred secrets manager.

## Packages

Install the dependencies listed below (package names target PyPI):

- beautifulsoup4 (`bs4`)
- chromadb
- huggingface-hub
- langchain
- llama-cpp-python (`llama_cpp`)
- matplotlib
- numpy
- opencv-python (`cv2`)
- openai
- Pillow (`PIL`)
- pypdf
- python-dotenv (`dotenv`)
- requests
- rich
- scikit-learn (`sklearn`)
- torch
- torchvision
- tqdm
- typer
- ultralytics (`https://github.com/ralampay/ultralytics`)
