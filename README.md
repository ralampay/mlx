# MLX (Machine Learning eXecutor)

A CLI that wraps several ML/DL workflows behind a consistent interface.

## Usage

All commands share a common signature. Pick the `--module` you want to run, select a `--platform`, and supply any module-specific arguments. Example:

```bash
mlx --module chat --platform openai --model gpt-4o-mini
```

The sections below detail every built-in module, their supported platforms, and the key parameters you can tweak.

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

The CLI loads `.env` automatically on startup. You can confirm the current values (masked where appropriate) with:

```bash
mlx --module system --action ls-env
```

Set any additional variables you rely on (for example `ROBOFLOW_API_KEY`) in the same `.env` file or through your preferred secrets manager.

### RAG Utilities

```bash
mlx --module rag \
    --action vectorization-summary \
    --chunk-size 800 \
    --chunk-overlap 100 \
    --dataset-path ./datasets/rag \
    --table-name demo_collection \
    --file-limit 50 \
    --local
```

- `vectorization-summary`: Scans the directory from `--dataset-path` for `.txt` and `.pdf` files, chunks their contents, and prints a summary table showing the model, embedding size, chunking hyperparameters, dataset statistics, expected row count, total token count, the destination table/collection, and the target database adapter. A sample JSON record (including id, the first five embedding coordinates, content, metadata, model name, platform, and source) is also shown to preview what will be inserted downstream. Tune the chunker with `--chunk-size` and `--chunk-overlap`, and cap ingestion with `--file-limit` if needed. Pass `--local` to load the GGUF model pointed to by `LOCAL_LLM_MODEL` via `llama-cpp-python`; omit it to fall back to the default Chroma embedding function. When `DB_ADAPTER` is `chromadb`, the summary also displays `DB_HOST` and `DB_PORT`.

```bash
mlx --module rag \
    --action vectorization-summary \
    --platform openai \
    --model text-embedding-3-large \
    --dataset-path ./datasets/rag \
    --table-name demo_collection
```

- Use the OpenAI embedding API by setting `--platform openai` (the `--model` flag selects the embedding model). All inserted records are tagged with the `openai` platform so they can be queried separately from local vectors.

```bash
mlx --module rag \
    --action batch-insert \
    --chunk-size 800 \
    --chunk-overlap 100 \
    --dataset-path ./datasets/rag \
    --table-name demo_collection \
    --file-limit 50 \
    --local
```

- `batch-insert`: Executes the same chunking workflow and inserts the resulting vectors into the collection specified by `--table-name`. Currently supports `DB_ADAPTER=chromadb`, connecting with `DB_HOST`, `DB_PORT`, `DB_USERNAME`, and `DB_PASSWORD`. Inserts occur per source file with a Rich progress spinner, respecting `--file-limit` to cap ingestion size and reusing the summary table for confirmation.

```bash
mlx --module rag \
    --action query \
    --platform openai \
    --model gpt-4o-mini \
    --table-name demo_collection \
    --top-k 5
```

- When `--platform openai` is supplied, embeddings and responses are generated with the OpenAI API, and retrieved vectors are filtered to those written by the OpenAI pipeline. For local answers, keep using the flags from the previous example.

```bash
mlx --module rag \
    --action query \
    --platform openai \
    --model gpt-4o-mini \
    --table-name demo_collection \
    --local
```

- Combine `--platform openai` with `--local` to retrieve locally generated embeddings (stored with `platform=local`) while still using the OpenAI model to craft the final response.

```bash
mlx --module rag \
    --action query \
    --table-name demo_collection \
    --top-k 5 \
    --local
```

- `query`: Prompts for a question, retrieves the top results from the specified collection, and generates a response with the selected backend. With `--platform openai`, embeddings and answers come from the OpenAI API (and only records tagged with `platform=openai` are considered). With `--local`, the GGUF model pointed to `LOCAL_LLM_GENERATION_MODEL` (or `LOCAL_LLM_MODEL`) is used to respond.
- For local answers, configure `LOCAL_LLM_GENERATION_MODEL` with a chat-capable GGUF (or rely on `LOCAL_LLM_MODEL` if it is generative).

```bash
mlx --module rag \
    --action delete-all \
    --table-name demo_collection
```

- `delete-all`: Removes every record from the specified collection. Currently supports `DB_ADAPTER=chromadb`.

All RAG commands require `--table-name` to identify the target collection.

## Modules

### Chat (OpenAI platform)

```bash
mlx --module chat \
    --platform openai \
    --model gpt-4o-mini \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 40
```

- `--module chat`: Selects the conversational agent.
- `--platform openai`: Routes to the OpenAI SDK integration.
- `--model`: OpenAI chat model name.
- `--temperature`: Controls response creativity.
- `--top-p`: Nucleus sampling probability cutoff.
- `--top-k`: Retained token count (ignored by OpenAI but accepted for parity).

### One-Shot Image Classification (Torch platform)

```bash
mlx --module ic-one-shot \
    --platform torch \
    --model siamese-le-net \
    --action train \
    --dataset-path ~/datasets/omniglot \
    --epochs 50 \
    --batch-size 8 \
    --device cuda:0
```

- `--module ic-one-shot`: Runs the Siamese network pipeline.
- `--platform torch`: Uses the PyTorch-backed implementation.
- `--model`: Model architecture (currently `siamese-le-net`).
- `--action`: Workflow to execute (`train`, `test`, `benchmark`, `infer-image`, `build-dataset`).
- `--dataset-path`: Root folder for the dataset (supports build/test modes).
- `--epochs`, `--batch-size`: Training loop controls.
- `--device`: Target device (`cpu`, `cuda:0`, etc.).
- Additional options (`--embedding-size`, `--input-img`, etc.) propagate to the trainer.

### Object Detection (Ultralytics platform)

```bash
mlx --module obj-detect \
    --platform ultralytics \
    --action train \
    --dataset-path ~/datasets/roboflow-yolo \
    --model ultralytics/cfg/models/ext/cad_yolo12.yaml \
    --epochs 100 \
    --batch-size 16 \
    --device cuda:0
```

- `--module obj-detect`: Activates the Ultralytics YOLO trainer.
- `--platform ultralytics`: Uses the `ralampay/ultralytics` fork.
- `--action`: Currently supports `train`.
- `--dataset-path`: Directory containing the YOLO-formatted dataset (`data.yaml`, images, labels).
- `--model`: Architecture YAML to instantiate the network (required when starting from scratch).
- `--model-path`: Optional weights (`.pt`) to warm start training; omit to train from random init.
- `--pretrained/--no-pretrained`: Whether Ultralytics should fetch pretrained weights for the provided YAML (default: disabled).
- `--lr0`: (Optional) Override the initial learning rate if the default is too high when training from scratch.
- `--optimizer`: Optimizer selection (`auto`, `adamw`, `sgd`, `adam`, `rmsprop`, etc.).
- `--nbs`: Nominal batch size used for Ultralytics’ LR scaling (default `64`).
- `--warmup-epochs`: Control the length of the LR warmup phase (default `3`).
- `--amp/--no-amp`: Toggle mixed-precision training.
- `--loss-clip`: Optional gradient clipping threshold.
- `--epochs`, `--batch-size`, `--device`: Standard training controls.
- `--height`, `--width`: Influence the YOLO `imgsz` (largest dimension wins).

For live camera inference with trained weights:

```bash
mlx --module obj-detect \
    --platform ultralytics \
    --action infer-camera \
    --model ultralytics/cfg/models/ext/cad_yolo12.yaml \
    --model-path ./runs/train/weights/best.pt \
    --device cpu \
    --confidence 0.35 \
    --camera-index 0
```

- `--model`: Path to the model YAML (architecture definition).
- `--model-path`: Trained weights (`.pt`) to load.
- `--confidence`: Detection confidence threshold.
- `--camera-index`: OpenCV camera index (`0` for default webcam).
- `--file-path`: Required for `infer-video`; supplies the video source.

For running inference on a video file:

```bash
mlx --module obj-detect \
    --platform ultralytics \
    --action infer-video \
    --model ultralytics/cfg/models/ext/cad_yolo12.yaml \
    --model-path ./runs/train/weights/best.pt \
    --file-path ~/videos/sample.mp4 \
    --device cpu \
    --confidence 0.35
```

For more options, run:

```bash
mlx --help
```

## Environment Variables

Some integrations expect API credentials to be present in your shell environment:

- `OPENAI_API_KEY`: Required for chat sessions via the OpenAI platform. Generate a key from the OpenAI dashboard and export it, e.g. `export OPENAI_API_KEY=sk-...`.
- `ROBOFLOW_API_KEY`: Needed when building or downloading datasets from Roboflow for the one-shot and object-detection workflows. Set it with `export ROBOFLOW_API_KEY=...`.

Ensure these are configured (or provided through your secrets manager) before invoking the corresponding modules.
