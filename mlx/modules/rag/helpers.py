from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import typer
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for older langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.table import Table
from mlx.ui import console, print_warning

_LLAMACPP_SILENCED = False
_LLAMACPP_CALLBACK: Optional[Any] = None


def _ensure_llama_silenced() -> None:
    global _LLAMACPP_SILENCED, _LLAMACPP_CALLBACK
    if _LLAMACPP_SILENCED:
        return
    try:
        from llama_cpp import llama_cpp  # type: ignore
    except ImportError:
        return

    @llama_cpp.llama_log_callback  # type: ignore[attr-defined]
    def _silent_logger(_: int, __: bytes, ___: Any) -> None:
        return

    _LLAMACPP_CALLBACK = _silent_logger
    llama_cpp.llama_log_set(_silent_logger, None)
    _LLAMACPP_SILENCED = True


class LocalLlamaEmbedder:
    """Thin wrapper around llama-cpp-python for local GGUF embedding models."""

    def __init__(self, model_path: str, n_ctx: int = 4096) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise typer.BadParameter(
                "llama-cpp-python is required for --local embedding. Install llama-cpp-python to proceed."
            ) from exc

        _ensure_llama_silenced()

        self._model_path = model_path
        self._llama = Llama(model_path=model_path, embedding=True, n_ctx=n_ctx, verbose=False)
        self._cached_dimension: Optional[int] = None

    @property
    def model_path(self) -> str:
        return self._model_path

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], int, int]:
        vectors: List[List[float]] = []
        total_tokens = 0
        for text in texts:
            tokens = self._llama.tokenize(text.encode("utf-8"), add_bos=False)
            total_tokens += len(tokens)
            result = self._llama.create_embedding(input=[text])
            vector = result["data"][0]["embedding"]
            vectors.append(vector)
            if self._cached_dimension is None:
                self._cached_dimension = len(vector)
        embedding_dim = self._cached_dimension or (len(vectors[0]) if vectors else 0)
        return vectors, total_tokens, embedding_dim

    @property
    def embedding_size(self) -> Optional[int]:
        return self._cached_dimension


def _default_embed(texts: List[str]) -> Tuple[List[List[float]], int, int]:
    try:
        from chromadb.utils import embedding_functions  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise typer.BadParameter(
            "chromadb is required when --local is disabled. Install chromadb to proceed."
        ) from exc

    default_embedder = embedding_functions.DefaultEmbeddingFunction()
    vectors_raw = default_embedder(texts)
    vectors: List[List[float]] = []
    for vector in vectors_raw:
        if hasattr(vector, "tolist"):
            vector_list = vector.tolist()
        else:
            vector_list = vector
        vectors.append([float(value) for value in vector_list])
    embedding_dim = len(vectors[0]) if vectors else 0
    tokens = _estimate_tokens_from_texts(texts)
    return vectors, tokens, embedding_dim


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise typer.BadParameter(
            "pypdf is required to read PDF files. Install pypdf to proceed."
        ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _collect_dataset_chunks(
    dataset_path: str,
    splitter: RecursiveCharacterTextSplitter,
    max_files: Optional[int] = None,
    on_file: Optional[Callable[[Path], None]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    root = Path(dataset_path)
    if not root.exists():
        raise typer.BadParameter(f"Dataset path not found: {dataset_path}")
    if not root.is_dir():
        raise typer.BadParameter(f"Dataset path must be a directory: {dataset_path}")

    supported_ext = {".txt", ".pdf"}
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_ext
    )
    if not files:
        raise typer.BadParameter(
            f"No supported files found under {dataset_path}. Include .txt or .pdf files."
        )

    chunks: List[Dict[str, Any]] = []
    processed_files = 0
    global_index = 0
    for file_path in files:
        if max_files is not None and processed_files >= max_files:
            break
        if on_file is not None:
            on_file(file_path)
        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            else:
                text = _extract_pdf_text(file_path)
        except Exception as exc:
            print_warning(f"Failed to read {file_path.name}: {exc}. Skipping.")
            continue

        if not text.strip():
            print_warning(f"{file_path.name} is empty after parsing. Skipping.")
            continue

        processed_files += 1
        parts = splitter.split_text(text)
        for local_idx, part in enumerate(parts):
            chunk_id = f"{file_path.stem}-{local_idx}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": part,
                    "source": str(file_path.relative_to(root)),
                    "global_index": global_index,
                }
            )
            global_index += 1

    if not chunks:
        raise typer.BadParameter(
            "No content was extracted from the dataset. Check your files or adjust chunk parameters."
        )

    return chunks, processed_files


def _estimate_tokens_from_texts(texts: List[str]) -> int:
    total = 0
    for text in texts:
        # Rough heuristic: assume ~4 characters per token.
        total += max(1, len(text) // 4)
    return total


def _get_chroma_collection(
    table_name: str,
    db_config: Dict[str, str],
) -> Tuple[Any, str, int]:
    if db_config["adapter"].lower() != "chromadb":
        raise typer.BadParameter("This action currently supports only DB_ADAPTER=chromadb.")

    db_host = db_config.get("host")
    db_port_raw = db_config.get("port")
    if db_host in {"not set", "", None} or db_port_raw in {"not set", "", None}:
        raise typer.BadParameter("DB_HOST and DB_PORT must be set for chromadb operations.")

    try:
        db_port = int(db_port_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise typer.BadParameter("DB_PORT must be an integer for chromadb operations.")

    try:
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise typer.BadParameter("chromadb is required. Install chromadb to proceed.") from exc

    settings_kwargs: Dict[str, Any] = {}
    if db_config.get("username") or db_config.get("password"):
        settings_kwargs["chroma_client_auth_provider"] = "chromadb.auth.basic"
        credentials = f"{db_config.get('username', '')}:{db_config.get('password', '')}"
        settings_kwargs["chroma_client_auth_credentials"] = credentials

    settings = Settings(**settings_kwargs) if settings_kwargs else None
    client = chromadb.HttpClient(host=db_host, port=db_port, settings=settings)
    collection = client.get_or_create_collection(table_name)
    return collection, db_host or "unknown", db_port


def _infer_collection_platform(collection: Any) -> Optional[str]:
    try:
        sample = collection.get(limit=1, include=["metadatas"])
    except Exception:  # pragma: no cover - defensive
        return None

    metadatas = sample.get("metadatas") or []
    # metadatas structure is typically List[List[Dict]]
    for entry in metadatas:
        if isinstance(entry, list):
            for item in entry:
                if isinstance(item, dict) and item.get("platform"):
                    return str(item["platform"])
        elif isinstance(entry, dict) and entry.get("platform"):
            return str(entry["platform"])
    return None


def _determine_openai_embedding_model(model_name: Optional[str]) -> str:
    if not model_name or "gpt" in model_name.lower():
        return "text-embedding-3-large"
    return model_name


def _determine_openai_chat_model(model_name: Optional[str]) -> str:
    if not model_name or "embedding" in model_name.lower():
        return "gpt-4o-mini"
    return model_name


def _create_embedding_runner(
    use_local: bool,
    configured_model_name: Optional[str],
    platform: Optional[str],
) -> Tuple[
    Callable[[List[str]], Tuple[List[List[float]], int, int]],
    str,
    str,
]:
    if use_local:
        local_model_path = os.environ.get("LOCAL_LLM_MODEL")
        if not local_model_path:
            raise typer.BadParameter(
                "LOCAL_LLM_MODEL is not set. Export the variable or populate your .env file."
            )
        embedder = LocalLlamaEmbedder(local_model_path)

        def embed_fn(texts: List[str]) -> Tuple[List[List[float]], int, int]:
            return embedder.embed(texts)

        model_name = Path(local_model_path).name
        return embed_fn, model_name, "local"

    if platform == "huggingface":
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise typer.BadParameter(
                "huggingface-hub is required for --platform huggingface. Install huggingface-hub to proceed."
            ) from exc

        if not configured_model_name:
            raise typer.BadParameter(
                "Specify --model with the Hugging Face repository id when using --platform huggingface."
            )

        token = os.environ.get("HUGGINGFACE_TOKEN")
        client = InferenceClient(model=configured_model_name, token=token)

        def embed_fn(texts: List[str]) -> Tuple[List[List[float]], int, int]:
            vectors: List[List[float]] = []
            total_tokens = 0
            embedding_dim = 0
            for text in texts:
                result = client.feature_extraction(text)
                if result is None:
                    continue
                if hasattr(result, "tolist"):
                    result = result.tolist()
                if isinstance(result[0], list):
                    length = len(result[0])
                    summed = [0.0] * length
                    for token_vec in result:  # type: ignore[iteration-over-annotation]
                        for idx, value in enumerate(token_vec):
                            summed[idx] += float(value)
                    arr = [value / len(result) for value in summed]
                else:
                    arr = [float(value) for value in result]  # type: ignore[arg-type]
                vectors.append(arr)
                embedding_dim = len(arr)
                total_tokens += max(1, len(text) // 4)
            return vectors, total_tokens, embedding_dim

        return embed_fn, configured_model_name, "huggingface"

    if platform == "openai":
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise typer.BadParameter(
                "openai package is required for OpenAI embeddings. Install openai to proceed."
            ) from exc

        client = OpenAI()
        embedding_model = _determine_openai_embedding_model(configured_model_name)

        def embed_fn(texts: List[str]) -> Tuple[List[List[float]], int, int]:
            vectors: List[List[float]] = []
            total_tokens = 0
            embedding_dim = 0
            for text in texts:
                response = client.embeddings.create(
                    model=embedding_model,
                    input=text,
                )
                vector = response.data[0].embedding
                vectors.append(vector)
                embedding_dim = len(vector)
                usage = getattr(response, "usage", None)
                if usage and getattr(usage, "total_tokens", None):
                    total_tokens += usage.total_tokens
            return vectors, total_tokens, embedding_dim

        return embed_fn, embedding_model, "openai"

    def embed_fn(texts: List[str]) -> Tuple[List[List[float]], int, int]:
        return _default_embed(texts)

    model_name = configured_model_name or "chromadb-default-embedder"
    return embed_fn, model_name, "local"


def _render_run_metadata(
    model_name: str,
    embedding_size: int,
    chunk_size: int,
    chunk_overlap: int,
    dataset_path: str,
    file_count: int,
    row_count: int,
    total_tokens: int,
    db_adapter: str,
    db_host: str,
    db_port: str,
    db_status: str,
    db_username: str,
    db_password_status: str,
    table_name: str,
    embedding_platform: str,
) -> None:
    table = Table(title="Vectorization Summary")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Model", model_name)
    table.add_row("Embedding Size", str(embedding_size))
    table.add_row("Chunk Size", str(chunk_size))
    table.add_row("Chunk Overlap", str(chunk_overlap))
    table.add_row("Dataset Path", dataset_path)
    table.add_row("Files Processed", str(file_count))
    table.add_row("Rows (Chunks)", str(row_count))
    table.add_row("Total Tokens", str(total_tokens))
    table.add_row("DB Adapter", db_adapter)
    table.add_row("DB Host", db_host)
    table.add_row("DB Port", db_port)
    table.add_row("DB Status", db_status)
    table.add_row("DB Username", db_username)
    table.add_row("DB Password", db_password_status)
    table.add_row("Embedding Platform", embedding_platform)
    table.add_row("Table / Collection", table_name)
    console.print(table)


def _generate_local_answer(question: str, context: str, model_path: str) -> str:
    _ensure_llama_silenced()
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise typer.BadParameter(
            "llama-cpp-python is required for --local query responses. Install llama-cpp-python to proceed."
        ) from exc

    context_window = 4096
    llm = Llama(
        model_path=model_path,
        n_ctx=context_window,
        n_batch=256,
        embedding=False,
        verbose=False,
        cache_capacity=context_window * 2,
    )
    prompt = (
        "You answer questions using the provided context. Cite only what is supplied; "
        "if the answer cannot be found, reply that you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        result = llm(
            prompt=prompt,
            max_tokens=512,
            temperature=0.2,
            top_p=0.95,
        )
    except Exception as exc:  # pragma: no cover - protective guard
        return (
            "Local generation failed. Ensure LOCAL_LLM_MODEL points to a text-generative GGUF model. "
            f"Details: {exc}"
        )

    return result.get("choices", [{}])[0].get("text", "").strip() or "Unable to generate a response."


def _generate_openai_answer(question: str, context: str, model_name: Optional[str]) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise typer.BadParameter(
            "openai package is required for OpenAI responses. Install openai to proceed."
        ) from exc

    client = OpenAI()
    chat_model = _determine_openai_chat_model(model_name)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer the user's question using only the provided context. "
                "If the answer is not present, respond that you do not know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.2,
    )
    choices = getattr(response, "choices", [])
    if not choices:
        return "Unable to generate a response."
    message = choices[0].message
    return (message.content or "").strip() or "Unable to generate a response."


def _generate_hf_answer(question: str, context: str, model_name: str) -> str:
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise typer.BadParameter(
            "huggingface-hub is required for Hugging Face responses. Install huggingface-hub to proceed."
        ) from exc

    token = os.environ.get("HUGGINGFACE_TOKEN")
    client = InferenceClient(model=model_name, token=token)
    prompt = (
        "You answer questions using the provided context. Cite only what is supplied; "
        "if the answer cannot be found, reply that you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    primary_error: Optional[Exception] = None
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=400,
            temperature=0.2,
            top_p=0.95,
        )
        if isinstance(response, str):
            text = response
        elif isinstance(response, dict):
            text = response.get("generated_text") or ""
        elif isinstance(response, list) and response and isinstance(response[0], dict):
            text = response[0].get("generated_text") or ""
        else:
            text = str(response)
        text = text.strip()
        if text:
            return text
    except Exception as exc:  # pragma: no cover - defensive
        primary_error = exc

    try:
        chat_response = client.chat_completion(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using the provided context. "
                        "If the answer cannot be inferred, say you do not know."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        choices = getattr(chat_response, "choices", [])
        if choices:
            message = choices[0].message
            content = (message.get("content") if isinstance(message, dict) else getattr(message, "content", "")) or ""
            content = content.strip()
            if content:
                return content
    except Exception as exc:  # pragma: no cover - defensive
        secondary_error = exc
        detail_msg = f"{primary_error or secondary_error}"
        return (
            "Hugging Face generation failed. Ensure the model supports text-generation/chat and your "
            f"token (if required) is valid. Details: {detail_msg}"
        )

    return "Unable to generate a response."
