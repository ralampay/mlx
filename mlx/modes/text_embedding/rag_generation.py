"""Lazy local GGUF generator behind the RAG answer interface."""

from __future__ import annotations

from pathlib import Path

from mlx.core.exceptions import MLXUserError


class LlamaCppRagGenerator:
    SYSTEM_MESSAGE = (
        "Answer using only the provided sources. If they do not contain the answer, "
        "say I don't know. Give a short answer without explanation."
    )

    def __init__(self, model_path, *, context_length=4096, max_tokens=64, seed=42,
                 temperature=0):
        self.model_path = Path(model_path).expanduser()
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.seed = seed
        self.temperature = temperature
        self._model = None

    @property
    def provenance(self):
        return {"context_length": self.context_length, "max_tokens": self.max_tokens,
                "seed": self.seed, "temperature": self.temperature,
                "system_message": self.SYSTEM_MESSAGE}

    def generate(self, prompt):
        if not self.model_path.is_file():
            raise MLXUserError(f"RAG generator model not found: {self.model_path}")
        if self._model is None:
            try:
                from llama_cpp import Llama
                self._model = Llama(model_path=str(self.model_path), n_ctx=self.context_length,
                                    n_gpu_layers=-1, verbose=False, seed=self.seed)
            except Exception as exc:
                raise MLXUserError(f"Unable to load RAG generator {self.model_path}: {exc}") from exc
        try:
            result = self._model.create_chat_completion(
                messages=[{"role": "system", "content": self.SYSTEM_MESSAGE},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            answer = result["choices"][0]["message"]["content"]
        except Exception as exc:
            raise MLXUserError(f"RAG answer generation failed for {self.model_path}: {exc}") from exc
        if not isinstance(answer, str):
            raise MLXUserError("RAG generator returned no text answer.")
        return answer.strip()

    def close(self):
        if self._model is not None:
            self._model.close()
            self._model = None
