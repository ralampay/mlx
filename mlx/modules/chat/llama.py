from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ModuleConfig = Dict[str, Any]

console = Console()
_LLAMACPP_SILENCED = False
_LLAMACPP_LOGGER: Optional[Callable[..., None]] = None


def _ensure_llama_silenced() -> None:
    global _LLAMACPP_SILENCED, _LLAMACPP_LOGGER
    if _LLAMACPP_SILENCED:
        return
    try:
        from llama_cpp import llama_cpp  # type: ignore
    except ImportError:
        return

    @llama_cpp.llama_log_callback  # type: ignore[attr-defined]
    def _silent_logger(_: int, __: bytes, ___: Any) -> None:
        return

    _LLAMACPP_LOGGER = _silent_logger
    llama_cpp.llama_log_set(_silent_logger, None)
    _LLAMACPP_SILENCED = True


def _resolve_model_path(config: ModuleConfig) -> str:
    configured = config.get("model")
    if configured:
        candidate = Path(configured)
    else:
        env_candidates = (
            os.environ.get("LOCAL_LLM_GENERATION_MODEL"),
            os.environ.get("LOCAL_LLM_MODEL"),
        )
        candidate = None
        for path_str in env_candidates:
            if path_str:
                candidate = Path(path_str)
                break
    if not candidate or not candidate.is_file():
        raise typer.BadParameter(
            "Local chat requires a GGUF model file: set --model or LOCAL_LLM_GENERATION_MODEL / LOCAL_LLM_MODEL."
        )
    return str(candidate)


def _print_session_summary(
    model_path: str,
    temperature: float,
    top_p: float,
    top_k: int,
    context_window: Optional[int],
    max_tokens: int,
) -> None:
    model_name = Path(model_path).name
    console.print(
        Panel.fit(
            f"[bold cyan]Local Llama Chat Session[/bold cyan]\nModel: [green]{model_name}[/green]"
        )
    )
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", justify="left", style="bold white")
    table.add_column("Value", justify="center", style="cyan")

    table.add_row("Temperature (creativity)", f"{temperature}")
    table.add_row("Top P (nucleus sampling)", f"{top_p}")
    table.add_row("Top K (cutoff)", f"{top_k}")
    context_display = f"{context_window}" if context_window is not None else "model-defined default"
    table.add_row("Context window", context_display)
    table.add_row("Max tokens per reply", f"{max_tokens}")
    console.print(table)
    console.print("[dim]Type 'exit' or 'quit' to end.[/dim]\n")


class _LocalLlamaChat:
    def __init__(
        self,
        model_path: str,
        context_window: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise typer.BadParameter(
                "llama-cpp-python is required for local chat. Install llama-cpp-python to proceed."
            ) from exc

        _ensure_llama_silenced()

        llama_kwargs = {
            "model_path": model_path,
            "verbose": False,
            "n_ctx": context_window if context_window is not None else 0,
        }

        self._llama = Llama(**llama_kwargs)
        detected_context = None
        n_ctx_attr = getattr(self._llama, "n_ctx", None)
        if callable(n_ctx_attr):
            try:
                detected_context = n_ctx_attr()
            except TypeError:
                detected_context = None
        elif isinstance(n_ctx_attr, int):
            detected_context = n_ctx_attr
        self._context_window = (
            detected_context if detected_context is not None else context_window
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

    @property
    def context_window(self) -> Optional[int]:
        return self._context_window

    def generate(self, prompt: str) -> str:
        result = self._llama(
            prompt=prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
        )
        choices = getattr(result, "choices", [])
        if not choices:
            choices = result.get("choices", [])
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                text = choice.get("text") or choice.get("content") or ""
            else:
                text = getattr(choice, "text", "") or getattr(choice, "content", "") or ""
            return text.strip()
        return ""


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    prompt_parts: List[str] = []
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    if system_messages:
        prompt_parts.append(system_messages[0]["content"])
    for msg in messages:
        if msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"Assistant: {msg['content']}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(part for part in prompt_parts if part)


def run_chat(config: ModuleConfig) -> None:
    model_path = _resolve_model_path(config)
    temperature = config.get("temperature", 0.7)
    top_p = config.get("top_p", 0.7)
    top_k = config.get("top_k", 50)
    context_window = config.get("context_window")
    max_tokens = config.get("max_tokens", 512)
    initial_content = config.get(
        "initial_content",
        "You are an expert AI assistant for general inquiries",
    )

    chat_model = _LocalLlamaChat(
        model_path=model_path,
        context_window=context_window,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": initial_content}]

    _print_session_summary(
        model_path,
        temperature,
        top_p,
        top_k,
        chat_model.context_window,
        max_tokens,
    )

    while True:
        user_input = console.input("[bold green]You: [/bold green]").strip()
        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            return

        messages.append({"role": "user", "content": user_input})
        prompt = _build_prompt(messages)

        try:
            console.print("[bold cyan]MLX:[/bold cyan] ", end="")
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                response_text = chat_model.generate(prompt)
            console.print(response_text, style="white")
            console.print()
            messages.append({"role": "assistant", "content": response_text})
        except Exception as exc:
            console.print(f"[bold red]Error: {exc}[/bold red]\n")
