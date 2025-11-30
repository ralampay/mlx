# Chat

## Overview

The `chat` module now supports both hosted OpenAI chat sessions and fully local llama.cpp conversations. Every invocation begins with `mlx --module chat`; add `--platform` to target a specific backend (platform defaults to OpenAI when unset). Sampling flags such as `--temperature`, `--top-p`, `--top-k`, and the global `--system-prompt` apply uniformly.

## Online Chat (ChatGPT / OpenAI)

```bash
mlx --module chat \
    --platform openai \
    --model gpt-4o-mini \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 40 \
    --system-prompt "You are a helpful product strategy advisor."
```

- `--platform openai` (or omit `--platform` entirely) routes to the OpenAI streaming chat experience.
- `--model` chooses the chat completion model (e.g., `gpt-4o-mini`); sampling parameters control creativity.
- `--system-prompt` overrides the session’s initial instruction (default: `You are a general purpose assistant.`) and is sent as the first system message.
- The CLI keeps prompting you until you type `exit` or `quit`, mirroring the previous streaming behavior.

## Local Llama Chat (GGUF)

```bash
mlx --module chat \
    --platform local \
    --model /path/to/model.gguf \
    --system-prompt "You are a customer success bot." \
    --temperature 0.5 \
    --top-p 0.9
```

- Set `--platform local` (or `llama`) to load a GGUF file via `llama-cpp-python`; sampling flags are respected.
- Point `--model` at a local GGUF checkpoint, or export `LOCAL_LLM_GENERATION_MODEL` / `LOCAL_LLM_MODEL` instead.
- Install `llama_cpp_python` (`pip install llama_cpp_python`) for python bindings; the module silences llama.cpp logs automatically.
- The prompt history is reconstructed into a single text block before every generation, so replies consider the conversation to date.
- The interactive loop is identical: type `exit` or `quit` to stop.
