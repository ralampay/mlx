# Chat

## Overview

The `chat` module provides a conversational interface backed by OpenAI APIs (currently the only supported platform). Commands always start with `mlx --module chat --platform openai`, and you can customize the model and sampling parameters through CLI flags.

## Sample Conversation

```bash
mlx --module chat \
    --platform openai \
    --model gpt-4o-mini \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 40
```

- `--model`: OpenAI chat model to drive the conversation.
- `--temperature`: Controls creativity/riskiness; higher values produce more diverse replies.
- `--top-p`: Nucleus sampling cutoff; works with temperature to shape randomness.
- `--top-k`: Limits the number of candidate tokens (accepted for parity, though OpenAI ignores it).

The CLI handles the session loop for you, prompting for input and streaming replies until you exit.

## System Prompt Customization

Supply `--system-prompt` to tweak the assistant's initial instructions (default: `You are a general purpose assistant.`). This option flows through both OpenAI and local chat backends so you can inject the exact tone or guardrails you need for the session.

## Local Llama Chat (GGUF)

Local chat runs entirely on-device via a GGUF model served by `llama-cpp-python`. Use it like this:

```bash
mlx --module chat \
    --platform local \
    --model /path/to/model.gguf \
    --system-prompt "You are a customer success bot."
```

- `--platform local` (alias `llama`) routes the CLI to the local backend.
- `--model` or the `LOCAL_LLM_GENERATION_MODEL` / `LOCAL_LLM_MODEL` env vars must point to a GGUF file.
- `llama-cpp-python` must be installed (`pip install llama_cpp_python`), and the module will silence llama.cpp logs automatically.
- Sampling knobs such as `--temperature`, `--top-p`, and `--top-k` work just like they do for OpenAI.

Local chat keeps the entire prompt history within the conversation and streams responses until you type `exit` or `quit`.
