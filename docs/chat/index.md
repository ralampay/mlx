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
