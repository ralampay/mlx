from mlx.platforms.openai.chat import run_chat as _run_chat


def run_chat(platform, model, temperature, top_p, top_k):
    """Backward compatible shim for legacy imports."""
    if platform != "openai":
        raise ValueError(f"Unsupported platform: {platform}")

    config = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    _run_chat(config)
