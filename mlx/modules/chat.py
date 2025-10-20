from mlx.platforms.openai_client import openai_chat_session

def run_chat(platform, model, temperature, top_p, top_k):
    if platform == "openai":
        openai_chat_session(model, temperature, top_p, top_k)
    else:
        print(f"Unsupported platform: {platform}")
