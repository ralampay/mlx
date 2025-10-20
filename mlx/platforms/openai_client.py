from openai import OpenAI

def openai_chat_session(
    model: str,
    temperature: float,
    top_p: float,
    top_k: int
):
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": "You are an expert AI assistant for general inquiries"
        }
    ]

    print(f"OpenAI Chat Session ({model})")
    print(f"Temperature: {temperature}")
    print(f"Top K: {top_k}")
    print(f"Top P: {top_p}")
    print(f"Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in { "exit", "quit" }:
            print("Goodbye!")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p
            )

            reply = response.choices[0].message.content

            print(f"MLX: {reply}\n")
            messages.append({
                "role": "assistant",
                "content": reply
            })

        except Exception as e:
            print(f"Error: {e}")
