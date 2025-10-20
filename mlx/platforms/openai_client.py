from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def openai_chat_session(
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    initial_content: str = "You are an expert AI assistant for general inquiries"
):
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": initial_content
        }
    ]

    console.print(Panel.fit(f"[bold cyan]OpenAI Chat Session[/bold cyan]\nModel: [green]{model}[/green]"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", justify="left", style="bold white")
    table.add_column("Value", justify="center", style="cyan")

    table.add_row("Temperature (creativity)", f"{temperature}")
    table.add_row("Top P (nucleus sampling)", f"{top_p}")
    table.add_row("Top K (cutoff, ignored)", f"{top_k} [dim](ignored for OpenAI)[/dim]")

    console.print(table)
    console.print("[dim]Type 'exit' or 'quit' to end.[/dim]\n")

    while True:
        user_input = console.input("[bold green]You: [/bold green]").strip()

        if user_input.lower() in { "exit", "quit" }:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")

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

            console.print(f"[bold cyan]MLX: [/bold cyan]{reply}\n")
            messages.append({
                "role": "assistant",
                "content": reply
            })

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
