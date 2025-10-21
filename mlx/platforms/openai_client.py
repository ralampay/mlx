from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

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
            console.print("[bold cyan]MLX:[/bold cyan] ", end="")
            partial_content = ""

            # Stream the response
            with client.chat.completions.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p
            ) as stream:
                for event in stream:
                    # The new SDK sends different event types
                    if event.type == "content.delta":
                        # This is the actual streamed text
                        content_piece = event.delta
                        console.print(content_piece, end="", style="white")
                        partial_content += content_piece
                    elif event.type == "message.stop":
                        console.print("\n")
                        break
                    elif event.type == "error":
                        console.print(f"[red]{event.error}[/red]")
                        break

                # Ensure the stream finishes before moving on
                stream.until_done()
           
            console.print()
            messages.append({"role": "assistant", "content": partial_content})
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
