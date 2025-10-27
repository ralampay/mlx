import json
from abc import ABC, abstractmethod
from .tool_registry import ToolRegistry

class BaseAgent(ABC):
    def __init__(self, model=None, temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.tools = ToolRegistry()

    # Core abstract method
    @abstractmethod
    def response(self, system_prompt: str, user_prompt: str):
        pass

    def run(self, task):
        reply = self.respond(task["system"], task["user"])

        if reply.startswith("TOOL:"):
            try:
                tool_call = json.loads(reply.replace("TOOL:", "").strip())
                tool_name = tool_call["tool"]
                tool_input = tool_call["input"]

                result = self.tools.run(tool_name, tool_input)

                return self.respond(
                    task["system"],
                    f"Tool result: \n{result}\n\nNow summarize the findings."
                )
            except Exception as e:
                return f"Error executing tool: {e}"

        return reply
