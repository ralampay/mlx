from platform import openai
from .base_agent import BaseAgent

class OpenAIAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini", temperature=0.3):
        super().__init__(model=model, temperature=temperature)

    def respond(self, system_prompt, user_prompt):
        response = openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ]
        )

        return response.choices[0].message.content.strip()
