# benchutils.py
from openai import OpenAI

class OpenAIWrapper:
    def __init__(self, model, temperature=0):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def fast_run(self, prompt, response_format=None):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format=response_format,
        )
        return resp.choices[0].message.content

def create_LLM(cfg):
    if cfg["type"] == "openai":
        return OpenAIWrapper(cfg["model"], cfg.get("temperature", 0))
    raise ValueError(f"Unknown LLM type: {cfg['type']}")
