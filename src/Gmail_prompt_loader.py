from pathlib import Path

class PromptLoader:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

        if not self.base_path.exists():
            raise FileNotFoundError(f"Prompt directory not found: {self.base_path}")

    def load(self, prompt_name: str) -> str:
        prompt_path = self.base_path / prompt_name

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        return prompt_path.read_text(encoding="utf-8")

    def load_with_format(self, prompt_name: str, **kwargs) -> str:
        prompt = self.load(prompt_name)
        return prompt.format(**kwargs)
