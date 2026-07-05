from dataclasses import dataclass
import os
from dotenv import load_dotenv


class InvalidLLMConfigurationError(ValueError):
    """Raised when the OpenAI LLM configuration contains invalid values."""
    pass

@dataclass
class Config:
    api_key: str | None
    base_url: str | None
    llm: str | None
    system_prompt_path: str

    def llm_enabled(self) -> bool:
        return bool(
            self.api_key
            and self.base_url
            and self.llm
        )


def load_config() -> Config:
    load_dotenv()

    return Config(
        api_key=os.getenv("CHUTES_API_KEY"),
        base_url=os.getenv("CHUTES_API_BASE"),
        llm=os.getenv("CHUTES_LLM"),
        system_prompt_path="./prompts/system_prompt.txt"
    )