"""Gemini API configuration for REFINE pipeline."""

import os
from dataclasses import dataclass


@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 1.0
    max_output_tokens: int = 16384

    @classmethod
    def from_env(cls, **overrides) -> "GeminiConfig":
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set.\n"
                "Export it before running:\n"
                "  export GOOGLE_API_KEY=your_key_here\n"
                "Find your key at: https://aistudio.google.com/apikey"
            )
        return cls(**overrides)

    @property
    def api_key(self) -> str:
        return os.environ["GOOGLE_API_KEY"]
