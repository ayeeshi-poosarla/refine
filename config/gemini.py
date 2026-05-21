"""Vertex AI / Gemini configuration for REFINE pipeline.

Uses Application Default Credentials (ADC) — no API key needed on this GCP VM.
"""

from dataclasses import dataclass


@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 1.0
    max_output_tokens: int = 16384
    project: str = "som-nero-plevriti-deidbdf"
    location: str = "us-central1"
    # GCS bucket used for Vertex AI batch input/output
    gcs_bucket: str = "vista_bench"
    gcs_prefix: str = "temp/pinnacle_templated_summaries"

    @classmethod
    def default(cls, **overrides) -> "GeminiConfig":
        return cls(**overrides)
