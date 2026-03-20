from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("MODEL", "claude-sonnet-4-6"))

settings = Settings()
