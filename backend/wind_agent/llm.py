"""Optional LLM configuration; secrets are loaded only from the local environment."""
from __future__ import annotations

import os
from dotenv import load_dotenv
from .config import ROOT

load_dotenv(ROOT / ".env")


def provider() -> str:
    selected = os.getenv("LLM_PROVIDER", "auto").lower()
    if selected not in {"auto", "openai", "anthropic", "none"}:
        raise ValueError("LLM_PROVIDER must be auto, openai, anthropic or none")
    if selected == "auto":
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "none"
    if selected == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    if selected == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
    return selected


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
