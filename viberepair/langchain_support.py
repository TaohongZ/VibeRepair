from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
MIN_LANGCHAIN_VERSION = "1.0.0"


def _missing_dependency_message() -> str:
    return (
        "VibeRepair modules require `langchain>=1.0.0`, `langchain-core`, and "
        "`langchain-openai`. Install them with "
        "`pip install -U langchain langchain-core langchain-openai`."
    )


def require_chat_openai() -> Any:
    """Import ChatOpenAI lazily so modules fail with a clear error message."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(_missing_dependency_message()) from exc

    return ChatOpenAI


def require_create_agent() -> Any:
    """Import LangChain's create_agent lazily for agent-style pipeline stages."""
    try:
        from langchain.agents import create_agent
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(_missing_dependency_message()) from exc

    return create_agent


def require_chat_prompt_template() -> Any:
    """Import ChatPromptTemplate lazily for consistent LangChain usage."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(_missing_dependency_message()) from exc

    return ChatPromptTemplate


def ensure_openai_api_key() -> None:
    """Validate the shared OpenAI credential expected by VibeRepair modules."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")


def build_chat_model(
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Any:
    """Create the shared chat model used by modular VibeRepair stages."""
    ensure_openai_api_key()
    ChatOpenAI = require_chat_openai()
    return ChatOpenAI(model=model_name, temperature=temperature)
