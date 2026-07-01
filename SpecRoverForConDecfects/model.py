from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

MODEL_PRICES_PER_TOKEN = {
    "gpt-4o": (0.000005, 0.000015),
    "gpt-4o-2024-05-13": (0.000005, 0.000015),
    "gpt-4-turbo-2024-04-09": (0.00001, 0.00003),
    "gpt-4-0125-preview": (0.00001, 0.00003),
    "gpt-3.5-turbo-0125": (0.0000005, 0.0000015),
}


@dataclass(frozen=True)
class ChatUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: ChatUsage


class ChatModel:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY or OPENAI_KEY is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for model invocation. "
                "Install it in the active Python environment before running without --dry-run."
            ) from exc
        self.client = OpenAI(api_key=api_key)

    def complete(self, messages: list[dict[str, str]]) -> ChatResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        message = response.choices[0].message
        content = message.content or ""

        usage = response.usage
        if usage is None:
            return ChatResult(content=content, usage=ChatUsage())

        input_tokens = int(usage.prompt_tokens)
        output_tokens = int(usage.completion_tokens)
        estimated_cost = estimate_cost(self.model_name, input_tokens, output_tokens)
        return ChatResult(
            content=content,
            usage=ChatUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost,
            ),
        )


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    prices = MODEL_PRICES_PER_TOKEN.get(model_name)
    if prices is None:
        return None
    input_price, output_price = prices
    return input_tokens * input_price + output_tokens * output_price

