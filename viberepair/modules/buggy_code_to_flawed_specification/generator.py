from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from viberepair.langchain_support import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
    require_chat_prompt_template,
)

from .prompt import SYSTEM_PROMPT


@dataclass(slots=True, frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(slots=True, frozen=True)
class ModelPricing:
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    def estimate_cost_usd(self, usage: TokenUsage) -> float | None:
        if self.input_price_per_million is None or self.output_price_per_million is None:
            return None
        if usage.input_tokens is None or usage.output_tokens is None:
            return None
        return (
            usage.input_tokens * self.input_price_per_million
            + usage.output_tokens * self.output_price_per_million
        ) / 1_000_000


DEFAULT_GPT_4O_PRICING = ModelPricing(
    input_price_per_million=2.5,
    output_price_per_million=10.0,
)


@dataclass(slots=True, frozen=True)
class FlawedSpecificationResult:
    buggy_code: str
    flawed_specification: str
    model_name: str
    temperature: float
    usage: TokenUsage
    estimated_cost_usd: float | None = None


class FlawedSpecificationGenerator:
    """Generate a flawed specification directly from buggy code."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
        model: Any | None = None,
    ) -> None:
        ChatPromptTemplate = require_chat_prompt_template()
        self.model_name = model_name
        self.temperature = temperature
        self.pricing = pricing
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Buggy code:\n{buggy_code}"),
            ]
        )
        self._model = model or build_chat_model(
            model_name=model_name,
            temperature=temperature,
        )

    def generate(self, buggy_code: str) -> FlawedSpecificationResult:
        normalized_buggy_code = buggy_code.strip()
        if not normalized_buggy_code:
            raise ValueError("buggy_code must not be empty.")

        messages = self._prompt.format_messages(buggy_code=normalized_buggy_code)
        response = self._model.invoke(messages)

        flawed_specification = _coerce_response_text(response).strip()
        usage = _extract_usage(response)
        estimated_cost_usd = self.pricing.estimate_cost_usd(usage) if self.pricing else None

        return FlawedSpecificationResult(
            buggy_code=normalized_buggy_code,
            flawed_specification=flawed_specification,
            model_name=self.model_name,
            temperature=self.temperature,
            usage=usage,
            estimated_cost_usd=estimated_cost_usd,
        )


def generate_flawed_specification_result(
    buggy_code: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> FlawedSpecificationResult:
    generator = FlawedSpecificationGenerator(
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    )
    return generator.generate(buggy_code)


def generate_flawed_specification(
    buggy_code: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> str:
    result = generate_flawed_specification_result(
        buggy_code=buggy_code,
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    )
    return result.flawed_specification


def _coerce_response_text(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def _extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage_metadata", None) or {}
    return TokenUsage(
        input_tokens=_coerce_int(usage.get("input_tokens")),
        output_tokens=_coerce_int(usage.get("output_tokens")),
    )


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None

