from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from viberepair.langchain_support import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
    require_chat_prompt_template,
)
from .prompt import SYSTEM_PROMPT, build_fix_code_prompt


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
class FixCodeResult:
    fix_specification: str
    buggy_code: str
    replacement_code: str
    raw_response: str
    model_name: str
    temperature: float
    usage: TokenUsage
    estimated_cost_usd: float | None = None
    structure_adapted: bool = False
    structure_adaptation_reason: str | None = None


class FixCodeGenerator:
    """Generate replacement code from a corrected specification and buggy code."""

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
                ("human", "{input_prompt}"),
            ]
        )
        self._model = model or build_chat_model(
            model_name=model_name,
            temperature=temperature,
        )

    def generate(self, fix_specification: str, buggy_code: str) -> FixCodeResult:
        normalized_fix_specification = _normalize_text(fix_specification, field_name="fix_specification")
        normalized_buggy_code = _normalize_text(buggy_code, field_name="buggy_code")

        input_prompt = build_fix_code_prompt(
            fix_specification=normalized_fix_specification,
            buggy_code=normalized_buggy_code,
        )
        messages = self._prompt.format_messages(input_prompt=input_prompt)
        response = self._model.invoke(messages)

        raw_response = _coerce_response_text(response).strip()
        replacement_code = extract_replacement_code(raw_response)
        usage = _extract_usage(response)
        estimated_cost_usd = self.pricing.estimate_cost_usd(usage) if self.pricing else None

        return FixCodeResult(
            fix_specification=normalized_fix_specification,
            buggy_code=normalized_buggy_code,
            replacement_code=replacement_code,
            raw_response=raw_response,
            model_name=self.model_name,
            temperature=self.temperature,
            usage=usage,
            estimated_cost_usd=estimated_cost_usd,
        )


def generate_fix_code_result(
    fix_specification: str,
    buggy_code: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> FixCodeResult:
    generator = FixCodeGenerator(
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    )
    return generator.generate(
        fix_specification=fix_specification,
        buggy_code=buggy_code,
    )


def generate_fix_code(
    fix_specification: str,
    buggy_code: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> str:
    result = generate_fix_code_result(
        fix_specification=fix_specification,
        buggy_code=buggy_code,
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    )
    return result.replacement_code


def extract_replacement_code(text: str) -> str:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Model response did not contain any code.")

    label_match = re.search(
        r"(?:^|\n)(?:Fix code|Replacement code)\s*:\s*```[a-zA-Z0-9_+-]*\s*([\s\S]*?)```",
        normalized_text,
    )
    if label_match:
        return _ensure_trailing_newline(label_match.group(1).strip())

    fenced_match = re.search(r"```[a-zA-Z0-9_+-]*\s*([\s\S]*?)```", normalized_text)
    if fenced_match:
        return _ensure_trailing_newline(fenced_match.group(1).strip())

    stripped_label = re.sub(r"^(?:Fix code|Replacement code)\s*:\s*", "", normalized_text, count=1)
    return _ensure_trailing_newline(stripped_label.strip())


def _normalize_text(value: str, field_name: str) -> str:
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized_value


def _ensure_trailing_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return f"{text}\n"


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

