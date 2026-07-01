from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from viberepair.langchain_support import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
)

from .prompt import (
    SYSTEM_PROMPT,
    build_fix_specification_follow_up_prompt,
    build_fix_specification_initial_prompt,
)


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


_SECTION_LABELS = (
    "Intended Behavior",
    "Diagnosis and Repair Directions",
    "Fixed Specification",
)
_SECTION_LABEL_PATTERN = "|".join(re.escape(label) for label in _SECTION_LABELS)
_SECTION_HEADING_RE = re.compile(
    rf"(?im)^\s*(?:#{{1,6}}\s*)?(?:\d+\.\s*)?(?:\*\*)?"
    rf"(?P<label>{_SECTION_LABEL_PATTERN})(?:\*\*)?\s*:\s*(?P<inline>.*?)\s*$"
)


@dataclass(slots=True, frozen=True)
class _FixSpecificationSections:
    intended_behavior: str = ""
    diagnosis_and_repair_directions: str = ""
    fix_specification: str = ""


@dataclass(slots=True, frozen=True)
class FixSpecificationResult:
    input_information: str
    fix_specification: str
    model_name: str
    temperature: float
    usage: TokenUsage
    estimated_cost_usd: float | None = None
    history_length: int = 0
    intended_behavior: str = ""
    diagnosis_and_repair_directions: str = ""
    raw_response: str = ""


class FixSpecificationSession:
    """History-aware corrected-specification reasoning generator."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.pricing = pricing
        self._model = model or build_chat_model(
            model_name=model_name,
            temperature=temperature,
        )
        self._messages: list[BaseMessage] = []

    def start(self, information: str) -> FixSpecificationResult:
        normalized_information = _normalize_information(information)
        self._messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_fix_specification_initial_prompt(normalized_information)),
        ]
        return self._invoke_and_store(normalized_information)

    def first_align(self, information: str) -> FixSpecificationResult:
        return self.start(information)

    def add_information(self, additional_information: str) -> FixSpecificationResult:
        if not self._messages:
            raise ValueError("The fix_specification session has not been started yet.")

        normalized_information = _normalize_information(additional_information)
        self._messages.append(
            HumanMessage(content=build_fix_specification_follow_up_prompt(normalized_information))
        )
        return self._invoke_and_store(normalized_information)

    def additional_info(self, additional_information: str) -> FixSpecificationResult:
        return self.add_information(additional_information)

    def get_history(self) -> list[BaseMessage]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages = []

    def _invoke_and_store(self, input_information: str) -> FixSpecificationResult:
        response = self._model.invoke(self._messages)
        raw_response = _coerce_response_text(response).strip()
        sections = _parse_fix_specification_response(raw_response)
        self._messages.append(AIMessage(content=raw_response))

        usage = _extract_usage(response)
        estimated_cost_usd = self.pricing.estimate_cost_usd(usage) if self.pricing else None

        return FixSpecificationResult(
            input_information=input_information,
            fix_specification=sections.fix_specification,
            model_name=self.model_name,
            temperature=self.temperature,
            usage=usage,
            estimated_cost_usd=estimated_cost_usd,
            history_length=len(self._messages),
            intended_behavior=sections.intended_behavior,
            diagnosis_and_repair_directions=sections.diagnosis_and_repair_directions,
            raw_response=raw_response,
        )


def generate_fix_specification_result(
    information: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> FixSpecificationResult:
    session = FixSpecificationSession(
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    )
    return session.start(information)


def generate_fix_specification(
    information: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> str:
    result = generate_fix_specification_result(
        information=information,
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    )
    return result.fix_specification


def _parse_fix_specification_response(raw_response: str) -> _FixSpecificationSections:
    sections = _extract_labeled_sections(raw_response)
    fix_specification = sections.get("Fixed Specification") or raw_response
    return _FixSpecificationSections(
        intended_behavior=sections.get("Intended Behavior", ""),
        diagnosis_and_repair_directions=sections.get(
            "Diagnosis and Repair Directions",
            "",
        ),
        fix_specification=fix_specification.strip(),
    )


def _extract_labeled_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION_HEADING_RE.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        inline = _clean_inline_heading_text(match.group("inline"))
        body = text[match.end() : end].strip()
        value = "\n".join(part for part in (inline, body) if part).strip()
        sections[match.group("label")] = value
    return sections


def _clean_inline_heading_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned == "**":
        return ""
    if cleaned.endswith("**"):
        cleaned = cleaned[:-2].strip()
    return cleaned


def _normalize_information(information: str) -> str:
    normalized_information = information.strip()
    if not normalized_information:
        raise ValueError("information must not be empty.")
    return normalized_information


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

