from __future__ import annotations

import ast
from dataclasses import dataclass, field
import re
import textwrap
from typing import Any

from viberepair.langchain_support import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
    require_chat_prompt_template,
)

from .prompt import SYSTEM_PROMPT, build_structure_adaptation_prompt


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
class StructureAdaptationResult:
    buggy_code: str
    original_fix_code: str
    adapted_fix_code: str
    changed: bool
    reason: str | None = None
    raw_response: str = ""
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    usage: TokenUsage = field(default_factory=TokenUsage)
    estimated_cost_usd: float | None = None
    deterministic_fallback_applied: bool = False


@dataclass(slots=True, frozen=True)
class _DeterministicAdaptation:
    adapted_fix_code: str
    changed: bool
    reason: str | None = None


class FixCodeStructureAdapter:
    """LLM-backed agent that preserves the replacement shape of a generated fix."""

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

    def adapt(
        self,
        buggy_code: str,
        fix_code: str,
        fix_specification: str | None = None,
    ) -> StructureAdaptationResult:
        normalized_buggy_code = _normalize_code(buggy_code, field_name="buggy_code")
        normalized_fix_code = _normalize_code(fix_code, field_name="fix_code")

        input_prompt = build_structure_adaptation_prompt(
            buggy_code=normalized_buggy_code,
            fix_code=normalized_fix_code,
            fix_specification=fix_specification,
        )
        messages = self._prompt.format_messages(input_prompt=input_prompt)
        response = self._model.invoke(messages)
        raw_response = _coerce_response_text(response).strip()
        llm_adapted_code = extract_adapted_fix_code(raw_response)

        deterministic_result = _apply_deterministic_structure_adaptation(
            buggy_code=normalized_buggy_code,
            fix_code=llm_adapted_code,
        )
        if deterministic_result is not None:
            adapted_fix_code = deterministic_result.adapted_fix_code
            deterministic_reason = deterministic_result.reason
            deterministic_changed = deterministic_result.changed
        else:
            adapted_fix_code = _ensure_trailing_newline(llm_adapted_code)
            deterministic_reason = None
            deterministic_changed = False

        changed = adapted_fix_code != _ensure_trailing_newline(normalized_fix_code)
        usage = _extract_usage(response)
        estimated_cost_usd = self.pricing.estimate_cost_usd(usage) if self.pricing else None
        reason = _build_reason(
            original_fix_code=normalized_fix_code,
            llm_adapted_code=llm_adapted_code,
            final_adapted_code=adapted_fix_code,
            deterministic_reason=deterministic_reason,
        )

        return StructureAdaptationResult(
            buggy_code=normalized_buggy_code,
            original_fix_code=normalized_fix_code,
            adapted_fix_code=adapted_fix_code,
            changed=changed,
            reason=reason,
            raw_response=raw_response,
            model_name=self.model_name,
            temperature=self.temperature,
            usage=usage,
            estimated_cost_usd=estimated_cost_usd,
            deterministic_fallback_applied=deterministic_changed,
        )


def adapt_fix_code_structure_result(
    buggy_code: str,
    fix_code: str,
    fix_specification: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> StructureAdaptationResult:
    adapter = FixCodeStructureAdapter(
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    )
    return adapter.adapt(
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
    )


def adapt_fix_code_structure(
    buggy_code: str,
    fix_code: str,
    fix_specification: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    model: Any | None = None,
) -> str:
    return adapt_fix_code_structure_result(
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
        model_name=model_name,
        temperature=temperature,
        pricing=pricing,
        model=model,
    ).adapted_fix_code


def extract_adapted_fix_code(text: str) -> str:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Model response did not contain any adapted fix code.")

    label_match = re.search(
        (
            r"(?:^|\n)(?:Adapted fix code|Structure-compatible fix code|"
            r"Final fix code|Replacement code|Fix code)\s*:\s*"
            r"```[a-zA-Z0-9_+-]*\s*([\s\S]*?)```"
        ),
        normalized_text,
    )
    if label_match:
        return _ensure_trailing_newline(label_match.group(1).strip())

    fenced_match = re.search(r"```[a-zA-Z0-9_+-]*\s*([\s\S]*?)```", normalized_text)
    if fenced_match:
        return _ensure_trailing_newline(fenced_match.group(1).strip())

    stripped_label = re.sub(
        (
            r"^(?:Adapted fix code|Structure-compatible fix code|Final fix code|"
            r"Replacement code|Fix code)\s*:\s*"
        ),
        "",
        normalized_text,
        count=1,
    )
    return _ensure_trailing_newline(stripped_label.strip())


def _apply_deterministic_structure_adaptation(
    buggy_code: str,
    fix_code: str,
) -> _DeterministicAdaptation | None:
    python_result = _adapt_python_structure(buggy_code, fix_code)
    if python_result is not None:
        return python_result

    java_result = _adapt_java_structure(buggy_code, fix_code)
    if java_result is not None:
        return java_result

    return None


def _adapt_python_structure(buggy_code: str, fix_code: str) -> _DeterministicAdaptation | None:
    try:
        buggy_tree = ast.parse(buggy_code)
        fix_tree = ast.parse(fix_code)
    except SyntaxError:
        return None

    if not _is_python_top_level_script(buggy_tree):
        return None

    wrapper = _find_python_function_wrapper(fix_tree)
    if wrapper is None:
        return None

    prefix_nodes, function_node, suffix_nodes = wrapper
    if _contains_return(function_node):
        return None

    prefix = _source_for_nodes(fix_code, prefix_nodes)
    body = _source_for_nodes(fix_code, _function_body_without_docstring(function_node))
    suffix = _source_for_nodes(fix_code, suffix_nodes)

    body = textwrap.dedent(body).strip()
    if not body:
        return None

    pieces = [part.strip() for part in (prefix, body, suffix) if part.strip()]
    adapted = _ensure_trailing_newline("\n\n".join(pieces))
    return _DeterministicAdaptation(
        adapted_fix_code=adapted,
        changed=adapted != _ensure_trailing_newline(fix_code),
        reason="unwrapped generated Python function to match top-level buggy script",
    )


def _is_python_top_level_script(tree: ast.Module) -> bool:
    if not tree.body:
        return False
    if len(tree.body) == 1 and isinstance(
        tree.body[0],
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
    ):
        return False
    return any(
        not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )


def _find_python_function_wrapper(
    tree: ast.Module,
) -> tuple[list[ast.stmt], ast.FunctionDef | ast.AsyncFunctionDef, list[ast.stmt]] | None:
    body = list(tree.body)
    if not body:
        return None

    function_indexes = [
        index
        for index, node in enumerate(body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(function_indexes) != 1:
        return None

    function_index = function_indexes[0]
    prefix_nodes = body[:function_index]
    function_node = body[function_index]
    suffix_nodes = body[function_index + 1 :]

    if not all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in prefix_nodes):
        return None
    if suffix_nodes and not _is_only_main_guard_or_function_call(suffix_nodes, function_node.name):
        return None

    kept_suffix: list[ast.stmt] = []
    return prefix_nodes, function_node, kept_suffix


def _is_only_main_guard_or_function_call(nodes: list[ast.stmt], function_name: str) -> bool:
    if len(nodes) != 1:
        return False
    node = nodes[0]
    if isinstance(node, ast.Expr) and _is_call_to(node.value, function_name):
        return True
    if isinstance(node, ast.If):
        return (
            _is_main_name_check(node.test)
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Expr)
            and _is_call_to(node.body[0].value, function_name)
        )
    return False


def _is_call_to(value: ast.AST, function_name: str) -> bool:
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == function_name


def _is_main_name_check(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _function_body_without_docstring(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(function_node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _contains_return(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(node, ast.Return) for node in ast.walk(function_node))


def _source_for_nodes(source: str, nodes: list[ast.stmt]) -> str:
    if not nodes:
        return ""
    lines = source.splitlines()
    start = min(node.lineno for node in nodes)
    end = max(getattr(node, "end_lineno", node.lineno) for node in nodes)
    return "\n".join(lines[start - 1 : end])


def _adapt_java_structure(buggy_code: str, fix_code: str) -> _DeterministicAdaptation | None:
    if re.search(r"\bclass\s+\w+", buggy_code):
        return None
    class_body = _extract_single_java_class_body(fix_code)
    if class_body is None:
        return None
    adapted = _ensure_trailing_newline(textwrap.dedent(class_body).strip())
    if not adapted.strip():
        return None
    return _DeterministicAdaptation(
        adapted_fix_code=adapted,
        changed=adapted != _ensure_trailing_newline(fix_code),
        reason="unwrapped generated Java class to match non-class buggy snippet",
    )


def _extract_single_java_class_body(source: str) -> str | None:
    match = re.search(r"(?:public\s+)?class\s+\w+[^\{]*\{", source)
    if not match:
        return None
    open_brace_index = source.find("{", match.start())
    close_brace_index = _matching_brace_index(source, open_brace_index)
    if close_brace_index is None:
        return None
    before = source[: match.start()].strip()
    after = source[close_brace_index + 1 :].strip()
    if before or after:
        return None
    return source[open_brace_index + 1 : close_brace_index]


def _matching_brace_index(source: str, open_brace_index: int) -> int | None:
    depth = 0
    for index in range(open_brace_index, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _normalize_code(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _ensure_trailing_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return f"{text}\n"


def _build_reason(
    original_fix_code: str,
    llm_adapted_code: str,
    final_adapted_code: str,
    deterministic_reason: str | None,
) -> str | None:
    original = _ensure_trailing_newline(original_fix_code)
    llm_adapted = _ensure_trailing_newline(llm_adapted_code)
    final_adapted = _ensure_trailing_newline(final_adapted_code)
    reasons: list[str] = []
    if llm_adapted != original:
        reasons.append("LLM adapted fix code to match the buggy snippet structure")
    if deterministic_reason and final_adapted != llm_adapted:
        reasons.append(deterministic_reason)
    if not reasons:
        return None
    return "; ".join(reasons)


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
