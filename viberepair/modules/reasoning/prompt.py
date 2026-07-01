from __future__ import annotations

from pathlib import Path


MAX_PROMPT_CONTEXT_CHARS = 16000

SYSTEM_PROMPT = (
    "You are a bug reasoning agent in the VibeRepair APR pipeline. "
    "Your job is to inspect the buggy code context, identify the most likely root cause, "
    "retrieve similar historical repair examples, and produce implementation-oriented repair suggestions. "
    "Always gather evidence before answering. "
    "Use the AST/static-analysis tools to inspect the buggy code, enclosing function or method, imports, "
    "calls, assignments, return statements, and control flow. "
    "When a ConDefects program id is available, open the project before using project-scoped tools and "
    "close it after you have gathered enough context. "
    "After you have a concrete root-cause hypothesis, call the similar-fix retrieval tool with the buggy "
    "code concatenated with `Root Cause: <your root cause>`. "
    "The retrieval tool is part of this VibeRepair reasoning module's configured repair-example database. "
    "Use retrieved examples only as repair-pattern evidence; do not copy irrelevant code. "
    "The final answer must be concise, evidence-grounded, and useful for a later fix-generation module. "
    "Do not generate a patch. "
    "Do not include ReAct traces, tool transcripts, markdown code fences, or extra headings in the final answer. "
    "Use exactly these labels in this order: "
    "Root Cause:; "
    "Similar Fix Examples:; "
    "Repair Suggestion:. "
    "The Repair Suggestion section should contain independent, concrete repair directions that preserve "
    "the original language, function/method/class shape, signature, entry point, stdin/stdout contract, "
    "and return/print behavior. "
    "Respond in English."
)


def _truncate_prompt_text(text: str, limit: int = MAX_PROMPT_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[truncated {omitted} characters]"


def build_reasoning_prompt(
    *,
    program_id: str | None = None,
    buggy_code: str | None = None,
    buggy_file_path: str | Path | None = None,
    language: str | None = None,
    fault_line_number: int | None = None,
    trigger_tests: str | None = None,
    error_message: str | None = None,
    additional_context: str | None = None,
) -> str:
    context_parts: list[str] = []

    if program_id:
        context_parts.append(f"ConDefects program id: {program_id}")
    if buggy_file_path:
        context_parts.append(f"Buggy file path: {Path(buggy_file_path).expanduser()}")
    if language:
        context_parts.append(f"Language: {language}")
    if fault_line_number is not None:
        context_parts.append(f"Fault line number: {fault_line_number}")
    if buggy_code and buggy_code.strip():
        context_parts.extend(
            [
                "Buggy code:",
                _truncate_prompt_text(buggy_code.strip()),
            ]
        )
    if trigger_tests and trigger_tests.strip():
        context_parts.extend(
            [
                "Trigger tests or expected behavior evidence:",
                _truncate_prompt_text(trigger_tests.strip(), limit=8000),
            ]
        )
    if error_message and error_message.strip():
        context_parts.extend(
            [
                "Observed error message or failing output evidence:",
                _truncate_prompt_text(error_message.strip(), limit=8000),
            ]
        )
    if additional_context and additional_context.strip():
        context_parts.extend(
            [
                "Additional context:",
                _truncate_prompt_text(additional_context.strip(), limit=8000),
            ]
        )

    if not context_parts:
        context_parts.append("No explicit code context was provided. Use available tools to inspect the workspace.")

    return (
        "Analyze the following buggy-code context and produce repair suggestions.\n\n"
        + "\n\n".join(context_parts)
        + "\n\nRequired process:\n"
        "1. Inspect the buggy code with AST/static-analysis tools. If a ConDefects program id is provided, "
        "open that project first and inspect the relevant function/method, classes, imports, usage, and control flow.\n"
        "2. Identify the most likely root cause using concrete code evidence and trigger/error evidence.\n"
        "3. Call the similar-fix retrieval tool using the full buggy code plus `Root Cause: <root cause>`.\n"
        "4. Compare retrieved examples against the current bug and extract only the applicable repair pattern.\n"
        "5. Return exactly these labels, in this exact order, with no extra sections:\n"
        "Root Cause:\n"
        "Similar Fix Examples:\n"
        "Repair Suggestion:\n"
        "6. The Repair Suggestion must be specific enough for a later generator to produce a correct fix, "
        "but it must not include a full patch."
    )
