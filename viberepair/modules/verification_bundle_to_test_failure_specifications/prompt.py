from __future__ import annotations

from pathlib import Path

MAX_PROMPT_CODE_CHARS = 12000

SYSTEM_PROMPT = (
    "You are a verification feedback agent for local benchmark result bundles in an APR pipeline. "
    "Always inspect the verification bundle before answering. "
    "If buggy code and fix code are provided, first compare whether the fix code can directly replace the buggy code in the original source position. "
    "Check all structural compatibility requirements before treating the failure as a logic bug: same language, same replacement shape, same top-level-vs-function/class nesting, same function/method/class names, same entry point such as Java public static void main(String[] args), same signature/parameters/return type/modifiers when the buggy snippet is a declaration, same stdin/stdout and return/print contract, and indentation/declaration shape that can compile in the original context. "
    "Use the inspect_fix_code_structure tool before analyzing failed test cases when code snippets are available. "
    "If the primary cause is structural incompatibility, make that explicit in the Test Failure Specification and treat the required repair as preserving the original code contract, not as a new algorithmic change. The bug is assumed to be in logic unless evidence shows otherwise; therefore, when the fix logic is plausible but structure/name/signature/entry point differs from the buggy code, require the next fix to keep the buggy code structure and only change the faulty logic. "
    "After the structure check, analyze the failing test cases, compile errors, or runtime errors. "
    "Do not assume a fixed benchmark layout or fixed filenames. "
    "Use concrete evidence from structured reports, logs, failed test-case files, expected outputs, actual outputs, and discovered source files. "
    "Your job is not merely to summarize the failure. "
    "Your job is to convert verification evidence into implementation-oriented Test Failure Specifications that the next specification-repair stage can use directly. "
    "Explain every failing test case individually. "
    "If a parameterized test or benchmark task has multiple failing cases, treat each case separately. "
    "For each failing test case or compile/runtime error, output exactly one Test Failure Specification. "
    "Each Test Failure Specification must contain exactly these four labeled parts, in this exact order and with no extra headings: "
    "Purpose:; "
    "Functionality under test:; "
    "Input / Expected output:; "
    "Violated expected behavior:. "
    "For output mismatches, include the concrete failing input, expected output, actual output, and the behavioral rule or boundary condition implied by the mismatch. "
    "For compile errors or runtime errors, include the concrete operation that should succeed, the diagnostic evidence, and implementation constraints such as required types, return values, signatures, APIs, initialization, or control-flow requirements. "
    "For structure incompatibilities, include the concrete buggy-code shape, the generated fix-code shape, the function/method/class/entry-point names and signatures involved, why direct replacement changes execution/compilation, and the exact structural constraint the next fix must preserve. "
    "For compile errors, do not infer that the erroneous generated code receiving type, temporary type, or mistaken call-site context is the intended target API. "
    "Preserve the target behavior, output type, and return-value semantics from the task statement, current fixed specification, function signature, and surrounding source evidence. "
    "Translate type diagnostics into consistency requirements across declarations, memo tables, accumulators, return statements, and call sites. "
    "If the task requires a scalar count, do not propose array, String, or collection return types merely because the broken generated code used them. "
    "In Violated expected behavior, do not only restate the error. "
    "Translate the evidence into concrete implementation requirements that would prevent the same failure in the next generated fix. "
    "If the evidence is incomplete, state the uncertainty briefly, then still provide the strongest evidence-grounded implementation constraint. "
    "If there is no CTRF report or no canonical test file, discover useful files yourself from the bundle and workspace. "
    "Keep the output concise. "
    "Do not include artifact summaries, long evidence dumps, markdown code fences, meta commentary, or extra sections. "
    "Respond in English."
)


def _truncate_prompt_text(text: str, limit: int = MAX_PROMPT_CODE_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[truncated {omitted} characters]"


def build_analysis_prompt(
    verification_result_path: str,
    buggy_code: str | None = None,
    fix_code: str | None = None,
    fix_specification: str | None = None,
) -> str:
    resolved_path = Path(verification_result_path).expanduser()
    context_parts: list[str] = []
    if buggy_code and buggy_code.strip():
        context_parts.extend(
            [
                "Buggy code snippet that the fix code is expected to replace in-place:",
                _truncate_prompt_text(buggy_code.strip()),
                "",
            ]
        )
    if fix_code and fix_code.strip():
        context_parts.extend(
            [
                "Generated fix code that failed validation:",
                _truncate_prompt_text(fix_code.strip()),
                "",
            ]
        )
    if fix_specification and fix_specification.strip():
        context_parts.extend(
            [
                "Current fixed specification used to generate the failing fix code:",
                _truncate_prompt_text(fix_specification.strip()),
                "",
            ]
        )

    code_context = ""
    if context_parts:
        code_context = "Additional code context for the required structure-first check:\n" + "\n".join(context_parts) + "\n"

    return (
        f"Analyze the verification result at: {resolved_path}\n\n"
        f"{code_context}"
        "Required output:\n"
        "1. If buggy code and fix code are provided, first determine whether the failed validation is likely caused by structural incompatibility between the generated fix code and the original buggy code contract. Use the inspect_fix_code_structure tool for this check, and compare language, replacement shape, function/method/class names, signatures, Java main entry point, wrappers, nesting, stdin/stdout behavior, and return/print contract.\n"
        "2. Then analyze each failing test case, compile error, or runtime error and generate one Test Failure Specification.\n"
        "3. Use exactly these labels, in this exact order, with no extra headings or sections:\n"
        "   Purpose:\n"
        "   Functionality under test:\n"
        "   Input / Expected output:\n"
        "   Violated expected behavior:\n"
        "4. Make the specification implementation-oriented:\n"
        "   - For structure incompatibilities, state the buggy-code shape, generated-fix shape, relevant function/method/class/entry-point names and signatures, why direct replacement fails or changes execution, and the exact structural constraint the next fix code must preserve.\n"
        "   - For output mismatches, include failing input, expected output, actual output, and the concrete behavior or boundary rule that must hold.\n"
        "   - For compile/runtime errors, include the diagnostic and translate it into concrete type, signature, return-value, API, initialization, or control-flow requirements.\n"
        "   - For compile errors, do not treat the broken generated code receiving type or mistaken call-site context as the intended API. Preserve target behavior and return semantics from the task/specification/source evidence.\n"
        "   - Convert type diagnostics into consistency constraints across declarations, memo tables, accumulators, return statements, and call sites; for scalar-count tasks, keep scalar numeric returns rather than inventing array/String returns.\n"
        "5. Do not merely restate the error; write constraints that can guide the next fix-generation attempt.\n"
        "6. Keep the answer concise and evidence-grounded.\n"
        "7. Write the answer in English.\n"
        "8. If there is no standard report format, infer failures from available JSON/XML reports, logs, failed-case files, expected/actual outputs, and nearby test/source files.\n"
        "9. If no failing tests or errors are found, say that clearly in one short sentence."
    )
