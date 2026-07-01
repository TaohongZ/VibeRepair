from __future__ import annotations

SYSTEM_PROMPT = """
You are the fix_specification stage in an APR pipeline.

You will receive free-form text about a flawed specification.
That text may include the flawed specification itself, issue descriptions,
failing tests, error messages, or later verification feedback.
There is no required input schema. Treat the user's text as the source of truth.

Your job is to reason through the misalignment before producing the corrected specification.

Rules:
- Use the full conversation history. Later user messages refine or correct earlier evidence.
- When new verification feedback arrives, revise the previous reasoning and corrected specification instead of ignoring prior context.
- Return exactly these sections, in this order:
  Intended Behavior:
  Diagnosis and Repair Directions:
  Fixed Specification:
- In Intended Behavior, infer the target function's correct intended behavior from the supplied evidence.
- In Diagnosis and Repair Directions, explain where and why the flawed specification deviates from the intended behavior, then list concrete directions for realigning it.
- In Fixed Specification, provide the final corrected specification only.
- Keep the reasoning concise, evidence-grounded, and implementation-oriented.
- Do not output markdown fences or extra sections.
- In the Fixed Specification section, when possible, preserve this structure:
  Function:
  Purpose:
  Signature:
  Input:
  Output:
  Behavior:
""".strip()


def build_fix_specification_initial_prompt(information: str) -> str:
    normalized_information = information.strip()
    return (
        "Free-form context for correcting a flawed specification:\n"
        f"{normalized_information}\n\n"
        "Work step by step using the required sections: first infer the intended behavior, "
        "then diagnose the root cause of the flawed specification and derive repair directions, "
        "and finally produce the fixed specification."
    )


def build_fix_specification_follow_up_prompt(additional_information: str) -> str:
    normalized_information = additional_information.strip()
    return (
        "Additional context for correcting the flawed specification:\n"
        f"{normalized_information}\n\n"
        "Use the full conversation history and this additional information to revise the corrected specification. "
        "First decide whether the new feedback indicates a semantic behavior error, a code-structure/entry-point incompatibility, or both. "
        "If the feedback reports that the generated fix code does not preserve the buggy code replacement shape, revise the specification so the next fix must preserve the original execution structure, signature, entry point, return/print contract, and surrounding API while still implementing the intended behavior. "
        "If the feedback reports failed input/output behavior, rethink the repair direction from the concrete failing cases instead of repeating the previous fix. "
        "Return the same required sections."
    )
