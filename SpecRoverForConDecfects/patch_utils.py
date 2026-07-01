from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExtractStatus(str, Enum):
    APPLICABLE_PATCH = "APPLICABLE_PATCH"
    RAW_PATCH_BUT_UNPARSED = "RAW_PATCH_BUT_UNPARSED"
    RAW_PATCH_BUT_UNMATCHED = "RAW_PATCH_BUT_UNMATCHED"
    MATCHED_BUT_EMPTY_DIFF = "MATCHED_BUT_EMPTY_DIFF"
    MATCHED_BUT_EMPTY_ORIGIN = "MATCHED_BUT_EMPTY_ORIGIN"


@dataclass(frozen=True)
class Edit:
    filename: str
    before: str
    after: str


@dataclass(frozen=True)
class PatchExtractionResult:
    status: ExtractStatus
    summary: str
    fixed_code: str
    edits: list[Edit]
    applied_edit_count: int

    @property
    def applicable(self) -> bool:
        return self.status == ExtractStatus.APPLICABLE_PATCH


def parse_edits(chat_string: str) -> list[Edit]:
    def parse_in_fence(lines: list[str]) -> list[Edit]:
        sus_contents = ["# Rest of the code..."]
        lines = [line for line in lines if line.strip() not in sus_contents]

        content = "\n".join(lines)
        file_pattern = re.compile(r"<file>(.*?)</file>", re.DOTALL)
        original_pattern = re.compile(r"<original>(.*?)</original>", re.DOTALL)
        patched_pattern = re.compile(r"<patched>(.*?)</patched>", re.DOTALL)

        files = file_pattern.findall(content)
        originals = original_pattern.findall(content)
        patched = patched_pattern.findall(content)

        edits: list[Edit] = []
        for file, original, replacement in zip(files, originals, patched):
            edits.append(
                Edit(
                    filename=file.strip(),
                    before=original.strip("\n"),
                    after=replacement.strip("\n"),
                )
            )
        return edits

    edits: list[Edit] = []
    current_edit: list[str] = []
    in_fence = False

    for line in chat_string.split("\n"):
        if line.startswith("```") and in_fence:
            edits.extend(parse_in_fence(current_edit))
            current_edit = []
            in_fence = False
            continue
        if line.startswith("```") and not in_fence:
            in_fence = True
            continue
        if in_fence:
            current_edit.append(line)

    if not edits:
        edits.extend(parse_in_fence(chat_string.splitlines()))

    return edits


def extract_fixed_function_from_response(
    response: str,
    faulty_function: str,
    source_file_name: str,
    language: str,
) -> PatchExtractionResult:
    try:
        raw_edits = parse_edits(response)
    except Exception as exc:
        return PatchExtractionResult(
            status=ExtractStatus.RAW_PATCH_BUT_UNPARSED,
            summary=f"Exception {exc} happened when parsing edits.",
            fixed_code="",
            edits=[],
            applied_edit_count=0,
        )

    edits = [
        edit
        for edit in raw_edits
        if _edit_targets_source_file(edit.filename, source_file_name)
        and not _is_test_file(edit.filename)
    ]

    if not edits:
        return PatchExtractionResult(
            status=ExtractStatus.RAW_PATCH_BUT_UNPARSED,
            summary="No edits can be parsed.",
            fixed_code="",
            edits=raw_edits,
            applied_edit_count=0,
        )

    if any(not edit.before.strip() for edit in edits):
        return PatchExtractionResult(
            status=ExtractStatus.MATCHED_BUT_EMPTY_ORIGIN,
            summary="Please contain **non-whitespace** original code snippet in edits.",
            fixed_code="",
            edits=edits,
            applied_edit_count=0,
        )

    fixed_code = _strip_fault_reference_comment(faulty_function)
    applied_count = 0
    unmatched_indexes: list[int] = []

    for idx, edit in enumerate(edits):
        updated = apply_edit_to_text(fixed_code, edit.before, edit.after, language)
        if updated is None:
            unmatched_indexes.append(idx)
            continue
        fixed_code = updated
        applied_count += 1

    if applied_count == 0:
        fallback = _fallback_replacement_if_full_function(edits, faulty_function)
        if fallback is None:
            return PatchExtractionResult(
                status=ExtractStatus.RAW_PATCH_BUT_UNMATCHED,
                summary="None of the edits can match the original program.",
                fixed_code="",
                edits=edits,
                applied_edit_count=0,
            )
        fixed_code = fallback
        applied_count = 1

    if _normalize_code(fixed_code) == _normalize_code(_strip_fault_reference_comment(faulty_function)):
        return PatchExtractionResult(
            status=ExtractStatus.MATCHED_BUT_EMPTY_DIFF,
            summary="The matched edits do not introduce any change to the codebase.",
            fixed_code="",
            edits=edits,
            applied_edit_count=applied_count,
        )

    summary = ""
    if unmatched_indexes:
        numbers = ", ".join(str(index + 1) for index in unmatched_indexes)
        summary = f"Edits number {numbers} cannot be matched to the original program."

    return PatchExtractionResult(
        status=ExtractStatus.APPLICABLE_PATCH,
        summary=summary,
        fixed_code=fixed_code.rstrip(),
        edits=edits,
        applied_edit_count=applied_count,
    )


def apply_edit_to_text(
    text: str,
    before: str,
    after: str,
    language: str,
) -> str | None:
    before_clean = _strip_fault_reference_comment(before)
    after_clean = _strip_fault_reference_comment(after)

    candidates = _replacement_candidates(text, before_clean, after_clean)
    for candidate in candidates:
        if _syntax_ok(candidate, language):
            return candidate.rstrip()
    return None


def _replacement_candidates(text: str, before: str, after: str) -> list[str]:
    exact = _replace_exact(text, before, after)
    candidates = [exact] if exact is not None else []

    line_match = _replace_by_stripped_lines(text, before, after, reindent=False)
    if line_match is not None:
        candidates.append(line_match)

    reindented = _replace_by_stripped_lines(text, before, after, reindent=True)
    if reindented is not None:
        candidates.append(reindented)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def _replace_exact(text: str, before: str, after: str) -> str | None:
    if before not in text:
        return None
    return text.replace(before, after, 1)


def _replace_by_stripped_lines(
    text: str,
    before: str,
    after: str,
    reindent: bool,
) -> str | None:
    text_lines = text.splitlines()
    before_lines = before.splitlines()
    if not before_lines:
        return None

    cleaned_text_lines = [line.strip() for line in text_lines]
    cleaned_before_lines = [line.strip() for line in before_lines]

    match_start = -1
    match_end = -1
    for idx in range(len(cleaned_text_lines) - len(cleaned_before_lines) + 1):
        if cleaned_text_lines[idx : idx + len(cleaned_before_lines)] == cleaned_before_lines:
            match_start = idx
            match_end = idx + len(cleaned_before_lines)
            break

    if match_start < 0:
        return None

    after_lines = after.splitlines()
    if reindent:
        target_indent = _leading_ws(text_lines[match_start])
        source_indent = _leading_ws(_first_non_empty_line(after_lines))
        after_lines = [
            target_indent + line[len(source_indent) :]
            if line.startswith(source_indent) and line.strip()
            else line
            for line in after_lines
        ]

    new_lines = text_lines[:match_start] + after_lines + text_lines[match_end:]
    return "\n".join(new_lines)


def _first_non_empty_line(lines: list[str]) -> str:
    for line in lines:
        if line.strip():
            return line
    return lines[0] if lines else ""


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _fallback_replacement_if_full_function(
    edits: list[Edit],
    faulty_function: str,
) -> str | None:
    normalized_faulty = _normalize_code(_strip_fault_reference_comment(faulty_function))
    for edit in edits:
        if _normalize_code(_strip_fault_reference_comment(edit.before)) == normalized_faulty:
            return _strip_fault_reference_comment(edit.after).rstrip()
    return None


def _syntax_ok(code: str, language: str) -> bool:
    if language.lower() != "python":
        return True
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _edit_targets_source_file(filename: str, source_file_name: str) -> bool:
    if not filename:
        return True
    normalized = filename.replace("\\", "/")
    source = source_file_name.replace("\\", "/")
    return normalized == source or normalized.endswith("/" + source) or Path(normalized).name == source


def _is_test_file(filename: str) -> bool:
    parts = set(Path(filename.replace("\\", "/")).parts)
    return "test" in parts or "tests" in parts or filename.endswith("_test.py")


def _strip_fault_reference_comment(code: str) -> str:
    cleaned_lines = []
    markers = ("Ochiai fault localization identifies this line",)
    for line in code.splitlines():
        updated = line
        for marker in markers:
            index = updated.find(marker)
            if index == -1:
                continue
            comment_index = max(updated.rfind("#", 0, index), updated.rfind("//", 0, index))
            if comment_index != -1:
                updated = updated[:comment_index].rstrip()
        cleaned_lines.append(updated)
    return "\n".join(cleaned_lines).rstrip()


def _normalize_code(code: str) -> str:
    return "\n".join(line.strip() for line in code.strip().splitlines() if line.strip())

