from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_DATASET_ROOT


OCHIAI_FL_FILENAME = "Ochiai_FL.txt"
OCHIAI_REFERENCE_TEXT = (
    "Ochiai fault localization identifies this line as having the highest suspiciousness "
    "score (T0). It may be related to the bug, but this is only a reference and should "
    "not be considered conclusive evidence."
)

class ProgramFunctionLookupError(RuntimeError):
    """Raised when a program's enclosing faulty function cannot be resolved."""


@dataclass(slots=True, frozen=True)
class ProgramContext:
    program_id: str
    program_dir: Path
    task_name: str
    language: str

    @property
    def source_path(self) -> Path:
        java_path = self.program_dir / "faultyVersion.java"
        if java_path.exists():
            return java_path
        python_path = self.program_dir / "faultyVersion.py"
        if python_path.exists():
            return python_path
        raise ProgramFunctionLookupError(
            f"Program {self.program_id} does not contain faultyVersion.java or "
            f"faultyVersion.py: {self.program_dir}"
        )

    @property
    def file_name(self) -> str:
        return self.source_path.name


@dataclass(slots=True, frozen=True)
class FaultyFunctionLocation:
    context: ProgramContext
    source_path: Path
    fault_location_path: Path
    fault_line_number: int
    start_line_index: int
    end_line_index: int
    code: str

    @property
    def fault_location_source(self) -> str:
        return self.fault_location_path.name


def iter_program_contexts(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    program_ids: Iterable[str] | None = None,
    require_fault_location: bool = True,
    languages: Iterable[str] | None = None,
) -> list[ProgramContext]:
    dataset_root = Path(dataset_root)
    wanted_ids = {
        str(program_id).strip()
        for program_id in program_ids or []
        if str(program_id).strip()
    }
    wanted_languages = {language.lower() for language in languages or []}

    contexts: list[ProgramContext] = []
    for source_path in sorted(dataset_root.rglob("faultyVersion.*")):
        if source_path.name not in {"faultyVersion.java", "faultyVersion.py"}:
            continue
        program_dir = source_path.parent
        if wanted_ids and program_dir.name not in wanted_ids:
            continue
        if require_fault_location and _fault_location_path(program_dir) is None:
            continue

        context = context_from_program_dir(program_dir, dataset_root)
        if wanted_languages and context.language.lower() not in wanted_languages:
            continue
        contexts.append(context)

    if wanted_ids:
        found = {context.program_id for context in contexts}
        missing = sorted(wanted_ids - found)
        if missing:
            raise ProgramFunctionLookupError(
                f"No matching program folders with faultyVersion files were found for: {missing}"
            )

    return contexts


def resolve_program_context(
    program_id: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    require_fault_location: bool = True,
) -> ProgramContext:
    contexts = iter_program_contexts(
        dataset_root=dataset_root,
        program_ids=[program_id],
        require_fault_location=require_fault_location,
    )
    if len(contexts) != 1:
        raise ProgramFunctionLookupError(
            f"Expected exactly one program folder for {program_id}, found {len(contexts)}"
        )
    return contexts[0]


def context_from_program_dir(
    program_dir: str | Path,
    dataset_root: str | Path,
) -> ProgramContext:
    program_dir = Path(program_dir)
    dataset_root = Path(dataset_root)
    try:
        relative = program_dir.relative_to(dataset_root)
        parts = relative.parts
    except ValueError:
        parts = program_dir.parts

    if len(parts) >= 3:
        task_name = parts[-3]
        language = parts[-2]
    else:
        language = "Java" if (program_dir / "faultyVersion.java").exists() else "Python"
        task_name = program_dir.parent.name

    return ProgramContext(
        program_id=program_dir.name,
        program_dir=program_dir,
        task_name=task_name,
        language=language,
    )


def get_ochiai_faulty_function_with_reference_comment(
    context: ProgramContext,
) -> tuple[str, int]:
    location = locate_ochiai_faulty_function(context)
    return location.code, location.fault_line_number


def locate_ochiai_faulty_function(context: ProgramContext) -> FaultyFunctionLocation:
    fault_location_path = _fault_location_path(context.program_dir)
    if fault_location_path is None:
        raise ProgramFunctionLookupError(
            f"Program {context.program_id} does not contain {OCHIAI_FL_FILENAME}: "
            f"{context.program_dir}"
        )

    fault_line_number = _read_fault_line_number(fault_location_path)
    reference_text = _reference_text_for_fault_location_source(fault_location_path.name)
    source_path = context.source_path

    if source_path.suffix == ".java":
        start, end = _java_enclosing_method_range(
            source_path,
            fault_line_number,
            context.program_id,
        )
    else:
        start, end = _python_enclosing_function_range(
            source_path,
            fault_line_number,
            context.program_id,
        )

    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    code = _format_lines_with_ochiai_comment(
        lines[start : end + 1],
        fault_line_number - 1 - start,
        "Java" if source_path.suffix == ".java" else "Python",
        reference_text=reference_text,
    )
    return FaultyFunctionLocation(
        context=context,
        source_path=source_path,
        fault_location_path=fault_location_path,
        fault_line_number=fault_line_number,
        start_line_index=start,
        end_line_index=end,
        code=code,
    )


def _fault_location_path(program_dir: Path) -> Path | None:
    candidate = program_dir / OCHIAI_FL_FILENAME
    return candidate if candidate.exists() else None


def _reference_text_for_fault_location_source(source_name: str) -> str:
    return OCHIAI_REFERENCE_TEXT


def _read_fault_line_number(path: Path) -> int:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ProgramFunctionLookupError(
            f"Invalid {path.name} value in {path}: {raw!r}"
        ) from exc


def _java_enclosing_method_range(
    java_path: Path,
    fault_line_number: int,
    program_id: str,
) -> tuple[int, int]:
    lines = java_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not 1 <= fault_line_number <= len(lines):
        raise ProgramFunctionLookupError(
            f"Fault line {fault_line_number} is out of range for {java_path}"
        )

    target_index = fault_line_number - 1
    for start_index, end_index in _find_method_ranges(lines):
        if start_index <= target_index <= end_index:
            return start_index, end_index

    raise ProgramFunctionLookupError(
        f"Could not find an enclosing Java method for program {program_id} at line "
        f"{fault_line_number}"
    )


def _python_enclosing_function_range(
    python_path: Path,
    fault_line_number: int,
    program_id: str,
) -> tuple[int, int]:
    source = python_path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    if not 1 <= fault_line_number <= len(lines):
        raise ProgramFunctionLookupError(
            f"Fault line {fault_line_number} is out of range for {python_path}"
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramFunctionLookupError(
            f"Could not parse faultyVersion.py for program {program_id}: {exc}"
        ) from exc

    target_node = _find_smallest_enclosing_python_function(tree, fault_line_number)
    if target_node is None:
        return 0, len(lines) - 1

    end_line = getattr(target_node, "end_lineno", None)
    if end_line is None:
        raise ProgramFunctionLookupError(
            f"Could not determine the end line of the enclosing Python function for "
            f"program {program_id}"
        )
    return _python_node_start_line(target_node) - 1, end_line - 1


def _find_smallest_enclosing_python_function(
    tree: ast.AST,
    fault_line_number: int,
):
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = _python_node_start_line(node)
            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                continue
            if start_line <= fault_line_number <= end_line:
                candidates.append((end_line - start_line, start_line, node))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _python_node_start_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return node.lineno


def _format_lines_with_ochiai_comment(
    lines: list[str],
    fault_line_index: int,
    language: str,
    reference_text: str,
) -> str:
    if not 0 <= fault_line_index < len(lines):
        raise ProgramFunctionLookupError("Fault line is outside the extracted faulty function")

    marked_lines = list(lines)
    marked_lines[fault_line_index] = _append_reference_comment(
        marked_lines[fault_line_index],
        language,
        reference_text=reference_text,
    )
    return "\n".join(marked_lines).rstrip()


def _append_reference_comment(line: str, language: str, reference_text: str) -> str:
    marker = "//" if language == "Java" else "#"
    comment = f"{marker} {reference_text}"
    if not line:
        return comment
    return f"{line} {comment}"


CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "do",
    "try",
    "else",
    "return",
    "throw",
    "new",
    "case",
}


def _find_method_ranges(lines: list[str]) -> list[tuple[int, int]]:
    cleaned_lines = _strip_comments_and_strings(lines)
    brace_depth = 0
    methods: list[tuple[int, int]] = []
    open_method = None
    annotation_lines: list[int] = []
    signature_start = None
    signature_parts: list[str] = []

    for idx, clean_line in enumerate(cleaned_lines):
        stripped = clean_line.strip()
        depth_before = brace_depth

        if open_method is None:
            if stripped.startswith("@") and depth_before >= 1 and not signature_parts:
                annotation_lines.append(idx)
            elif signature_parts:
                if stripped:
                    signature_parts.append(stripped)
                if "{" in stripped or ";" in stripped:
                    signature_text = " ".join(signature_parts)
                    start_idx = annotation_lines[0] if annotation_lines else signature_start
                    if "{" in stripped and _looks_like_method_signature(signature_text):
                        open_method = {"start": start_idx, "depth_before": depth_before}
                    signature_parts = []
                    annotation_lines = []
                    signature_start = None
                elif not stripped:
                    signature_parts = []
                    annotation_lines = []
                    signature_start = None
            else:
                if depth_before >= 1 and _can_start_method_signature(stripped):
                    signature_start = idx
                    signature_parts = [stripped]
                    if "{" in stripped or ";" in stripped:
                        signature_text = " ".join(signature_parts)
                        start_idx = annotation_lines[0] if annotation_lines else signature_start
                        if "{" in stripped and _looks_like_method_signature(signature_text):
                            open_method = {"start": start_idx, "depth_before": depth_before}
                        signature_parts = []
                        annotation_lines = []
                        signature_start = None
                elif stripped:
                    annotation_lines = []

        brace_depth += clean_line.count("{") - clean_line.count("}")

        if open_method is not None and brace_depth == open_method["depth_before"]:
            methods.append((open_method["start"], idx))
            open_method = None
            annotation_lines = []
            signature_parts = []
            signature_start = None

    return methods


def _can_start_method_signature(stripped_line: str) -> bool:
    if not stripped_line or "(" not in stripped_line:
        return False
    lowered = stripped_line.lower()
    return not any(
        lowered.startswith(keyword + " ") or lowered.startswith(keyword + "(")
        for keyword in CONTROL_KEYWORDS
    )


def _looks_like_method_signature(signature: str) -> bool:
    import re

    normalized = re.sub(r"\s+", " ", signature).strip()
    if "(" not in normalized or "{" not in normalized:
        return False
    if re.search(r"\b(class|interface|enum|record)\b", normalized):
        return False
    if "->" in normalized:
        return False

    before_paren = normalized.split("(", 1)[0].strip()
    if not before_paren or "=" in before_paren:
        return False

    tokens = before_paren.split()
    if not tokens or tokens[0] in CONTROL_KEYWORDS:
        return False

    name_match = re.search(r"([A-Za-z_$][\w$]*)\s*$", before_paren)
    if not name_match:
        return False
    return name_match.group(1) not in CONTROL_KEYWORDS


def _strip_comments_and_strings(lines: list[str]) -> list[str]:
    cleaned_lines = []
    in_block_comment = False

    for line in lines:
        i = 0
        cleaned_chars: list[str] = []
        in_string: str | None = None
        while i < len(line):
            char = line[i]
            next_char = line[i + 1] if i + 1 < len(line) else ""

            if in_block_comment:
                if char == "*" and next_char == "/":
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue

            if in_string is not None:
                if char == "\\":
                    i += 2
                elif char == in_string:
                    in_string = None
                    i += 1
                else:
                    i += 1
                continue

            if char == "/" and next_char == "*":
                in_block_comment = True
                i += 2
                continue
            if char == "/" and next_char == "/":
                break
            if char in {'"', "'"}:
                in_string = char
                cleaned_chars.append(" ")
                i += 1
                continue

            cleaned_chars.append(char)
            i += 1

        cleaned_lines.append("".join(cleaned_chars))

    return cleaned_lines

