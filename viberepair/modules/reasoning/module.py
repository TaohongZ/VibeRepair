from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from viberepair.langchain_support import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE as SHARED_DEFAULT_TEMPERATURE,
    ensure_openai_api_key,
    require_chat_openai,
    require_create_agent,
)

from .prompt import SYSTEM_PROMPT, build_reasoning_prompt


DEFAULT_MODEL = DEFAULT_MODEL_NAME
DEFAULT_TEMPERATURE = SHARED_DEFAULT_TEMPERATURE
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
CONDEFECTS_ROOT = REPO_ROOT / "ConDefects"
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("VIBEREPAIR_CONDEFECTS_DATASET_ROOT", CONDEFECTS_ROOT / "dataset_2024")
)
DEFAULT_TEST_ROOT = Path(os.environ.get("VIBEREPAIR_CONDEFECTS_TEST_ROOT", CONDEFECTS_ROOT / "Test"))
RAG_SERVER_HOST = os.environ.get("VIBEREPAIR_REASONING_RAG_HOST", "127.0.0.1")
RAG_SERVER_PORT = int(os.environ.get("VIBEREPAIR_REASONING_RAG_PORT", "5000"))
RAG_SEARCH_URL = os.environ.get(
    "VIBEREPAIR_REASONING_SEARCH_URL",
    os.environ.get(
        "VIBEREPAIR_CONDEFECTS_SEARCH_URL",
        f"http://{RAG_SERVER_HOST}:{RAG_SERVER_PORT}/search",
    ),
)
DEFAULT_REASONING_RETRIEVAL_BASE = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_BASE",
        REPO_ROOT / "ReinFixForConDecfects" / "retrieval_base",
    )
)
RETRIEVAL_CSV_PATH = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_CSV",
        DEFAULT_REASONING_RETRIEVAL_BASE / "embedded_cause_97k.csv",
    )
)
RETRIEVAL_PARQUET_PATH = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_PARQUET",
        DEFAULT_REASONING_RETRIEVAL_BASE / "embedded_cause_97k.parquet",
    )
)
RETRIEVAL_PICKLE_PATH = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RETRIEVAL_PICKLE",
        DEFAULT_REASONING_RETRIEVAL_BASE / "embedded_cause_97k.pkl",
    )
)
RAG_AUTO_START = os.environ.get("VIBEREPAIR_REASONING_AUTO_START_RAG", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RAG_STARTUP_TIMEOUT_SECONDS = float(
    os.environ.get("VIBEREPAIR_REASONING_RAG_STARTUP_TIMEOUT_SECONDS", "120")
)
RAG_SERVER_LOG_PATH = Path(
    os.environ.get(
        "VIBEREPAIR_REASONING_RAG_SERVER_LOG",
        Path("/tmp") / "viberepair_reasoning_rag_server.log",
    )
)
MAX_TOOL_TEXT_CHARS = 24000
MAX_READ_LINES = 260
DEFAULT_GPT_4O_PRICING = None


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
class ReasoningContext:
    program_id: str | None = None
    buggy_code: str | None = None
    buggy_file_path: str | None = None
    language: str | None = None
    fault_line_number: int | None = None
    trigger_tests: str | None = None
    error_message: str | None = None
    additional_context: str | None = None
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    test_root: str = str(DEFAULT_TEST_ROOT)


class ProgramFunctionLookupError(RuntimeError):
    """Raised when a ConDefects program context cannot be resolved."""


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
            f"Program {self.program_id} does not contain faultyVersion.java or faultyVersion.py: "
            f"{self.program_dir}"
        )

    @property
    def file_name(self) -> str:
        return self.source_path.name


@dataclass(slots=True)
class CodeEntity:
    name: str
    start_line: int
    end_line: int
    code: str
    parameters: list[str]
    kind: str


@dataclass(slots=True)
class ProjectState:
    context: ProgramContext
    source_path: Path
    source: str
    lines: list[str]
    language: str
    methods: list[CodeEntity]
    classes: list[CodeEntity]


_current_project: ProjectState | None = None


@dataclass(slots=True, frozen=True)
class _ReasoningSections:
    root_cause: str = ""
    similar_fix_examples: str = ""
    repair_suggestion: str = ""


@dataclass(slots=True, frozen=True)
class ReasoningResult:
    repair_suggestion: str
    model_name: str
    temperature: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    estimated_cost_usd: float | None = None
    program_id: str | None = None
    buggy_file_path: str | None = None
    language: str | None = None
    fault_line_number: int | None = None
    root_cause: str = ""
    similar_fix_examples: str = ""
    raw_response: str = ""


_LAST_REASONING_USAGE = TokenUsage()
_RAG_SERVER_PROCESS: subprocess.Popen[Any] | None = None
_SECTION_LABELS = (
    "Root Cause",
    "Similar Fix Examples",
    "Repair Suggestion",
)
_SECTION_LABEL_PATTERN = "|".join(re.escape(label) for label in _SECTION_LABELS)
_SECTION_HEADING_RE = re.compile(
    rf"(?im)^\s*(?:#{{1,6}}\s*)?(?:\d+\.\s*)?(?:\*\*)?"
    rf"(?P<label>{_SECTION_LABEL_PATTERN})(?:\*\*)?\s*:\s*(?P<inline>.*?)\s*$"
)


def _require_langchain() -> tuple[Any, Any]:
    return require_create_agent(), require_chat_openai()


def _truncate(text: str, limit: int = MAX_TOOL_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[truncated {omitted} characters]"


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _read_text_file(path: str | os.PathLike[str], max_chars: int = MAX_TOOL_TEXT_CHARS) -> str:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    text = target.read_text(encoding="utf-8", errors="replace")
    return _truncate(text, max_chars)


def _normalize_project_name(proj_name: str) -> str:
    normalized = str(proj_name).strip().strip('"').strip("'")
    if normalized.endswith("_buggy"):
        normalized = normalized[: -len("_buggy")]
    return normalized


def _resolve_program_context(
    program_id: str,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
) -> ProgramContext:
    dataset_root = Path(dataset_root).expanduser()
    normalized = _normalize_project_name(program_id)
    matches = [
        source_path.parent
        for source_path in sorted(dataset_root.rglob("faultyVersion.*"))
        if source_path.name in {"faultyVersion.java", "faultyVersion.py"}
        and source_path.parent.name == normalized
    ]
    unique_matches = sorted(dict.fromkeys(matches))
    if len(unique_matches) != 1:
        raise ProgramFunctionLookupError(
            f"Expected exactly one program folder for {normalized}, found {len(unique_matches)}"
        )
    return _context_from_program_dir(unique_matches[0], dataset_root)


def _context_from_program_dir(
    program_dir: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
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


def _build_trigger_context(
    context: ProgramContext,
    test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
    max_cases: int = 3,
    max_chars_per_file: int = 1600,
) -> list[dict[str, str]]:
    test_dir = _test_dir_for_task(context.task_name, test_root)
    if test_dir is None:
        return []

    in_dir = test_dir / "in"
    out_dir = test_dir / "out"
    if not in_dir.is_dir() or not out_dir.is_dir():
        return []

    inputs = sorted(in_dir.glob("sample_*.txt")) + sorted(in_dir.glob("random_*.txt"))
    cases: list[dict[str, str]] = []
    for input_path in inputs[:max_cases]:
        output_path = out_dir / input_path.name
        cases.append(
            {
                "test_case": input_path.name,
                "input": _read_small_text(input_path, max_chars_per_file),
                "expected_output": _read_small_text(output_path, max_chars_per_file)
                if output_path.exists()
                else "",
            }
        )
    return cases


def _test_dir_for_task(task_name: str, test_root: str | os.PathLike[str]) -> Path | None:
    if "_" not in task_name:
        candidate = Path(test_root) / task_name
        return candidate if candidate.is_dir() else None

    contest, problem = task_name.rsplit("_", 1)
    problem_dir_name = "Ex" if problem.lower() == "ex" else problem.upper()
    candidate = Path(test_root) / contest / problem_dir_name
    return candidate if candidate.is_dir() else None


def _read_small_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _build_reasoning_context(
    *,
    program_id: str | None = None,
    buggy_code: str | None = None,
    buggy_file_path: str | os.PathLike[str] | None = None,
    language: str | None = None,
    fault_line_number: int | None = None,
    trigger_tests: str | None = None,
    error_message: str | None = None,
    additional_context: str | None = None,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
) -> ReasoningContext:
    normalized_program_id = _normalize_optional_text(program_id)
    normalized_buggy_code = _normalize_optional_text(buggy_code)
    normalized_file_path = str(Path(buggy_file_path).expanduser()) if buggy_file_path else None
    normalized_language = _normalize_optional_text(language)
    normalized_trigger_tests = _normalize_optional_text(trigger_tests)

    if normalized_file_path and normalized_buggy_code is None:
        normalized_buggy_code = _read_text_file(normalized_file_path)

    if normalized_program_id:
        try:
            program_context = _resolve_program_context(
                normalized_program_id,
                dataset_root=dataset_root,
            )
            normalized_file_path = normalized_file_path or str(program_context.source_path)
            normalized_language = normalized_language or program_context.language

            if normalized_buggy_code is None:
                normalized_buggy_code = _read_text_file(program_context.source_path)

            if normalized_trigger_tests is None:
                trigger_context = _build_trigger_context(program_context, test_root=test_root)
                if trigger_context:
                    normalized_trigger_tests = _json_dump(trigger_context)
        except ProgramFunctionLookupError:
            if normalized_buggy_code is None and normalized_file_path is None:
                raise

    if normalized_language is None:
        normalized_language = _infer_language(
            code=normalized_buggy_code,
            file_path=normalized_file_path,
        )

    return ReasoningContext(
        program_id=normalized_program_id,
        buggy_code=normalized_buggy_code,
        buggy_file_path=normalized_file_path,
        language=normalized_language,
        fault_line_number=fault_line_number,
        trigger_tests=normalized_trigger_tests,
        error_message=_normalize_optional_text(error_message),
        additional_context=_normalize_optional_text(additional_context),
        dataset_root=str(Path(dataset_root).expanduser()),
        test_root=str(Path(test_root).expanduser()),
    )


def _infer_language(code: str | None = None, file_path: str | None = None) -> str | None:
    if file_path:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".py":
            return "Python"
        if suffix == ".java":
            return "Java"

    if not code:
        return None
    try:
        ast.parse(code)
        return "Python"
    except SyntaxError:
        pass

    if re.search(r"\b(public|private|protected|class|static|void|int|String)\b", code):
        return "Java"
    return None


def _inspect_buggy_code_ast_payload(
    buggy_code: str | None,
    buggy_file_path: str | os.PathLike[str] | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    if not buggy_code or not buggy_code.strip():
        return {
            "available": False,
            "reason": "buggy_code was not provided",
            "buggy_file_path": str(buggy_file_path) if buggy_file_path else None,
        }

    code = buggy_code.strip("\n")
    inferred_language = language or _infer_language(code=code, file_path=str(buggy_file_path) if buggy_file_path else None)
    if inferred_language == "Python":
        return _inspect_python_ast_payload(code, buggy_file_path=buggy_file_path)
    if inferred_language == "Java":
        return _inspect_java_static_payload(code, buggy_file_path=buggy_file_path)
    return {
        "available": True,
        "language": inferred_language or "unknown",
        "buggy_file_path": str(buggy_file_path) if buggy_file_path else None,
        "line_count": len(code.splitlines()),
        "note": "Could not infer a supported language; AST details are unavailable.",
    }


def _inspect_python_ast_payload(
    code: str,
    buggy_file_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "available": True,
            "language": "Python",
            "buggy_file_path": str(buggy_file_path) if buggy_file_path else None,
            "syntax_error": {
                "message": exc.msg,
                "line": exc.lineno,
                "offset": exc.offset,
            },
            "line_count": len(lines),
        }

    functions = []
    classes = []
    imports = []
    calls = set()
    assignments = []
    returns = []
    conditions = []
    loops = []
    raises = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(_python_source_line(lines, node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "signature": _python_function_signature(node),
                    "start_line": _python_node_start_line(node),
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "parameters": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                }
            )
        elif isinstance(node, ast.ClassDef):
            classes.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                }
            )
        elif isinstance(node, ast.Call):
            name = _python_callable_name(node.func)
            if name:
                calls.add(name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assignments.append(_python_source_line(lines, node))
        elif isinstance(node, ast.Return):
            returns.append(_python_source_line(lines, node))
        elif isinstance(node, ast.If):
            conditions.append(_python_source_line(lines, node))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            loops.append(_python_source_line(lines, node))
        elif isinstance(node, ast.Raise):
            raises.append(_python_source_line(lines, node))

    module_level_executable = [
        _python_source_line(lines, node)
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    return {
        "available": True,
        "language": "Python",
        "buggy_file_path": str(buggy_file_path) if buggy_file_path else None,
        "line_count": len(lines),
        "imports": imports,
        "classes": classes,
        "functions": functions,
        "module_level_executable": module_level_executable,
        "calls": sorted(calls),
        "assignments": assignments[:80],
        "return_statements": returns[:80],
        "if_conditions": conditions[:80],
        "loops": loops[:80],
        "raise_statements": raises[:80],
    }


def _python_node_start_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return getattr(node, "lineno", 1)


def _python_function_signature(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = function_node.args
    parts: list[str] = []
    parts.extend(arg.arg for arg in args.posonlyargs)
    parts.extend(arg.arg for arg in args.args)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    parts.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return f"{function_node.name}({', '.join(parts)})"


def _python_callable_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _python_callable_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _python_source_line(lines: list[str], node: ast.AST) -> dict[str, Any]:
    line_no = getattr(node, "lineno", None)
    if line_no is None or not 1 <= line_no <= len(lines):
        return {"line": None, "code": ""}
    return {"line": line_no, "code": lines[line_no - 1].strip()}


def _inspect_java_static_payload(
    code: str,
    buggy_file_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    lines = code.splitlines()
    methods = []
    for start_index, end_index in _find_method_ranges(lines):
        method_code = "\n".join(lines[start_index : end_index + 1]).rstrip()
        method_name, parameters = _java_signature_name_and_params(method_code)
        methods.append(
            {
                "name": method_name or f"method_at_line_{start_index + 1}",
                "start_line": start_index + 1,
                "end_line": end_index + 1,
                "parameters": parameters,
                "signature": _first_java_signature_line(method_code),
                "calls": _java_calls(method_code),
                "assignments": _lines_matching(method_code.splitlines(), r"(?<![=!<>])=(?!=)")[:80],
                "return_statements": _lines_matching(method_code.splitlines(), r"\breturn\b")[:80],
                "throw_statements": _lines_matching(method_code.splitlines(), r"\bthrow\b")[:80],
                "if_conditions": _lines_matching(method_code.splitlines(), r"\bif\s*\(")[:80],
                "control_flow": _lines_matching(
                    method_code.splitlines(),
                    r"\b(if|else|for|while|switch|case|try|catch)\b",
                )[:80],
            }
        )

    return {
        "available": True,
        "language": "Java",
        "buggy_file_path": str(buggy_file_path) if buggy_file_path else None,
        "line_count": len(lines),
        "imports": [
            {"line": index, "code": line.strip()}
            for index, line in enumerate(lines, start=1)
            if line.strip().startswith("import ")
        ],
        "classes": _java_classes(lines),
        "methods": methods,
    }


def _java_signature_name_and_params(code: str) -> tuple[str, list[str]]:
    signature_parts = []
    for line in code.splitlines():
        stripped = line.strip()
        signature_parts.append(stripped)
        if "{" in stripped:
            break
    signature = " ".join(signature_parts)
    before_paren = signature.split("(", 1)[0].strip()
    name_match = re.search(r"([A-Za-z_$][\w$]*)\s*$", before_paren)
    params_match = re.search(r"\((.*?)\)", signature)
    params = []
    if params_match:
        params = [part.strip() for part in params_match.group(1).split(",") if part.strip()]
    return name_match.group(1) if name_match else "", params


def _first_java_signature_line(code: str) -> str:
    parts = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped:
            parts.append(stripped)
        if "{" in stripped:
            break
    return " ".join(parts)


def _java_classes(lines: list[str]) -> list[dict[str, Any]]:
    classes = []
    class_pattern = re.compile(r"\b(class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")
    for index, line in enumerate(lines, start=1):
        match = class_pattern.search(line)
        if match:
            classes.append({"name": match.group(2), "kind": match.group(1), "line": index})
    return classes


def _lines_matching(lines: list[str], pattern: str) -> list[str]:
    regex = re.compile(pattern)
    return [line.strip() for line in lines if regex.search(line)]


def _java_calls(code: str) -> list[str]:
    calls = []
    for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*\(", code):
        name = match.group(1)
        if name not in {"if", "for", "while", "switch", "catch", "return", "new"}:
            calls.append(name)
    return sorted(set(calls))


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
                    cleaned_chars.append(" ")
                    i += 1
                else:
                    cleaned_chars.append(" ")
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


def _open_project(proj_name: str, dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT) -> str:
    normalized = _normalize_project_name(proj_name)
    context = _resolve_program_context(normalized, dataset_root=dataset_root)
    source_path = context.source_path
    source = source_path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    language = "Java" if source_path.suffix == ".java" else "Python"

    global _current_project
    _current_project = ProjectState(
        context=context,
        source_path=source_path,
        source=source,
        lines=lines,
        language=language,
        methods=_extract_methods(source, language),
        classes=_extract_classes(source, language),
    )
    return (
        f"Opened ConDefects project {context.program_id} from {source_path}. "
        "Static local analysis is active."
    )


def _close_project(proj_name: str = "") -> str:
    global _current_project
    normalized = _normalize_project_name(proj_name) if proj_name else (
        _current_project.context.program_id if _current_project else ""
    )
    _current_project = None
    return f"Closed ConDefects project {normalized}."


def _require_project() -> ProjectState:
    if _current_project is None:
        raise ProgramFunctionLookupError("No ConDefects project is open. Call open_project first.")
    return _current_project


def _split_two_args(input_string: str) -> tuple[str, str]:
    parts = input_string.split(",", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _file_matches(source_path: Path, file_name: str) -> bool:
    normalized = file_name.strip()
    if not normalized:
        return True
    return normalized == source_path.name or normalized in str(source_path)


def _identify_variable(input_string: str) -> str:
    project = _require_project()
    variable_name, file_name = _split_two_args(input_string)
    variable_name = variable_name.strip().replace('"', "")
    file_name = file_name.strip().replace('"', "")
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."

    pattern = re.compile(rf"\b{re.escape(variable_name)}\b")
    hits = []
    for index, line in enumerate(project.lines, start=1):
        if pattern.search(line):
            hits.append({"line": index, "code": line.strip(), "file": project.source_path.name})
        if len(hits) >= 80:
            break
    return _truncate(_json_dump(hits))


def _trace_method_usage(method_name: str) -> str:
    project = _require_project()
    method_name = method_name.strip().replace('"', "")
    hits = []
    call_pattern = re.compile(rf"\b{re.escape(method_name)}\s*\(")
    for index, line in enumerate(project.lines, start=1):
        if call_pattern.search(line):
            hits.append({"line": index, "code": line.strip(), "file": project.source_path.name})
    return _truncate(_json_dump(hits))


def _find_method_in_file(input_string: str) -> str:
    project = _require_project()
    method_name, file_name = _split_two_args(input_string.replace('"', ""))
    method_name = method_name.strip()
    file_name = file_name.strip()
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."

    methods = _find_methods(project, method_name)
    if not methods:
        return f"No method named {method_name} was found in {project.source_path.name}."
    return _truncate(_json_dump([_entity_to_dict(method) for method in methods]))


def _analyze_method_details(input_string: str) -> str:
    project = _require_project()
    method_name, file_name = _split_two_args(input_string.replace('"', ""))
    method_name = method_name.strip()
    file_name = file_name.strip()
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."

    methods = _find_methods(project, method_name)
    if not methods:
        return f"No method named {method_name} was found in {project.source_path.name}."
    return _truncate(_json_dump([_analyze_entity(project, method) for method in methods]))


def _analyze_method_control_flow(method_name: str) -> str:
    project = _require_project()
    method_name = method_name.strip().replace('"', "")
    methods = _find_methods(project, method_name)
    if not methods:
        return f"No method named {method_name} was found in {project.source_path.name}."
    payload = [
        {"method": method.name, "control_flow": _control_flow_lines(method.code)}
        for method in methods
    ]
    return _truncate(_json_dump(payload))


def _find_class_loc(class_name: str) -> str:
    project = _require_project()
    class_name = class_name.strip().replace('"', "")
    classes = [klass for klass in project.classes if class_name in klass.name or klass.name in class_name]
    if not classes:
        return f"No class named {class_name} was found in {project.source_path.name}."
    return _truncate(
        _json_dump(
            [
                {
                    "class": klass.name,
                    "file": project.source_path.name,
                    "start_line": klass.start_line,
                    "end_line": klass.end_line,
                }
                for klass in classes
            ]
        )
    )


def _identify_class(class_name: str) -> str:
    project = _require_project()
    class_name = class_name.strip().replace('"', "")
    classes = [klass for klass in project.classes if class_name in klass.name or klass.name in class_name]
    if not classes:
        return f"No class named {class_name} was found in {project.source_path.name}."
    return _truncate(_json_dump([_entity_to_dict(klass) for klass in classes]))


def _get_imports(file_name: str) -> str:
    project = _require_project()
    file_name = file_name.strip().replace('"', "")
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."
    imports = [
        {"line": index, "code": line.strip(), "file": project.source_path.name}
        for index, line in enumerate(project.lines, start=1)
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    return _truncate(_json_dump(imports))


def _extract_methods(source: str, language: str) -> list[CodeEntity]:
    if language == "Python":
        return _extract_python_methods(source)
    return _extract_java_methods(source)


def _extract_python_methods(source: str) -> list[CodeEntity]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [CodeEntity("module", 1, len(lines), source.rstrip(), [], "module")]

    methods: list[CodeEntity] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = _python_node_start_line(node)
            end_line = getattr(node, "end_lineno", node.lineno)
            methods.append(
                CodeEntity(
                    name=node.name,
                    start_line=start_line,
                    end_line=end_line,
                    code="\n".join(lines[start_line - 1 : end_line]).rstrip(),
                    parameters=[arg.arg for arg in node.args.args],
                    kind="function",
                )
            )

    if not methods:
        methods.append(CodeEntity("module", 1, len(lines), source.rstrip(), [], "module"))
    return sorted(methods, key=lambda item: (item.start_line, item.end_line))


def _extract_java_methods(source: str) -> list[CodeEntity]:
    lines = source.splitlines()
    methods: list[CodeEntity] = []
    for start, end in _find_method_ranges(lines):
        code = "\n".join(lines[start : end + 1]).rstrip()
        name, parameters = _java_signature_name_and_params(code)
        methods.append(
            CodeEntity(
                name=name or f"method_at_line_{start + 1}",
                start_line=start + 1,
                end_line=end + 1,
                code=code,
                parameters=parameters,
                kind="method",
            )
        )
    if not methods:
        methods.append(CodeEntity("module", 1, len(lines), source.rstrip(), [], "module"))
    return methods


def _extract_classes(source: str, language: str) -> list[CodeEntity]:
    if language == "Python":
        return _extract_python_classes(source)
    return _extract_java_classes(source)


def _extract_python_classes(source: str) -> list[CodeEntity]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", node.lineno)
            classes.append(
                CodeEntity(
                    name=node.name,
                    start_line=start_line,
                    end_line=end_line,
                    code="\n".join(lines[start_line - 1 : end_line]).rstrip(),
                    parameters=[],
                    kind="class",
                )
            )
    return sorted(classes, key=lambda item: (item.start_line, item.end_line))


def _extract_java_classes(source: str) -> list[CodeEntity]:
    lines = source.splitlines()
    class_pattern = re.compile(r"\b(class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")
    classes: list[CodeEntity] = []
    for index, line in enumerate(lines):
        match = class_pattern.search(line)
        if not match:
            continue
        end_index = _matching_brace_end(lines, index)
        classes.append(
            CodeEntity(
                name=match.group(2),
                start_line=index + 1,
                end_line=end_index + 1,
                code="\n".join(lines[index : end_index + 1]).rstrip(),
                parameters=[],
                kind=match.group(1),
            )
        )
    return classes


def _matching_brace_end(lines: list[str], start_index: int) -> int:
    depth = 0
    seen_open = False
    for index in range(start_index, len(lines)):
        depth += lines[index].count("{")
        if depth:
            seen_open = True
        depth -= lines[index].count("}")
        if seen_open and depth <= 0:
            return index
    return len(lines) - 1


def _find_methods(project: ProjectState, method_name: str) -> list[CodeEntity]:
    normalized = method_name.strip()
    if not normalized:
        return project.methods
    return [
        method
        for method in project.methods
        if method.name == normalized or normalized in method.name or method.name in normalized
    ]


def _entity_to_dict(entity: CodeEntity) -> dict[str, Any]:
    return {
        "name": entity.name,
        "kind": entity.kind,
        "start_line": entity.start_line,
        "end_line": entity.end_line,
        "parameters": entity.parameters,
        "code": entity.code,
    }


def _analyze_entity(project: ProjectState, entity: CodeEntity) -> dict[str, Any]:
    code_lines = entity.code.splitlines()
    if project.language == "Python":
        return _analyze_python_entity(entity, code_lines)
    return {
        "method": entity.name,
        "parameters": entity.parameters,
        "filename": project.source_path.name,
        "throw_statements": _lines_matching(code_lines, r"\bthrow\b"),
        "if_conditions": _lines_matching(code_lines, r"\bif\s*\("),
        "method_calls": _java_calls(entity.code),
        "assignments": _lines_matching(code_lines, r"(?<![=!<>])=(?!=)"),
        "return_statements": _lines_matching(code_lines, r"\breturn\b"),
    }


def _analyze_python_entity(entity: CodeEntity, code_lines: list[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(entity.code)
    except SyntaxError:
        return {
            "method": entity.name,
            "parameters": entity.parameters,
            "throw_statements": _lines_matching(code_lines, r"\braise\b"),
            "if_conditions": _lines_matching(code_lines, r"^\s*if\b"),
            "method_calls": _python_calls_by_regex(entity.code),
            "assignments": _lines_matching(code_lines, r"(?<![=!<>])=(?!=)"),
            "return_statements": _lines_matching(code_lines, r"^\s*return\b"),
        }

    return {
        "method": entity.name,
        "parameters": entity.parameters,
        "throw_statements": _python_node_lines(tree, ast.Raise, code_lines),
        "if_conditions": _python_if_conditions(tree, code_lines),
        "method_calls": _python_call_names(tree),
        "assignments": _python_node_lines(tree, (ast.Assign, ast.AnnAssign, ast.AugAssign), code_lines),
        "return_statements": _python_node_lines(tree, ast.Return, code_lines),
    }


def _control_flow_lines(code: str) -> list[str]:
    regex = re.compile(r"\b(if|else|elif|for|while|switch|case|try|catch)\b")
    return [line.strip() for line in code.splitlines() if regex.search(line)]


def _python_calls_by_regex(code: str) -> list[str]:
    return sorted(
        {
            match.group(1)
            for match in re.finditer(r"\b([A-Za-z_][\w]*)\s*\(", code)
            if match.group(1) not in {"if", "for", "while", "return"}
        }
    )


def _python_call_names(tree: ast.AST) -> list[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _python_callable_name(node.func)
            if name:
                names.add(name)
    return sorted(names)


def _python_node_lines(
    tree: ast.AST,
    node_type: type[ast.AST] | tuple[type[ast.AST], ...],
    code_lines: list[str],
) -> list[str]:
    results = []
    for node in ast.walk(tree):
        if isinstance(node, node_type):
            line_no = getattr(node, "lineno", None)
            if line_no is not None and 1 <= line_no <= len(code_lines):
                results.append(code_lines[line_no - 1].strip())
    return results


def _python_if_conditions(tree: ast.AST, code_lines: list[str]) -> list[str]:
    conditions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            line_no = getattr(node, "lineno", None)
            if line_no is not None and 1 <= line_no <= len(code_lines):
                conditions.append(code_lines[line_no - 1].strip())
    return conditions


def _format_buggy_source_excerpt(
    buggy_code: str | None,
    start_line: int = 1,
    end_line: int = MAX_READ_LINES,
) -> str:
    if not buggy_code:
        return "No buggy code is available in this reasoning context."
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    if end_line - start_line + 1 > MAX_READ_LINES:
        end_line = start_line + MAX_READ_LINES - 1

    lines = buggy_code.splitlines()
    excerpt = [
        f"{idx:04d}: {lines[idx - 1]}"
        for idx in range(start_line, min(end_line, len(lines)) + 1)
    ]
    return "\n".join(excerpt) if excerpt else "<no content in requested range>"


def _parse_retrieval_threshold(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rag_health_url(search_url: str = RAG_SEARCH_URL) -> str:
    parsed = urlparse(search_url)
    return urlunparse(parsed._replace(path="/health", params="", query="", fragment=""))


def _is_local_rag_search_url(search_url: str = RAG_SEARCH_URL) -> bool:
    parsed = urlparse(search_url)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and hostname in {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    }


def _rag_server_is_healthy(search_url: str = RAG_SEARCH_URL, timeout: float = 1.0) -> bool:
    try:
        response = requests.get(_rag_health_url(search_url), timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def ensure_reasoning_rag_server(search_url: str = RAG_SEARCH_URL) -> None:
    """Start the local reasoning RAG server when the configured local URL is not alive."""
    if not RAG_AUTO_START or not _is_local_rag_search_url(search_url):
        return
    if _rag_server_is_healthy(search_url):
        return

    global _RAG_SERVER_PROCESS
    if _RAG_SERVER_PROCESS is None or _RAG_SERVER_PROCESS.poll() is not None:
        parsed = urlparse(search_url)
        host = parsed.hostname or RAG_SERVER_HOST
        if host in {"localhost", "0.0.0.0", "::1"}:
            host = "127.0.0.1"
        port = parsed.port or RAG_SERVER_PORT

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else f"{REPO_ROOT}:{existing_pythonpath}"
        env.setdefault("VIBEREPAIR_REASONING_RETRIEVAL_BASE", str(DEFAULT_REASONING_RETRIEVAL_BASE))
        env.setdefault("VIBEREPAIR_REASONING_RETRIEVAL_CSV", str(RETRIEVAL_CSV_PATH))
        env.setdefault("VIBEREPAIR_REASONING_RETRIEVAL_PARQUET", str(RETRIEVAL_PARQUET_PATH))
        env.setdefault("VIBEREPAIR_REASONING_RETRIEVAL_PICKLE", str(RETRIEVAL_PICKLE_PATH))

        RAG_SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "viberepair.modules.reasoning.rag_search_server",
            "--host",
            host,
            "--port",
            str(port),
            "--csv-path",
            str(RETRIEVAL_CSV_PATH),
            "--parquet-path",
            str(RETRIEVAL_PARQUET_PATH),
            "--pickle-path",
            str(RETRIEVAL_PICKLE_PATH),
        ]
        with RAG_SERVER_LOG_PATH.open("a", encoding="utf-8") as log_handle:
            _RAG_SERVER_PROCESS = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
            )

    deadline = time.monotonic() + RAG_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _rag_server_is_healthy(search_url):
            return
        if _RAG_SERVER_PROCESS is not None and _RAG_SERVER_PROCESS.poll() is not None:
            raise RuntimeError(
                "VibeRepair reasoning RAG server exited before becoming healthy. "
                f"See log: {RAG_SERVER_LOG_PATH}"
            )
        time.sleep(0.5)

    raise TimeoutError(
        "Timed out waiting for VibeRepair reasoning RAG server to become healthy. "
        f"Health URL: {_rag_health_url(search_url)}; log: {RAG_SERVER_LOG_PATH}"
    )


def build_reasoning_agent(
    *,
    program_id: str | None = None,
    buggy_code: str | None = None,
    buggy_file_path: str | os.PathLike[str] | None = None,
    language: str | None = None,
    fault_line_number: int | None = None,
    trigger_tests: str | None = None,
    error_message: str | None = None,
    additional_context: str | None = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
) -> Any:
    create_agent, ChatOpenAI = _require_langchain()
    context = _build_reasoning_context(
        program_id=program_id,
        buggy_code=buggy_code,
        buggy_file_path=buggy_file_path,
        language=language,
        fault_line_number=fault_line_number,
        trigger_tests=trigger_tests,
        error_message=error_message,
        additional_context=additional_context,
        dataset_root=dataset_root,
        test_root=test_root,
    )
    ensure_openai_api_key()
    ensure_reasoning_rag_server(RAG_SEARCH_URL)

    def inspect_buggy_code_ast() -> str:
        """Inspect the provided buggy code with Python AST or Java static parsing."""
        return _truncate(
            _json_dump(
                _inspect_buggy_code_ast_payload(
                    context.buggy_code,
                    buggy_file_path=context.buggy_file_path,
                    language=context.language,
                )
            )
        )

    def read_buggy_source(start_line: int = 1, end_line: int = MAX_READ_LINES) -> str:
        """Read the current buggy code context with line numbers."""
        return _truncate(_format_buggy_source_excerpt(context.buggy_code, start_line, end_line))

    def open_project(proj_name: str = "") -> str:
        """Open a ConDefects project for project-scoped static analysis tools."""
        target = (proj_name or context.program_id or "").strip()
        if not target:
            return "No ConDefects program id is available. Use inspect_buggy_code_ast instead."
        return _open_project(target, dataset_root=context.dataset_root)

    def close_project(proj_name: str = "") -> str:
        """Close the current ConDefects project after project-scoped analysis."""
        target = (proj_name or context.program_id or "").strip()
        if not target:
            return "No ConDefects program id is available."
        return _close_project(target)

    def identify_variable_usage(variable_name: str, file_name: str = "") -> str:
        """Find definitions and usages of a variable in the opened project."""
        return _identify_variable(f"{variable_name}, {file_name}")

    def trace_method_usage(method_name: str) -> str:
        """Trace call sites for a method in the opened project."""
        return _trace_method_usage(method_name)

    def find_method_in_file(method_name: str, file_name: str = "") -> str:
        """Find a method/function and return its location and code in the opened project."""
        effective_file = file_name or (Path(context.buggy_file_path).name if context.buggy_file_path else "")
        return _find_method_in_file(f"{method_name}, {effective_file}")

    def analyze_method_details(method_name: str, file_name: str = "") -> str:
        """Analyze calls, assignments, returns, throws, and branch conditions for a method/function."""
        effective_file = file_name or (Path(context.buggy_file_path).name if context.buggy_file_path else "")
        return _analyze_method_details(f"{method_name}, {effective_file}")

    def analyze_method_control_flow(method_name: str) -> str:
        """Analyze control-flow constructs inside a method/function."""
        return _analyze_method_control_flow(method_name)

    def find_class_location(class_name: str) -> str:
        """Find the location of a class in the opened project."""
        return _find_class_loc(class_name)

    def identify_class_code(class_name: str) -> str:
        """Return class code for a class in the opened project."""
        return _identify_class(class_name)

    def get_imports(file_name: str = "") -> str:
        """Return imports from the current file in the opened project."""
        effective_file = file_name or (Path(context.buggy_file_path).name if context.buggy_file_path else "")
        return _get_imports(effective_file)

    def search_similar_fix_examples(
        query: str,
        n: int = 3,
        threshold: float | str | None = 0.6,
    ) -> str:
        """Retrieve similar historical fix examples from the configured repair database."""
        payload = {
            "query": query or context.buggy_code or "",
            "n": int(n or 3),
            "threshold": _parse_retrieval_threshold(threshold),
        }
        if not payload["query"]:
            return _json_dump([{"message": "No query was provided for similar-fix retrieval."}])
        try:
            ensure_reasoning_rag_server(RAG_SEARCH_URL)
            response = requests.post(RAG_SEARCH_URL, json=payload, timeout=120)
            response.raise_for_status()
            data: Any = response.json()
        except Exception as exc:
            data = [{"message": f"Example search failed: {exc}"}]
        return _truncate(_json_dump(data))

    model = ChatOpenAI(model=model_name, temperature=temperature)
    return create_agent(
        model=model,
        tools=[
            inspect_buggy_code_ast,
            read_buggy_source,
            open_project,
            identify_variable_usage,
            trace_method_usage,
            find_method_in_file,
            analyze_method_details,
            analyze_method_control_flow,
            find_class_location,
            identify_class_code,
            get_imports,
            search_similar_fix_examples,
            close_project,
        ],
        system_prompt=SYSTEM_PROMPT,
    )


class Reasoning:
    """Analyze buggy code and produce repair suggestions."""

    def __init__(
        self,
        *,
        program_id: str | None = None,
        buggy_code: str | None = None,
        buggy_file_path: str | os.PathLike[str] | None = None,
        language: str | None = None,
        fault_line_number: int | None = None,
        trigger_tests: str | None = None,
        error_message: str | None = None,
        additional_context: str | None = None,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
        test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
        pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    ) -> None:
        self.program_id = program_id
        self.buggy_code = buggy_code
        self.buggy_file_path = str(Path(buggy_file_path).expanduser()) if buggy_file_path else None
        self.language = language
        self.fault_line_number = fault_line_number
        self.trigger_tests = trigger_tests
        self.error_message = error_message
        self.additional_context = additional_context
        self.model_name = model_name
        self.temperature = temperature
        self.dataset_root = str(Path(dataset_root).expanduser())
        self.test_root = str(Path(test_root).expanduser())
        self.pricing = pricing

    def build_agent(self) -> Any:
        return build_reasoning_agent(
            program_id=self.program_id,
            buggy_code=self.buggy_code,
            buggy_file_path=self.buggy_file_path,
            language=self.language,
            fault_line_number=self.fault_line_number,
            trigger_tests=self.trigger_tests,
            error_message=self.error_message,
            additional_context=self.additional_context,
            model_name=self.model_name,
            temperature=self.temperature,
            dataset_root=self.dataset_root,
            test_root=self.test_root,
        )

    def generate(self) -> ReasoningResult:
        global _LAST_REASONING_USAGE
        _LAST_REASONING_USAGE = TokenUsage()
        raw_response = analyze_bug_reasoning(
            program_id=self.program_id,
            buggy_code=self.buggy_code,
            buggy_file_path=self.buggy_file_path,
            language=self.language,
            fault_line_number=self.fault_line_number,
            trigger_tests=self.trigger_tests,
            error_message=self.error_message,
            additional_context=self.additional_context,
            model_name=self.model_name,
            temperature=self.temperature,
            dataset_root=self.dataset_root,
            test_root=self.test_root,
        )
        usage = _LAST_REASONING_USAGE
        sections = _parse_reasoning_response(raw_response)
        estimated_cost_usd = self.pricing.estimate_cost_usd(usage) if self.pricing else None
        context = _build_reasoning_context(
            program_id=self.program_id,
            buggy_code=self.buggy_code,
            buggy_file_path=self.buggy_file_path,
            language=self.language,
            fault_line_number=self.fault_line_number,
            trigger_tests=self.trigger_tests,
            error_message=self.error_message,
            additional_context=self.additional_context,
            dataset_root=self.dataset_root,
            test_root=self.test_root,
        )

        return ReasoningResult(
            repair_suggestion=sections.repair_suggestion or raw_response,
            model_name=self.model_name,
            temperature=self.temperature,
            usage=usage,
            estimated_cost_usd=estimated_cost_usd,
            program_id=context.program_id,
            buggy_file_path=context.buggy_file_path,
            language=context.language,
            fault_line_number=context.fault_line_number,
            root_cause=sections.root_cause,
            similar_fix_examples=sections.similar_fix_examples,
            raw_response=raw_response,
        )


def analyze_bug_reasoning(
    *,
    program_id: str | None = None,
    buggy_code: str | None = None,
    buggy_file_path: str | os.PathLike[str] | None = None,
    language: str | None = None,
    fault_line_number: int | None = None,
    trigger_tests: str | None = None,
    error_message: str | None = None,
    additional_context: str | None = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
) -> str:
    global _LAST_REASONING_USAGE
    raw_response, usage = analyze_bug_reasoning_with_usage(
        program_id=program_id,
        buggy_code=buggy_code,
        buggy_file_path=buggy_file_path,
        language=language,
        fault_line_number=fault_line_number,
        trigger_tests=trigger_tests,
        error_message=error_message,
        additional_context=additional_context,
        model_name=model_name,
        temperature=temperature,
        dataset_root=dataset_root,
        test_root=test_root,
    )
    _LAST_REASONING_USAGE = usage
    return raw_response


def analyze_bug_reasoning_with_usage(
    *,
    program_id: str | None = None,
    buggy_code: str | None = None,
    buggy_file_path: str | os.PathLike[str] | None = None,
    language: str | None = None,
    fault_line_number: int | None = None,
    trigger_tests: str | None = None,
    error_message: str | None = None,
    additional_context: str | None = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
) -> tuple[str, TokenUsage]:
    context = _build_reasoning_context(
        program_id=program_id,
        buggy_code=buggy_code,
        buggy_file_path=buggy_file_path,
        language=language,
        fault_line_number=fault_line_number,
        trigger_tests=trigger_tests,
        error_message=error_message,
        additional_context=additional_context,
        dataset_root=dataset_root,
        test_root=test_root,
    )
    agent = build_reasoning_agent(
        program_id=program_id,
        buggy_code=buggy_code,
        buggy_file_path=buggy_file_path,
        language=language,
        fault_line_number=fault_line_number,
        trigger_tests=trigger_tests,
        error_message=error_message,
        additional_context=additional_context,
        model_name=model_name,
        temperature=temperature,
        dataset_root=dataset_root,
        test_root=test_root,
    )
    prompt = build_reasoning_prompt(
        program_id=context.program_id,
        buggy_code=context.buggy_code,
        buggy_file_path=context.buggy_file_path,
        language=context.language,
        fault_line_number=context.fault_line_number,
        trigger_tests=context.trigger_tests,
        error_message=context.error_message,
        additional_context=context.additional_context,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
    )
    return _extract_text_from_agent_result(result), _extract_usage_from_agent_result(result)


def generate_repair_suggestion_result(
    *,
    program_id: str | None = None,
    buggy_code: str | None = None,
    buggy_file_path: str | os.PathLike[str] | None = None,
    language: str | None = None,
    fault_line_number: int | None = None,
    trigger_tests: str | None = None,
    error_message: str | None = None,
    additional_context: str | None = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
) -> ReasoningResult:
    module = Reasoning(
        program_id=program_id,
        buggy_code=buggy_code,
        buggy_file_path=buggy_file_path,
        language=language,
        fault_line_number=fault_line_number,
        trigger_tests=trigger_tests,
        error_message=error_message,
        additional_context=additional_context,
        model_name=model_name,
        temperature=temperature,
        dataset_root=dataset_root,
        test_root=test_root,
        pricing=pricing,
    )
    return module.generate()


def generate_repair_suggestion(
    *,
    program_id: str | None = None,
    buggy_code: str | None = None,
    buggy_file_path: str | os.PathLike[str] | None = None,
    language: str | None = None,
    fault_line_number: int | None = None,
    trigger_tests: str | None = None,
    error_message: str | None = None,
    additional_context: str | None = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    test_root: str | os.PathLike[str] = DEFAULT_TEST_ROOT,
) -> str:
    result = generate_repair_suggestion_result(
        program_id=program_id,
        buggy_code=buggy_code,
        buggy_file_path=buggy_file_path,
        language=language,
        fault_line_number=fault_line_number,
        trigger_tests=trigger_tests,
        error_message=error_message,
        additional_context=additional_context,
        model_name=model_name,
        temperature=temperature,
        dataset_root=dataset_root,
        test_root=test_root,
    )
    return result.repair_suggestion


def _parse_reasoning_response(raw_response: str) -> _ReasoningSections:
    sections = _extract_labeled_sections(raw_response)
    return _ReasoningSections(
        root_cause=sections.get("Root Cause", ""),
        similar_fix_examples=sections.get("Similar Fix Examples", ""),
        repair_suggestion=sections.get("Repair Suggestion", raw_response).strip(),
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
    return text.strip().strip("*").strip()


def _extract_text_from_agent_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            message = messages[-1]
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        text = block.get("text") or block.get("content")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    return "\n".join(parts)

    return _json_dump(result)


def _extract_usage_from_agent_result(result: Any) -> TokenUsage:
    if not isinstance(result, dict):
        return TokenUsage()

    messages = result.get("messages") or []
    input_tokens = 0
    output_tokens = 0
    saw_input = False
    saw_output = False

    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            response_metadata = getattr(message, "response_metadata", None) or {}
            token_usage = response_metadata.get("token_usage") or {}
            usage = {
                "input_tokens": token_usage.get("prompt_tokens"),
                "output_tokens": token_usage.get("completion_tokens"),
            }

        current_input = _coerce_int(usage.get("input_tokens"))
        current_output = _coerce_int(usage.get("output_tokens"))
        if current_input is not None:
            input_tokens += current_input
            saw_input = True
        if current_output is not None:
            output_tokens += current_output
            saw_output = True

    return TokenUsage(
        input_tokens=input_tokens if saw_input else None,
        output_tokens=output_tokens if saw_output else None,
    )


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _read_optional_cli_file(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze buggy code and generate repair suggestions.")
    parser.add_argument("--program-id", default=None, help="Optional ConDefects program id.")
    parser.add_argument("--buggy-code-file", default=None, help="Optional file containing buggy code.")
    parser.add_argument("--buggy-file-path", default=None, help="Path to the original buggy source file.")
    parser.add_argument("--language", default=None, help="Optional source language hint.")
    parser.add_argument("--fault-line-number", type=int, default=None, help="Optional suspicious/fault line number.")
    parser.add_argument("--trigger-tests-file", default=None, help="Optional file containing trigger tests or expected behavior.")
    parser.add_argument("--error-message-file", default=None, help="Optional file containing observed error output.")
    parser.add_argument("--additional-context-file", default=None, help="Optional file containing extra task context.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="ConDefects dataset root.")
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT), help="ConDefects test root.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Agent model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Model temperature.")
    args = parser.parse_args(argv)

    output = generate_repair_suggestion(
        program_id=args.program_id,
        buggy_code=_read_optional_cli_file(args.buggy_code_file),
        buggy_file_path=args.buggy_file_path,
        language=args.language,
        fault_line_number=args.fault_line_number,
        trigger_tests=_read_optional_cli_file(args.trigger_tests_file),
        error_message=_read_optional_cli_file(args.error_message_file),
        additional_context=_read_optional_cli_file(args.additional_context_file),
        model_name=args.model,
        temperature=args.temperature,
        dataset_root=args.dataset_root,
        test_root=args.test_root,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
