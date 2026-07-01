from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from viberepair.langchain_support import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE as SHARED_DEFAULT_TEMPERATURE,
    ensure_openai_api_key,
    require_chat_openai,
    require_create_agent,
)

from .prompt import SYSTEM_PROMPT, build_analysis_prompt

DEFAULT_MODEL = DEFAULT_MODEL_NAME
DEFAULT_TEMPERATURE = SHARED_DEFAULT_TEMPERATURE
MAX_TOOL_TEXT_CHARS = 24000
MAX_READ_LINES = 260
MAX_DIRECTORY_ENTRIES = 200
MAX_SEARCH_MATCHES = 80
MAX_REFERENCE_FILES = 60
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".csv",
    ".diff",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rb",
    ".rst",
    ".scala",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TASK_PATH_RE = re.compile(r"(tasks(?:_excluded)?/[A-Za-z0-9._-]+)")


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
_LAST_ANALYSIS_USAGE = TokenUsage()
FAILED_STATUS_VALUES = {
    "broken",
    "error",
    "errored",
    "fail",
    "failed",
    "failure",
    "other",
}
GENERIC_TEST_FILE_PATTERNS = [
    "test*.py",
    "*_test.py",
    "test_*.py",
    "conftest.py",
    "*.spec.js",
    "*.test.js",
    "*.spec.ts",
    "*.test.ts",
    "*.spec.jsx",
    "*.test.jsx",
    "*.spec.tsx",
    "*.test.tsx",
    "*Test.java",
    "*Tests.java",
    "*IT.java",
    "*Spec.java",
    "*_test.go",
    "test_*.rb",
    "*_spec.rb",
]
GENERIC_REFERENCE_FILE_PATTERNS = [
    "README*",
    "readme*",
    "instruction*",
    "instructions*",
    "task.*",
    "pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Makefile",
    "CMakeLists.txt",
]
LOG_FILE_NAME_HINTS = {
    "console",
    "output",
    "pytest",
    "report",
    "results",
    "stderr",
    "stdout",
    "surefire",
    "test",
    "trace",
    "verifier",
}
EXACT_REFERENCE_FILE_NAMES = {
    "conftest.py",
    "instruction.md",
    "instructions.md",
    "makefile",
    "package.json",
    "pom.xml",
    "pyproject.toml",
    "pytest.ini",
    "readme",
    "readme.md",
    "task.toml",
    "task.yaml",
    "task.yml",
    "test.sh",
    "test_outputs.py",
    "tox.ini",
}


@dataclass(slots=True)
class VerificationContext:
    input_path: Path
    verification_root: Path
    workspace_root: Path


@dataclass(slots=True, frozen=True)
class TestFailureSpecificationsResult:
    verification_result_path: str
    workspace_root: str | None
    test_failure_specifications: str
    model_name: str
    temperature: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    estimated_cost_usd: float | None = None


def _require_langchain() -> tuple[Any, Any]:
    return require_create_agent(), require_chat_openai()


def _resolve_existing_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return path


def _build_context(
    verification_result_path: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
) -> VerificationContext:
    input_path = _resolve_existing_path(verification_result_path)
    resolved_workspace_root = _resolve_existing_path(workspace_root or Path.cwd())
    verification_root = input_path if input_path.is_dir() else input_path.parent
    return VerificationContext(
        input_path=input_path,
        verification_root=verification_root,
        workspace_root=resolved_workspace_root,
    )


def _truncate(text: str, limit: int = MAX_TOOL_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[truncated {omitted} characters]"


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _is_probably_text_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    return True


def _safe_read_text(path: Path, max_chars: int = MAX_TOOL_TEXT_CHARS) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        text = handle.read(max_chars + 1)
    if len(text) > max_chars:
        return _truncate(text, max_chars)
    return text


def _format_file_excerpt(path: Path, start_line: int = 1, end_line: int = MAX_READ_LINES) -> str:
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    if end_line - start_line + 1 > MAX_READ_LINES:
        end_line = start_line + MAX_READ_LINES - 1

    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle, start=1):
            if idx < start_line:
                continue
            if idx > end_line:
                break
            lines.append(f"{idx:04d}: {line.rstrip()}")

    if not lines:
        return f"{path}\n<no content in requested range>"

    return f"{path}\n" + "\n".join(lines)


def _iter_files(root: Path, limit: int = MAX_DIRECTORY_ENTRIES) -> list[Path]:
    if root.is_file():
        return [root]

    files: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            files.append(candidate)
        if len(files) >= limit:
            break
    return files


def _artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if "ctrf" in name and path.suffix.lower() == ".json":
        return "ctrf_report"
    if _is_probable_junit_xml(path):
        return "junit_xml_report"
    if name in {"pytest_output.txt", "pytest-output.txt", "test_output.log", "test-stdout.txt", "test-stdout.log"}:
        return "test_log"
    if _is_probable_test_log(path):
        return "test_log"
    if name == "reward.txt":
        return "reward"
    if name == "failed_reasons.txt":
        return "failure_note"
    if name == "score.json":
        return "score_json"
    if path.suffix.lower() == ".json":
        return "json_artifact"
    if path.suffix.lower() in {".log", ".txt"}:
        return "text_artifact"
    if path.suffix.lower() == ".py":
        return "python_artifact"
    return "artifact"


def _artifact_priority(path: Path) -> tuple[int, str]:
    kind = _artifact_kind(path)
    priority_map = {
        "ctrf_report": 0,
        "junit_xml_report": 1,
        "test_log": 2,
        "failure_note": 3,
        "score_json": 4,
        "reward": 5,
        "json_artifact": 6,
        "text_artifact": 7,
        "python_artifact": 8,
        "artifact": 9,
    }
    return (priority_map.get(kind, 99), path.name.lower())


def _collect_verification_artifacts(context: VerificationContext) -> list[Path]:
    artifacts = _iter_files(context.verification_root, limit=MAX_DIRECTORY_ENTRIES)
    if context.input_path.is_file() and context.input_path not in artifacts:
        artifacts.insert(0, context.input_path)
    return sorted(dict.fromkeys(artifacts), key=_artifact_priority)


def _list_directory_entries(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return [
            {
                "name": path.name,
                "path": str(path),
                "type": "file",
                "size_bytes": path.stat().st_size,
            }
        ]

    entries: list[dict[str, Any]] = []
    for candidate in sorted(path.iterdir())[:MAX_DIRECTORY_ENTRIES]:
        item_type = "dir" if candidate.is_dir() else "file"
        size_bytes = candidate.stat().st_size if candidate.is_file() else None
        entries.append(
            {
                "name": candidate.name,
                "path": str(candidate),
                "type": item_type,
                "size_bytes": size_bytes,
            }
        )
    return entries


def _is_likely_test_file(path: Path) -> bool:
    path_str = path.as_posix()
    name_lower = path.name.lower()
    parent_parts = {part.lower() for part in path.parts}

    if any(fnmatch.fnmatch(name_lower, pattern.lower()) for pattern in GENERIC_TEST_FILE_PATTERNS):
        return True
    if "tests" in parent_parts or "spec" in parent_parts or "specs" in parent_parts:
        if path.suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".sh"}:
            return True
    return "::" in path_str or "test" in name_lower


def _is_likely_reference_file(path: Path) -> bool:
    name_lower = path.name.lower()
    stem_lower = path.stem.lower()
    if name_lower in EXACT_REFERENCE_FILE_NAMES or stem_lower in EXACT_REFERENCE_FILE_NAMES:
        return True
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in GENERIC_REFERENCE_FILE_PATTERNS):
        return True
    return False


def _is_probable_test_log(path: Path) -> bool:
    name_lower = path.name.lower()
    if path.suffix.lower() not in {".log", ".txt", ".out", ".err"}:
        return False
    return any(token in name_lower for token in LOG_FILE_NAME_HINTS)


def _is_probable_junit_xml(path: Path) -> bool:
    if path.suffix.lower() != ".xml":
        return False
    name_lower = path.name.lower()
    return (
        name_lower.startswith("test-")
        or "junit" in name_lower
        or "surefire" in name_lower
        or "test-result" in name_lower
        or "testsuite" in name_lower
    )


def _identifier_tokens(identifier: str) -> list[str]:
    tokens = re.split(r"[^A-Za-z0-9_]+", identifier)
    return [token.lower() for token in tokens if len(token) >= 3]


def _collect_pattern_files(root: Path, patterns: list[str], limit: int) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        try:
            iterator = root.rglob(pattern)
        except OSError:
            continue
        for path in iterator:
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            matches.append(path)
            if len(matches) >= limit:
                return matches
    return matches


def _extract_task_path_strings(payload: Any) -> set[str]:
    found: set[str] = set()
    hint_tokens = (
        "test",
        "spec",
        "readme",
        "instruction",
        "task.",
        "pyproject.toml",
        "package.json",
        "pom.xml",
        "build.gradle",
        "tox.ini",
        "conftest.py",
    )

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
            return
        if isinstance(node, list):
            for value in node:
                visit(value)
            return
        if isinstance(node, str):
            for match in TASK_PATH_RE.findall(node):
                found.add(match)
            lowered = node.lower().strip()
            if "/" in lowered and any(token in lowered for token in hint_tokens):
                found.add(node.strip())

    visit(payload)
    return found


def _find_ancestor_files(start_path: Path, target_name: str, max_depth: int = 8) -> list[Path]:
    current = start_path if start_path.is_dir() else start_path.parent
    matches: list[Path] = []
    for _ in range(max_depth):
        candidate = current / target_name
        if candidate.exists():
            matches.append(candidate)
        if current.parent == current:
            break
        current = current.parent
    return matches


def _candidate_task_files_from_result_json(context: VerificationContext) -> list[Path]:
    results = _find_ancestor_files(context.input_path, "result.json")
    matches: list[Path] = []
    for result_json in results:
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        for task_path_str in _extract_task_path_strings(payload):
            raw_path = Path(task_path_str).expanduser()
            candidate_roots = []
            if raw_path.is_absolute():
                candidate_roots.append(raw_path)
            else:
                candidate_roots.append((context.workspace_root / raw_path).resolve())
                candidate_roots.append((result_json.parent / raw_path).resolve())

            for task_path in candidate_roots:
                if not task_path.exists():
                    continue
                if task_path.is_file():
                    matches.append(task_path)
                    continue
                for relative in (
                    "tests/test_outputs.py",
                    "tests/test.sh",
                    "instruction.md",
                    "instructions.md",
                    "task.toml",
                    "README.md",
                    "pyproject.toml",
                    "package.json",
                    "pom.xml",
                ):
                    candidate = task_path / relative
                    if candidate.exists():
                        matches.append(candidate)
    return matches


def _find_test_output_files(workspace_root: Path) -> list[Path]:
    return sorted(
        {
            path
            for path in _collect_pattern_files(workspace_root, GENERIC_TEST_FILE_PATTERNS, limit=2000)
            if path.is_file()
        },
        key=str,
    )


def _extract_test_function_name(test_identifier: str) -> str:
    match = re.search(r"(test_[A-Za-z0-9_]+)", test_identifier)
    if not match:
        raise ValueError(f"Could not extract a pytest test name from: {test_identifier}")
    return match.group(1)


def _score_test_file_for_context(context: VerificationContext, test_file: Path, failed_test_names: set[str]) -> int:
    score = 0
    path_text = context.input_path.as_posix().lower()
    file_text = test_file.as_posix().lower()
    parent_parts = {part.lower() for part in test_file.parts}

    if _is_likely_test_file(test_file):
        score += 120
    if _is_likely_reference_file(test_file):
        score += 40
    if "tests" in parent_parts or "spec" in parent_parts or "specs" in parent_parts:
        score += 30

    bundle_tokens = _identifier_tokens(context.input_path.stem) + _identifier_tokens(context.verification_root.name)
    for token in bundle_tokens:
        if token in file_text:
            score += 15

    if failed_test_names:
        try:
            text = _safe_read_text(test_file, max_chars=120000).lower()
        except OSError:
            text = ""
        for failed_name in failed_test_names:
            failed_name_lower = failed_name.lower()
            if failed_name_lower in file_text:
                score += 35
            if f"def {failed_name_lower}(" in text or failed_name_lower in text:
                score += 80
    return score


def _find_relevant_reference_files(
    context: VerificationContext,
    failed_test_identifiers: list[str] | None = None,
    extra_queries: list[str] | None = None,
) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path.exists() and path.is_file() and path not in seen:
            seen.add(path)
            matches.append(path)

    near_dirs: list[Path] = []
    current = context.verification_root if context.verification_root.is_dir() else context.verification_root.parent
    for _ in range(4):
        near_dirs.append(current)
        if current.parent == current:
            break
        current = current.parent

    for base_dir in near_dirs:
        for path in _collect_pattern_files(base_dir, GENERIC_TEST_FILE_PATTERNS + GENERIC_REFERENCE_FILE_PATTERNS, limit=200):
            add(path)

    for path in _candidate_task_files_from_result_json(context):
        add(path)

    for path in _find_test_output_files(context.workspace_root):
        add(path)

    for path in _collect_pattern_files(context.workspace_root, GENERIC_REFERENCE_FILE_PATTERNS, limit=400):
        add(path)

    failed_test_names: set[str] = set()
    for identifier in failed_test_identifiers or []:
        try:
            failed_test_names.add(_extract_test_function_name(identifier))
        except ValueError:
            failed_test_names.update(_identifier_tokens(identifier))

    for query in extra_queries or []:
        failed_test_names.update(_identifier_tokens(query))

    scored = sorted(
        ((-_score_test_file_for_context(context, file_path, failed_test_names), file_path) for file_path in matches),
        key=lambda item: (item[0], str(item[1])),
    )

    ranked: list[Path] = []
    for neg_score, file_path in scored:
        if -neg_score <= 0:
            continue
        ranked.append(file_path)
        if len(ranked) >= MAX_REFERENCE_FILES:
            break

    return ranked


def _parse_ctrf_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    results = payload.get("results", {})
    summary = results.get("summary", {})
    tests = results.get("tests", [])
    failed_tests: list[dict[str, Any]] = []

    for test in tests:
        status = str(test.get("status", "")).lower()
        if status not in {"failed", "error", "broken", "other"}:
            continue

        record: dict[str, Any] = {
            "name": test.get("name") or test.get("test") or "<unknown>",
            "status": status,
        }
        for key in ("message", "trace", "duration", "suite", "filePath", "line", "rawStatus"):
            value = test.get(key)
            if value not in (None, "", [], {}):
                record[key] = value

        if "message" not in record and "trace" not in record:
            record["raw"] = {
                key: value
                for key, value in test.items()
                if key not in {"attachments", "screenshot", "screenshots"}
            }

        failed_tests.append(record)

    return {
        "source": str(path),
        "summary": summary,
        "failed_tests": failed_tests,
    }


def _parse_junit_xml_report(path: Path) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    failed_tests: list[dict[str, Any]] = []

    for testcase in root.findall(".//testcase"):
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")
        if skipped is not None or (failure is None and error is None):
            continue

        failure_node = failure if failure is not None else error
        test_name = testcase.attrib.get("name") or "<unknown>"
        class_name = testcase.attrib.get("classname")
        full_name = f"{class_name}.{test_name}" if class_name else test_name
        failed_tests.append(
            {
                "name": full_name,
                "status": "failed" if failure is not None else "error",
                "message": failure_node.attrib.get("message") if failure_node is not None else None,
                "trace": (failure_node.text or "").strip() if failure_node is not None and failure_node.text else None,
                "duration": testcase.attrib.get("time"),
                "file": testcase.attrib.get("file"),
                "line": testcase.attrib.get("line"),
            }
        )

    summary = {
        "tests": root.attrib.get("tests"),
        "failures": root.attrib.get("failures"),
        "errors": root.attrib.get("errors"),
        "skipped": root.attrib.get("skipped"),
    }
    return {
        "source": str(path),
        "summary": {key: value for key, value in summary.items() if value not in (None, "")},
        "failed_tests": failed_tests,
    }


def _extract_failed_tests_from_json_node(node: Any, location: str = "root") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if isinstance(node, dict):
        name = None
        for key in ("name", "test", "nodeid", "title", "fullName", "displayName", "id"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break

        status = None
        for key in ("status", "outcome", "result", "state", "verdict"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                status = value.strip().lower()
                break

        if name and status in FAILED_STATUS_VALUES:
            message = None
            for key in ("message", "error", "failure", "trace", "traceback", "stack", "stderr"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    message = value.strip()
                    break

            records.append(
                {
                    "name": name,
                    "status": status,
                    "message": message,
                    "location": location,
                }
            )

        for key, value in node.items():
            records.extend(_extract_failed_tests_from_json_node(value, f"{location}.{key}"))

    elif isinstance(node, list):
        for index, value in enumerate(node):
            records.extend(_extract_failed_tests_from_json_node(value, f"{location}[{index}]"))

    return records


def _parse_generic_json_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    failed_tests = _extract_failed_tests_from_json_node(payload)
    return {
        "source": str(path),
        "failed_tests": failed_tests,
    }


def _extract_failed_identifiers_from_log(lines: list[str]) -> tuple[list[str], list[str]]:
    summary_lines: list[str] = []
    identifiers: list[str] = []

    patterns = [
        re.compile(r"^(?:FAILED|ERROR)\s+(.+?)(?:\s+-\s+.*)?$"),
        re.compile(r"^(?:FAIL|ERROR):\s+(.+)$"),
        re.compile(r"^(.+?)\s+<<<\s+(?:FAILURE|ERROR)!$"),
        re.compile(r"^\s*●\s+(.+)$"),
    ]

    for raw_line in lines:
        line = raw_line.rstrip()
        matched = False
        for pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue
            summary_lines.append(line.strip())
            identifiers.append(match.group(1).strip())
            matched = True
            break
        if matched:
            continue
        if "AssertionError" in line or "Traceback (most recent call last)" in line:
            summary_lines.append(line.strip())

    return summary_lines, identifiers


def _parse_test_log(path: Path) -> dict[str, Any]:
    text = _safe_read_text(path, max_chars=80000)
    lines = text.splitlines()
    summary_lines, failed_identifiers = _extract_failed_identifiers_from_log(lines)

    return {
        "source": str(path),
        "failed_summary_lines": summary_lines,
        "failed_test_identifiers": failed_identifiers,
        "tail_excerpt": "\n".join(lines[-80:]),
    }


def _build_failed_test_report(context: VerificationContext) -> dict[str, Any]:
    ctrf_reports: list[dict[str, Any]] = []
    junit_reports: list[dict[str, Any]] = []
    generic_json_reports: list[dict[str, Any]] = []
    log_reports: list[dict[str, Any]] = []
    failed_identifiers: list[str] = []

    for artifact in _collect_verification_artifacts(context):
        kind = _artifact_kind(artifact)
        if kind == "ctrf_report":
            try:
                report = _parse_ctrf_report(artifact)
            except (OSError, json.JSONDecodeError) as exc:
                report = {"source": str(artifact), "error": str(exc)}
            ctrf_reports.append(report)
            for failed_test in report.get("failed_tests", []):
                name = failed_test.get("name")
                if name:
                    failed_identifiers.append(str(name))
        elif kind == "junit_xml_report":
            try:
                report = _parse_junit_xml_report(artifact)
            except (OSError, ET.ParseError) as exc:
                report = {"source": str(artifact), "error": str(exc)}
            junit_reports.append(report)
            for failed_test in report.get("failed_tests", []):
                name = failed_test.get("name")
                if name:
                    failed_identifiers.append(str(name))
        elif kind == "test_log":
            try:
                report = _parse_test_log(artifact)
            except OSError as exc:
                report = {"source": str(artifact), "error": str(exc)}
            log_reports.append(report)
            failed_identifiers.extend(report.get("failed_test_identifiers", []))
        elif kind == "json_artifact":
            try:
                report = _parse_generic_json_report(artifact)
            except (OSError, json.JSONDecodeError) as exc:
                report = {"source": str(artifact), "error": str(exc)}
            if report.get("failed_tests"):
                generic_json_reports.append(report)
                for failed_test in report.get("failed_tests", []):
                    name = failed_test.get("name")
                    if name:
                        failed_identifiers.append(str(name))

    return {
        "verification_root": str(context.verification_root),
        "input_path": str(context.input_path),
        "ctrf_reports": ctrf_reports,
        "junit_reports": junit_reports,
        "generic_json_reports": generic_json_reports,
        "log_reports": log_reports,
        "all_failed_test_identifiers": sorted(dict.fromkeys(failed_identifiers)),
        "candidate_reference_files": [str(path) for path in _find_relevant_reference_files(context, failed_identifiers)],
    }


def _collect_function_matches(py_file: Path, target_name: str) -> list[dict[str, Any]]:
    source = py_file.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(py_file))
    source_lines = source.splitlines()
    matches: list[dict[str, Any]] = []

    class_name_stack: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            class_name_stack.append(node.name)
            for child in node.body:
                visit(child)
            class_name_stack.pop()
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == target_name:
                decorator_start = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
                start_line = decorator_start
                end_line = node.end_lineno or node.lineno
                snippet = "\n".join(
                    f"{idx:04d}: {source_lines[idx - 1]}"
                    for idx in range(start_line, min(end_line, start_line + 159) + 1)
                )
                matches.append(
                    {
                        "file": str(py_file),
                        "function_name": node.name,
                        "class_name": class_name_stack[-1] if class_name_stack else None,
                        "line_start": start_line,
                        "line_end": end_line,
                        "docstring": ast.get_docstring(node),
                        "source_excerpt": snippet,
                    }
                )
            for child in node.body:
                visit(child)
            return

        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return matches


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



def _normalize_optional_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _infer_python_shape(code: str) -> dict[str, Any] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    body = list(tree.body)
    functions = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node for node in body if isinstance(node, ast.ClassDef)]
    imports = (ast.Import, ast.ImportFrom)
    executable_nodes = [
        node
        for node in body
        if not isinstance(node, (*imports, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    has_entry_call = False
    if len(functions) == 1 and executable_nodes:
        function_name = functions[0].name
        has_entry_call = all(
            _is_python_entry_call_node(node, function_name) for node in executable_nodes
        )

    if functions and not executable_nodes and not classes:
        shape = "python_function"
    elif len(functions) == 1 and has_entry_call and not classes:
        shape = "python_function_wrapper_with_entry_call"
    elif executable_nodes:
        shape = "python_top_level_script"
    elif classes and not functions:
        shape = "python_class"
    else:
        shape = "python_module_declarations"

    return {
        "language": "Python",
        "shape": shape,
        "function_names": [node.name for node in functions],
        "function_signatures": [_python_function_signature(node) for node in functions],
        "class_names": [node.name for node in classes],
        "has_top_level_entry_call": has_entry_call,
        "top_level_executable_statement_count": len(executable_nodes),
    }


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


def _is_python_entry_call_node(node: ast.stmt, function_name: str) -> bool:
    if isinstance(node, ast.Expr) and _is_call_to_name(node.value, function_name):
        return True
    if isinstance(node, ast.If):
        return (
            _is_main_name_check_node(node.test)
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Expr)
            and _is_call_to_name(node.body[0].value, function_name)
        )
    return False


def _is_call_to_name(value: ast.AST, function_name: str) -> bool:
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == function_name


def _is_main_name_check_node(test: ast.AST) -> bool:
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


JAVA_METHOD_RE = re.compile(
    r"(?P<header>\s*(?P<prefix>(?:(?:public|private|protected|static|final|synchronized|abstract|native|strictfp)\s+)*)"
    r"(?P<return_type>[\w<>\[\], ?]+?)\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\((?P<params>[^;{}]*)\)"
    r"(?P<throws>\s+throws\s+[\w.,\s]+)?\s*)\{",
    re.MULTILINE,
)
JAVA_CLASS_RE = re.compile(r"\b(?:public\s+)?(?:final\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)\b")


def _infer_java_shape(code: str) -> dict[str, Any] | None:
    class_match = JAVA_CLASS_RE.search(code)
    method_matches = list(JAVA_METHOD_RE.finditer(code))
    if not class_match and not method_matches:
        return None

    if class_match:
        shape = "java_class"
    elif len(method_matches) == 1:
        shape = "java_method"
    else:
        shape = "java_methods"

    method_records = [_java_method_record(match) for match in method_matches]
    return {
        "language": "Java",
        "shape": shape,
        "class_names": [class_match.group("name")] if class_match else [],
        "method_names": [record["name"] for record in method_records],
        "method_signatures": [record["signature"] for record in method_records],
        "method_headers": [record["header"] for record in method_records],
        "entry_points": [record["signature"] for record in method_records if record["is_main_entry_point"]],
    }


def _java_method_record(match: re.Match[str]) -> dict[str, Any]:
    prefix = _normalize_signature_text(match.group("prefix") or "")
    return_type = _normalize_signature_text(match.group("return_type") or "")
    name = match.group("name")
    params = _normalize_signature_text(match.group("params") or "")
    throws = _normalize_signature_text(match.group("throws") or "")
    header = _normalize_signature_text(match.group("header") or "")
    signature = f"{return_type} {name}({params})".strip()
    if throws:
        signature = f"{signature} {throws}"
    return {
        "name": name,
        "prefix": prefix,
        "return_type": return_type,
        "params": params,
        "throws": throws,
        "header": header,
        "signature": signature,
        "is_main_entry_point": _is_java_main_entry_point(
            name=name,
            prefix=prefix,
            return_type=return_type,
            params=params,
        ),
    }


def _is_java_main_entry_point(name: str, prefix: str, return_type: str, params: str) -> bool:
    normalized_prefix = f" {prefix} "
    normalized_params = params.replace(" ", "")
    return (
        name == "main"
        and return_type == "void"
        and " static " in normalized_prefix
        and "String[]args" in normalized_params
    )


def _normalize_signature_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _infer_code_shape(code: str | None) -> dict[str, Any]:
    if not code:
        return {"available": False, "language": None, "shape": "missing"}

    python_shape = _infer_python_shape(code)
    if python_shape is not None:
        return {"available": True, **python_shape}

    java_shape = _infer_java_shape(code)
    if java_shape is not None:
        return {"available": True, **java_shape}

    return {"available": True, "language": "unknown", "shape": "unknown"}


def _inspect_fix_code_structure_payload(
    buggy_code: str | None,
    fix_code: str | None,
) -> dict[str, Any]:
    buggy_shape = _infer_code_shape(buggy_code)
    fix_shape = _infer_code_shape(fix_code)
    evidence: list[str] = []
    constraints: list[str] = []
    likely_mismatch = False

    if not buggy_code or not fix_code:
        return {
            "available": False,
            "likely_structure_mismatch": False,
            "reason": "buggy_code and fix_code were not both provided",
            "buggy_shape": buggy_shape,
            "fix_shape": fix_shape,
            "evidence": evidence,
            "required_next_fix_constraints": constraints,
        }

    buggy_lang = buggy_shape.get("language")
    fix_lang = fix_shape.get("language")
    if buggy_lang and fix_lang and buggy_lang != fix_lang and fix_lang != "unknown":
        likely_mismatch = True
        evidence.append(f"Buggy snippet appears to be {buggy_lang}, but fix code appears to be {fix_lang}.")
        constraints.append("The next fix code must be written in the same language as the buggy snippet.")

    buggy_kind = buggy_shape.get("shape")
    fix_kind = fix_shape.get("shape")

    if buggy_kind == "python_top_level_script" and fix_kind in {"python_function", "python_function_wrapper_with_entry_call"}:
        likely_mismatch = True
        names = fix_shape.get("function_names") or []
        evidence.append(
            "Buggy snippet is executable top-level Python code, but fix code only defines "
            f"function(s) {names} without preserving top-level execution."
        )
        constraints.append(
            "The next fix must preserve top-level execution and the original stdin/stdout contract: do not wrap script logic in a new function or entry guard when the buggy code was top-level code."
        )
    elif buggy_kind == "python_function" and fix_kind in {"python_top_level_script", "python_function_wrapper_with_entry_call"}:
        likely_mismatch = True
        evidence.append(
            "Buggy snippet is a Python function replacement target, but fix code changes it into top-level executable code."
        )
        constraints.append("The next fix must preserve the original Python function name, signature, and return/print contract.")

    buggy_functions = buggy_shape.get("function_names") or []
    fix_functions = fix_shape.get("function_names") or []
    if buggy_kind == "python_function" and fix_kind == "python_function" and buggy_functions and fix_functions:
        if buggy_functions[0] != fix_functions[0]:
            likely_mismatch = True
            evidence.append(
                f"Buggy Python function is named {buggy_functions[0]!r}, but fix code defines {fix_functions[0]!r}."
            )
            constraints.append("The next fix must preserve the original Python function name and signature.")

    buggy_function_signatures = buggy_shape.get("function_signatures") or []
    fix_function_signatures = fix_shape.get("function_signatures") or []
    if buggy_kind == "python_function" and fix_kind == "python_function" and buggy_function_signatures and fix_function_signatures:
        if buggy_function_signatures[0] != fix_function_signatures[0]:
            likely_mismatch = True
            evidence.append(
                f"Buggy Python function signature is {buggy_function_signatures[0]!r}, but fix code defines {fix_function_signatures[0]!r}."
            )
            constraints.append("The next fix must preserve the original Python function signature and only change the faulty logic inside it.")

    if buggy_kind == "java_method" and fix_kind == "java_class":
        likely_mismatch = True
        evidence.append("Buggy snippet is a Java method replacement target, but fix code introduces a class wrapper.")
        constraints.append("The next fix must return the replacement Java method body/declaration shape expected at the original source range, not a new class wrapper.")
    elif buggy_kind == "java_class" and fix_kind == "java_method":
        likely_mismatch = True
        evidence.append("Buggy snippet is a Java class replacement target, but fix code only returns a method.")
        constraints.append("The next fix must preserve the original Java class declaration shape.")

    buggy_methods = buggy_shape.get("method_names") or []
    fix_methods = fix_shape.get("method_names") or []
    buggy_method_signatures = buggy_shape.get("method_signatures") or []
    fix_method_signatures = fix_shape.get("method_signatures") or []
    buggy_entry_points = buggy_shape.get("entry_points") or []
    fix_entry_points = fix_shape.get("entry_points") or []
    if buggy_kind == "java_method" and fix_kind == "java_method" and buggy_methods and fix_methods:
        if buggy_methods[0] != fix_methods[0]:
            likely_mismatch = True
            evidence.append(f"Buggy Java method is named {buggy_methods[0]!r}, but fix code defines {fix_methods[0]!r}.")
            constraints.append("The next fix must preserve the original Java method name, especially entry-point names such as main, and only change the faulty logic inside it.")
        if buggy_method_signatures and fix_method_signatures and buggy_method_signatures[0] != fix_method_signatures[0]:
            likely_mismatch = True
            evidence.append(
                f"Buggy Java method signature is {buggy_method_signatures[0]!r}, but fix code defines {fix_method_signatures[0]!r}."
            )
            constraints.append("The next fix must preserve the original Java method signature, including return type, parameters, modifiers, static entry-point requirements, and throws clause.")
    if buggy_entry_points and not fix_entry_points:
        likely_mismatch = True
        evidence.append(
            f"Buggy code exposes Java entry point(s) {buggy_entry_points}, but fix code does not preserve a matching public/static main entry point."
        )
        constraints.append("The next fix must preserve the Java main entry point exactly, for example public static void main(String[] args), so the benchmark runner executes the repaired logic.")

    if not constraints:
        constraints.append("No clear structural mismatch was detected; prioritize the concrete test failure evidence.")

    return {
        "available": True,
        "likely_structure_mismatch": likely_mismatch,
        "buggy_shape": buggy_shape,
        "fix_shape": fix_shape,
        "evidence": evidence,
        "required_next_fix_constraints": constraints,
    }

def build_feedback_agent(
    verification_result_path: str | os.PathLike[str],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    workspace_root: str | os.PathLike[str] | None = None,
    buggy_code: str | None = None,
    fix_code: str | None = None,
    fix_specification: str | None = None,
) -> Any:
    create_agent, ChatOpenAI = _require_langchain()
    context = _build_context(verification_result_path, workspace_root=workspace_root)
    normalized_buggy_code = _normalize_optional_code(buggy_code)
    normalized_fix_code = _normalize_optional_code(fix_code)
    normalized_fix_specification = _normalize_optional_code(fix_specification)

    ensure_openai_api_key()

    def inspect_fix_code_structure() -> str:
        """Compare buggy and fix code replacement shapes before interpreting test failures."""
        payload = _inspect_fix_code_structure_payload(
            buggy_code=normalized_buggy_code,
            fix_code=normalized_fix_code,
        )
        if normalized_fix_specification:
            payload["fix_specification_available"] = True
        return _truncate(_json_dump(payload))

    def inspect_verification_bundle() -> str:
        """Scan the verification path and report likely artifacts plus nearby reference files."""
        artifacts = _collect_verification_artifacts(context)
        manifest = [
            {
                "path": str(path),
                "kind": _artifact_kind(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        ]
        payload = {
            "input_path": str(context.input_path),
            "verification_root": str(context.verification_root),
            "workspace_root": str(context.workspace_root),
            "artifacts": manifest,
            "candidate_reference_files": [str(path) for path in _find_relevant_reference_files(context)],
        }
        return _truncate(_json_dump(payload))

    def get_failed_tests() -> str:
        """Parse structured reports and logs under the verification bundle into failure data."""
        payload = _build_failed_test_report(context)
        return _truncate(_json_dump(payload))

    def find_relevant_files(query: str = "") -> str:
        """Rank likely useful test, config, and instruction files for this verification bundle."""
        payload = {
            "query": query,
            "files": [
                str(path)
                for path in _find_relevant_reference_files(
                    context,
                    extra_queries=[query] if query else None,
                )
            ],
        }
        return _truncate(_json_dump(payload))

    def list_directory(path: str = "") -> str:
        """List files and directories under a path. If omitted, list the verification root."""
        target = _resolve_existing_path(path) if path else context.verification_root
        return _truncate(_json_dump(_list_directory_entries(target)))

    def read_file(path: str, start_line: int = 1, end_line: int = MAX_READ_LINES) -> str:
        """Read a text file with line numbers. Use this for logs, tests, scripts, JSON, and markdown files."""
        target = _resolve_existing_path(path)
        if not target.is_file():
            raise ValueError(f"Not a file: {target}")
        if not _is_probably_text_file(target):
            raise ValueError(f"File does not look like text: {target}")
        return _truncate(_format_file_excerpt(target, start_line=start_line, end_line=end_line))

    def search_workspace(pattern: str, file_glob: str = "*.py", base_path: str = "") -> str:
        """Case-insensitive substring search across workspace files. Returns path:line matches."""
        search_root = _resolve_existing_path(base_path) if base_path else context.workspace_root
        lowered_pattern = pattern.lower()
        matches: list[str] = []

        for file_path in sorted(search_root.rglob(file_glob)):
            if not file_path.is_file() or not _is_probably_text_file(file_path):
                continue
            try:
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        if lowered_pattern in line.lower():
                            matches.append(f"{file_path}:{line_no}: {line.rstrip()}")
                            if len(matches) >= MAX_SEARCH_MATCHES:
                                return _truncate("\n".join(matches))
            except OSError:
                continue

        if not matches:
            return f"No matches for pattern: {pattern}"

        return _truncate("\n".join(matches))

    def find_test_definitions(test_identifier: str) -> str:
        """Locate likely test definitions or source snippets for a failed test identifier."""
        candidate_files = _find_relevant_reference_files(context, failed_test_identifiers=[test_identifier])
        patterns = [token for token in _identifier_tokens(test_identifier) if token]
        raw_identifier = test_identifier.lower()
        python_matches: list[dict[str, Any]] = []
        text_matches: list[dict[str, Any]] = []

        try:
            target_name = _extract_test_function_name(test_identifier)
        except ValueError:
            target_name = ""

        for file_path in candidate_files:
            if file_path.suffix.lower() == ".py" and target_name:
                try:
                    python_matches.extend(_collect_function_matches(file_path, target_name))
                except (OSError, SyntaxError):
                    pass

            if not _is_probably_text_file(file_path):
                continue

            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            file_hits: list[dict[str, Any]] = []
            for idx, line in enumerate(lines, start=1):
                lowered = line.lower()
                if raw_identifier and raw_identifier in lowered:
                    start = max(1, idx - 3)
                    end = min(len(lines), idx + 3)
                    snippet = "\n".join(f"{line_no:04d}: {lines[line_no - 1]}" for line_no in range(start, end + 1))
                    file_hits.append({"line": idx, "source_excerpt": snippet})
                elif patterns and all(token in lowered for token in patterns[:2]):
                    start = max(1, idx - 3)
                    end = min(len(lines), idx + 3)
                    snippet = "\n".join(f"{line_no:04d}: {lines[line_no - 1]}" for line_no in range(start, end + 1))
                    file_hits.append({"line": idx, "source_excerpt": snippet})

                if len(file_hits) >= 3:
                    break

            if file_hits:
                text_matches.append({"file": str(file_path), "matches": file_hits})

        payload = {
            "requested_test_identifier": test_identifier,
            "extracted_test_function_name": target_name or None,
            "python_matches": python_matches,
            "text_matches": text_matches,
        }
        return _truncate(_json_dump(payload))

    model = ChatOpenAI(model=model_name, temperature=temperature)

    return create_agent(
        model=model,
        tools=[
            inspect_fix_code_structure,
            inspect_verification_bundle,
            get_failed_tests,
            find_relevant_files,
            list_directory,
            read_file,
            search_workspace,
            find_test_definitions,
        ],
        system_prompt=SYSTEM_PROMPT,
    )


class VerificationBundleToTestFailureSpecifications:
    """Convert a verification result bundle into test failure specifications."""

    def __init__(
        self,
        verification_result_path: str | os.PathLike[str],
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        workspace_root: str | os.PathLike[str] | None = None,
        pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
        buggy_code: str | None = None,
        fix_code: str | None = None,
        fix_specification: str | None = None,
    ) -> None:
        self.verification_result_path = str(Path(verification_result_path).expanduser())
        self.model_name = model_name
        self.temperature = temperature
        self.workspace_root = str(Path(workspace_root).expanduser()) if workspace_root else None
        self.pricing = pricing
        self.buggy_code = buggy_code
        self.fix_code = fix_code
        self.fix_specification = fix_specification

    def build_agent(self) -> Any:
        return build_feedback_agent(
            verification_result_path=self.verification_result_path,
            model_name=self.model_name,
            temperature=self.temperature,
            workspace_root=self.workspace_root,
            buggy_code=self.buggy_code,
            fix_code=self.fix_code,
            fix_specification=self.fix_specification,
        )

    def generate(self) -> TestFailureSpecificationsResult:
        global _LAST_ANALYSIS_USAGE
        _LAST_ANALYSIS_USAGE = TokenUsage()
        test_failure_specifications = analyze_verification_result(
            verification_result_path=self.verification_result_path,
            model_name=self.model_name,
            temperature=self.temperature,
            workspace_root=self.workspace_root,
            buggy_code=self.buggy_code,
            fix_code=self.fix_code,
            fix_specification=self.fix_specification,
        )
        usage = _LAST_ANALYSIS_USAGE
        estimated_cost_usd = self.pricing.estimate_cost_usd(usage) if self.pricing else None
        return TestFailureSpecificationsResult(
            verification_result_path=self.verification_result_path,
            workspace_root=self.workspace_root,
            test_failure_specifications=test_failure_specifications,
            model_name=self.model_name,
            temperature=self.temperature,
            usage=usage,
            estimated_cost_usd=estimated_cost_usd,
        )


def analyze_verification_result(
    verification_result_path: str | os.PathLike[str],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    workspace_root: str | os.PathLike[str] | None = None,
    buggy_code: str | None = None,
    fix_code: str | None = None,
    fix_specification: str | None = None,
) -> str:
    global _LAST_ANALYSIS_USAGE
    test_failure_specifications, usage = analyze_verification_result_with_usage(
        verification_result_path=verification_result_path,
        model_name=model_name,
        temperature=temperature,
        workspace_root=workspace_root,
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
    )
    _LAST_ANALYSIS_USAGE = usage
    return test_failure_specifications


def analyze_verification_result_with_usage(
    verification_result_path: str | os.PathLike[str],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    workspace_root: str | os.PathLike[str] | None = None,
    buggy_code: str | None = None,
    fix_code: str | None = None,
    fix_specification: str | None = None,
) -> tuple[str, TokenUsage]:
    agent = build_feedback_agent(
        verification_result_path=verification_result_path,
        model_name=model_name,
        temperature=temperature,
        workspace_root=workspace_root,
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
    )

    prompt = build_analysis_prompt(
        str(verification_result_path),
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
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


def generate_test_failure_specifications_result(
    verification_result_path: str | os.PathLike[str],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    workspace_root: str | os.PathLike[str] | None = None,
    pricing: ModelPricing | None = DEFAULT_GPT_4O_PRICING,
    buggy_code: str | None = None,
    fix_code: str | None = None,
    fix_specification: str | None = None,
) -> TestFailureSpecificationsResult:
    module = VerificationBundleToTestFailureSpecifications(
        verification_result_path=verification_result_path,
        model_name=model_name,
        temperature=temperature,
        workspace_root=workspace_root,
        pricing=pricing,
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
    )
    return module.generate()


def generate_test_failure_specifications(
    verification_result_path: str | os.PathLike[str],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    workspace_root: str | os.PathLike[str] | None = None,
    buggy_code: str | None = None,
    fix_code: str | None = None,
    fix_specification: str | None = None,
) -> str:
    result = generate_test_failure_specifications_result(
        verification_result_path=verification_result_path,
        model_name=model_name,
        temperature=temperature,
        workspace_root=workspace_root,
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
    )
    return result.test_failure_specifications



def _read_optional_cli_file(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a verifier result bundle with a LangChain agent.")
    parser.add_argument("verification_result_path", help="Path to a verifier result file or directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Agent model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Model temperature.")
    parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root used when searching for related tests, docs, configs, and source files.",
    )
    parser.add_argument("--buggy-code-file", default=None, help="Optional file containing the buggy replacement snippet.")
    parser.add_argument("--fix-code-file", default=None, help="Optional file containing the generated fix code that failed validation.")
    parser.add_argument("--fix-specification-file", default=None, help="Optional file containing the fixed specification used for the failed fix.")
    args = parser.parse_args(argv)

    output = analyze_verification_result(
        verification_result_path=args.verification_result_path,
        model_name=args.model,
        temperature=args.temperature,
        workspace_root=args.workspace_root,
        buggy_code=_read_optional_cli_file(args.buggy_code_file),
        fix_code=_read_optional_cli_file(args.fix_code_file),
        fix_specification=_read_optional_cli_file(args.fix_specification_file),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
