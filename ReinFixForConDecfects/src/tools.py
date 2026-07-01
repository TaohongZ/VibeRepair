from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from langchain.tools import StructuredTool
except ImportError:  # pragma: no cover - depends on installed langchain version
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:  # pragma: no cover - local smoke tests without langchain
        class StructuredTool:  # type: ignore[no-redef]
            def __init__(self, func):
                self.func = func
                self.name = func.__name__
                self.description = (func.__doc__ or "").strip()

            @classmethod
            def from_function(cls, func):
                return cls(func)

            def invoke(self, tool_input):
                return self.func(tool_input)

            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional local optimization
    tiktoken = None

from .config import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_ROOT, RAG_SEARCH_URL
from .condefects_dataset import (
    ProgramContext,
    ProgramFunctionLookupError,
    _find_method_ranges,
    resolve_program_context,
)


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


proj_name_gl = ""
_current_project: ProjectState | None = None


def open_proj(proj_name: str) -> str:
    """
    Open the current analysing project by proj_name.

    @param proj_name: The name of project to open. "
    @return: The result of the request execution.

    """

    if not proj_name.endswith("\n"):
        print(proj_name)
    normalized = _normalize_project_name(proj_name)
    print(f'open("{normalized}")')

    context = resolve_program_context(normalized, DEFAULT_DATASET_ROOT, require_ochiai=False)
    source_path = context.source_path
    source = source_path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    language = "Java" if source_path.suffix == ".java" else "Python"
    global _current_project, proj_name_gl
    proj_name_gl = normalized
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
        f"Static local analysis is used as a Joern substitute."
    )


def identify_variable(input_string: str) -> str:
    """
    Identifies variable definitions and usages for a given variable name.

    @param input_string: A string containing both the variable name and file name,
                         separated by a comma. Format: variable_name, file_name
    @return: The result of the query execution.
    """
    project = _require_project()
    variable_name, file_name = _split_two_args(input_string)
    variable_name = variable_name.strip()
    file_name = file_name.strip()
    print(f'identify_variable("{variable_name}", "{file_name}")')
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."

    pattern = re.compile(rf"\b{re.escape(variable_name)}\b")
    hits = []
    for index, line in enumerate(project.lines, start=1):
        if pattern.search(line):
            hits.append({"line": index, "code": line.strip()})
        if len(hits) >= 20:
            break
    return json.dumps(hits, ensure_ascii=False, indent=2)


def trace_method_usage(method_name: str) -> str:
    """
    Traces the usage of a specific method in the code.

    @param method_name: The name of the method to trace.
    @return: The result of the query execution.
    """
    project = _require_project()
    method_name = method_name.strip().replace('"', "")
    print(f'trace_method_usage("{method_name}")')
    hits = []
    call_pattern = re.compile(rf"\b{re.escape(method_name)}\s*\(")
    for index, line in enumerate(project.lines, start=1):
        if call_pattern.search(line):
            hits.append({"line": index, "code": line.strip(), "file": project.source_path.name})
    return slim_joern_token(json.dumps(hits, ensure_ascii=False, indent=2))


def find_method_in_file(input_string: str) -> str:
    """
    Finds a specific method within a given file and retrieves its location and the whole method code. DO NOT Add any quota marks(") to the input parameter because the tool call will process it.

    @param input_string: A string containing both the method name and file name, separated by a comma. method_name: The name of the method to find. file_name: The name of the file to search within (only the file name, not the file path). Format: method_name, file_name.
    @return: The result of the query execution.

    input_string Example: visit, CheckSideEffects.java
    """
    project = _require_project()
    method_name, file_name = _split_two_args(input_string)
    method_name = method_name.strip().replace('"', "")
    file_name = file_name.strip().replace('"', "")
    print(f'find_method_in_file("{method_name}", "{file_name}")')
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."

    methods = _find_methods(project, method_name)
    if not methods:
        return f"No method named {method_name} was found in {project.source_path.name}."
    return json.dumps([_entity_to_dict(method) for method in methods], ensure_ascii=False, indent=2)


def analyze_method_details(input_string: str) -> str:
    """
    Identifies various aspects of a given method in a file. DO NOT Add any quota marks(") to the input parameter because the tool call will process it.

    @param input_string: A string containing both the method name and file name, separated by a comma. method_name: The name of the method to find. file_name: The name of the file to search within (only the file name, not the file path). Format: method_name, file_name. Example: visit, CheckSideEffects.java

    @return: The result of the query execution, including:
        - Method name and parameter types
        - Filename where the method is located
        - All throw statements within the method
        - All if statement conditions within the method
        - All method calls within the method
        - All variable assignments within the method
        - The return statement of the method
    """
    project = _require_project()
    input_string = input_string.replace('"', "")
    method_name, file_name = _split_two_args(input_string)
    method_name = method_name.strip()
    file_name = file_name.strip()
    print(f'analyze_method_details("{method_name}", "{file_name}")')
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."

    methods = _find_methods(project, method_name)
    if not methods:
        return f"No method named {method_name} was found in {project.source_path.name}."
    details = [_analyze_entity(project, method) for method in methods]
    return json.dumps(details, ensure_ascii=False, indent=2)


def find_class_loc(input_string: str) -> str:
    """
    Finds the location of a class with a given name.

    @param input_string: The name of the class to locate
    @return: A string containing the location/full type name of the class
    """
    project = _require_project()
    class_name = input_string.strip().replace('"', "")
    print(f'find_class_loc("{class_name}")')
    classes = [klass for klass in project.classes if class_name in klass.name or klass.name in class_name]
    if not classes:
        return f"No class named {class_name} was found in {project.source_path.name}."
    return json.dumps(
        [
            {
                "class": klass.name,
                "file": project.source_path.name,
                "start_line": klass.start_line,
                "end_line": klass.end_line,
            }
            for klass in classes
        ],
        ensure_ascii=False,
        indent=2,
    )


def identify_class(input_string: str) -> str:
    """
    Identifies the class code matches the given search string for a class name.

    @param input_string: The name of the class
    @return: A string containing the class codes
    """

    project = _require_project()
    class_name = input_string.strip().replace('"', "")
    print(f'identify_class("{class_name}")')
    classes = [klass for klass in project.classes if class_name in klass.name or klass.name in class_name]
    if not classes:
        return f"No class named {class_name} was found in {project.source_path.name}."
    return json.dumps([_entity_to_dict(klass) for klass in classes], ensure_ascii=False, indent=2)


def analyze_method_control_flow(method_name: str) -> str:
    """
    Analyzes the control flow (if, while, for, etc.) structures within a specific method.

    @param method_name: The name of the method to analyze.
    @return: The code of the control flow structures within the method.
    """
    project = _require_project()
    method_name = method_name.strip().replace('"', "")
    print(f'analyze_method_control_flow("{method_name}")')
    methods = _find_methods(project, method_name)
    if not methods:
        return f"No method named {method_name} was found in {project.source_path.name}."
    controls = []
    for method in methods:
        controls.append({"method": method.name, "control_flow": _control_flow_lines(method.code)})
    return json.dumps(controls, ensure_ascii=False, indent=2)


def get_imports(file_name: str) -> str:
    """
    Retrieves all import statements from a specific file.

    @param file_name: The name of the file to search within (only the file name, not the file path).
    @return: The code of all import statements within the file.
    """
    project = _require_project()
    file_name = file_name.strip().replace('"', "")
    print(f'get_imports("{file_name}")')
    if file_name and not _file_matches(project.source_path, file_name):
        return f"No file matching {file_name} is open."
    imports = [
        {"line": index, "code": line.strip()}
        for index, line in enumerate(project.lines, start=1)
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    return json.dumps(imports, ensure_ascii=False, indent=2)


def close_proj(proj_name: str) -> str:
    """
    Close the current analysing project by proj_name.

    @param proj_name: The name of project to close. Example format: "Chart-1_buggy"
    @return: The result of the request execution.

    Example: close_proj("Chart-1_buggy")
    """
    normalized = _normalize_project_name(proj_name)
    if proj_name.endswith("\n"):
        proj_name.replace("\n", "")
    if not proj_name.endswith("\n"):
        print(proj_name)

    print(f'close("{normalized}")')
    global _current_project, proj_name_gl
    _current_project = None
    proj_name_gl = ""
    return f"Closed ConDefects project {normalized}."


def example_patch_search(input_string: str) -> str:
    """
    Search for the most similar (by similarity rate) fix pattern example in a Knowledge Base and return it.

    @param input_string: Buggy code concatenate with Root cause. input example:  public boolean equals(Object obj) \n\n    if (obj == this) \n   return true, Root Cause: The `equals` method in the `ShapeList` class is calling `super.equals(obj)` which likely only checks the reference equality or fields defined in the superclass, rather than checking the properties specific to `ShapeList`. This would lead to instances that may contain the same shapes being considered unequal if the parent class does not override `equals` appropriately for deep equality checks.
    @return: The most similar fix pattern, including BuggyCode, FixedCode, RootCause

    """
    data = {
        "query": input_string,
        "n": 1,
        "threshold": 0.6,
    }
    start_time = datetime.now()
    try:
        response = requests.post(RAG_SEARCH_URL, json=data, timeout=120)
        payload: Any = response.json()
    except Exception as exc:
        payload = [{"message": f"Example search failed: {exc}"}]
    end_time = datetime.now()
    time_difference = end_time - start_time
    print(f"Time cost for retrieving: {time_difference}")
    if proj_name_gl:
        _record_search(payload)
        close_proj(proj_name=proj_name_gl)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def slim_joern_token(joern_response):
    token_upper_limit = 10000
    response_list = joern_response.split('\n')
    slim_res_list = []
    for line in response_list:
        curr_line_token_cnt = num_tokens_from_string(line)
        token_upper_limit -= curr_line_token_cnt
        if token_upper_limit <= 0:
            break
        slim_res_list.append(line)
    return '\n'.join(slim_res_list)


def num_tokens_from_string(string: str) -> int:
    if tiktoken is None:
        return max(1, len(string) // 4)
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _normalize_project_name(proj_name: str) -> str:
    normalized = str(proj_name).strip().strip('"').strip("'")
    if normalized.endswith("_buggy"):
        normalized = normalized[: -len("_buggy")]
    return normalized


def _require_project() -> ProjectState:
    if _current_project is None:
        raise ProgramFunctionLookupError("No ConDefects project is open. Call open_proj first.")
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
            decorators = getattr(node, "decorator_list", None) or []
            start_line = min([node.lineno] + [decorator.lineno for decorator in decorators])
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
    method_name = name_match.group(1) if name_match else ""
    params_match = re.search(r"\((.*?)\)", signature)
    if not params_match:
        return method_name, []
    params = []
    for raw_param in params_match.group(1).split(","):
        raw_param = raw_param.strip()
        if raw_param:
            params.append(raw_param)
    return method_name, params


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
        "assignments": _java_assignments(code_lines),
        "return_statements": _lines_matching(code_lines, r"\breturn\b"),
    }


def _analyze_python_entity(entity: CodeEntity, code_lines: list[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(entity.code)
    except SyntaxError:
        return {
            "method": entity.name,
            "parameters": entity.parameters,
            "filename": "faultyVersion.py",
            "throw_statements": _lines_matching(code_lines, r"\braise\b"),
            "if_conditions": _lines_matching(code_lines, r"^\s*if\b"),
            "method_calls": _python_calls_by_regex(entity.code),
            "assignments": _lines_matching(code_lines, r"(?<![=!<>])=(?!=)"),
            "return_statements": _lines_matching(code_lines, r"^\s*return\b"),
        }

    return {
        "method": entity.name,
        "parameters": entity.parameters,
        "filename": "faultyVersion.py",
        "throw_statements": _python_node_lines(tree, ast.Raise, code_lines),
        "if_conditions": _python_if_conditions(tree, code_lines),
        "method_calls": _python_call_names(tree),
        "assignments": _python_node_lines(tree, (ast.Assign, ast.AnnAssign, ast.AugAssign), code_lines),
        "return_statements": _python_node_lines(tree, ast.Return, code_lines),
    }


def _control_flow_lines(code: str) -> list[str]:
    patterns = [
        r"\bif\b",
        r"\belse\b",
        r"\belif\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\bswitch\b",
        r"\bcase\b",
        r"\btry\b",
        r"\bcatch\b",
    ]
    regex = re.compile("|".join(patterns))
    return [line.strip() for line in code.splitlines() if regex.search(line)]


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


def _java_assignments(lines: list[str]) -> list[str]:
    assignments = []
    for line in lines:
        stripped = line.strip()
        if re.search(r"(?<![=!<>])=(?!=)", stripped):
            assignments.append(stripped)
    return assignments


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


def _python_callable_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _python_callable_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


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


def _record_search(payload: Any) -> None:
    if "'similarity'" not in str(payload) and '"similarity"' not in str(payload):
        return
    output_path = DEFAULT_OUTPUT_ROOT / "searched_example.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(proj_name_gl + "\n")


# Define the tools
open_proj_tool = StructuredTool.from_function(open_proj)
identify_variable_tool = StructuredTool.from_function(identify_variable)
trace_method_usage_tool = StructuredTool.from_function(trace_method_usage)
find_method_in_file_tool = StructuredTool.from_function(find_method_in_file)
analyze_method_details_tool = StructuredTool.from_function(analyze_method_details)
analyze_method_control_flow_tool = StructuredTool.from_function(analyze_method_control_flow)
find_class_loc_tool = StructuredTool.from_function(find_class_loc)
identify_class_tool = StructuredTool.from_function(identify_class)
get_imports_tool = StructuredTool.from_function(get_imports)
close_proj_tool = StructuredTool.from_function(close_proj)
example_patch_search_tool = StructuredTool.from_function(example_patch_search)

