from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from time import sleep
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ConDefects.util.get_faulty_function_by_program_id import (  # noqa: E402
    ProgramFunctionLookupError,
    _find_method_ranges,
    _find_smallest_enclosing_python_function,
    _python_node_start_line,
)
from ConDefects.util.get_task_sections import get_task_sections  # noqa: E402
from demo_pipeline import (  # noqa: E402
    ProgramContext,
    build_fix_specification_input,
    import_viberepair_stage_functions,
    resolve_program_context,
)
from viberepair.langchain_support import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
)


DEFAULT_REPAIR_RUNS = 5
DEFAULT_DATASET_ROOT = REPO_ROOT / "ConDefects" / "datasetForExperiment"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "eval_specification_module_Ochiai"
OCHIAI_FL_FILENAME = "Ochiai_FL.txt"
OCHIAI_REFERENCE_TEXT = (
    "Ochiai fault localization identifies this line as having the highest suspiciousness "
    "score (T0). It may be related to the bug, but this is only a reference and should "
    "not be considered conclusive evidence."
)
LANGUAGES = ("Python", "Java")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched VibeRepair repairs for ConDefects datasetForExperiment and save each "
            "independent Ochiai-FL repair result as a JSON file."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the ConDefects datasetForExperiment directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where per-run JSON result files are written.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_REPAIR_RUNS,
        help=f"Number of independent repairs per program id. Default: {DEFAULT_REPAIR_RUNS}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Chat model name used by all VibeRepair modules. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature used by all VibeRepair modules. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--program-id",
        action="append",
        dest="program_ids",
        help="Limit the batch to one program id. Repeat this option for multiple ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of program ids to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between independent repair runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files instead of skipping them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first program/run error.",
    )
    return parser.parse_args()


def iter_program_contexts(
    dataset_root: str | Path,
    program_ids: list[str] | None = None,
) -> list[ProgramContext]:
    dataset_root = Path(dataset_root)
    if program_ids:
        return [resolve_program_context(program_id, dataset_root) for program_id in program_ids]

    contexts: list[ProgramContext] = []
    for task_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for language in LANGUAGES:
            language_dir = task_dir / language
            if not language_dir.is_dir():
                continue
            for program_dir in sorted(path for path in language_dir.iterdir() if path.is_dir()):
                if not _has_faulty_version(program_dir):
                    continue
                contexts.append(
                    ProgramContext(
                        program_id=program_dir.name,
                        program_dir=program_dir,
                        task_name=task_dir.name,
                        language=language,
                    )
                )
    return contexts


def _has_faulty_version(program_dir: Path) -> bool:
    return (program_dir / "faultyVersion.py").exists() or (program_dir / "faultyVersion.java").exists()


def result_path(output_dir: str | Path, context: ProgramContext, repair_index: int) -> Path:
    filename = (
        f"{_safe_filename(context.task_name)}_"
        f"{_safe_filename(context.language)}_"
        f"{_safe_filename(context.program_id)}_"
        f"repair_{repair_index:02d}.json"
    )
    return Path(output_dir) / filename


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def get_ochiai_faulty_function_with_reference_comment(context: ProgramContext) -> tuple[str, int]:
    ochiai_fl_path = context.program_dir / OCHIAI_FL_FILENAME
    if not ochiai_fl_path.exists():
        raise ProgramFunctionLookupError(
            f"Program {context.program_id} does not contain {OCHIAI_FL_FILENAME}: "
            f"{context.program_dir}"
        )

    fault_line_number = _read_ochiai_fault_line_number(ochiai_fl_path)

    java_path = context.program_dir / "faultyVersion.java"
    if java_path.exists():
        return (
            _extract_from_java_with_ochiai_comment(
                java_path,
                fault_line_number,
                context.program_id,
            ),
            fault_line_number,
        )

    python_path = context.program_dir / "faultyVersion.py"
    if python_path.exists():
        return (
            _extract_from_python_with_ochiai_comment(
                python_path,
                fault_line_number,
                context.program_id,
            ),
            fault_line_number,
        )

    raise ProgramFunctionLookupError(
        f"Program {context.program_id} does not contain faultyVersion.java or "
        f"faultyVersion.py: {context.program_dir}"
    )


def _extract_from_java_with_ochiai_comment(
    java_path: Path,
    fault_line_number: int,
    program_id: str,
) -> str:
    lines = java_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not 1 <= fault_line_number <= len(lines):
        raise ProgramFunctionLookupError(
            f"Fault line {fault_line_number} is out of range for {java_path}"
        )

    target_index = fault_line_number - 1
    for start_index, end_index in _find_method_ranges(lines):
        if start_index <= target_index <= end_index:
            return _format_lines_with_ochiai_comment(
                lines[start_index:end_index + 1],
                target_index - start_index,
                "Java",
            )

    raise ProgramFunctionLookupError(
        f"Could not find an enclosing Java method for program {program_id} at line "
        f"{fault_line_number}"
    )


def _extract_from_python_with_ochiai_comment(
    python_path: Path,
    fault_line_number: int,
    program_id: str,
) -> str:
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
        return _format_lines_with_ochiai_comment(lines, fault_line_number - 1, "Python")

    start_line = _python_node_start_line(target_node)
    end_line = getattr(target_node, "end_lineno", None)
    if end_line is None:
        raise ProgramFunctionLookupError(
            f"Could not determine the end line of the enclosing Python function for "
            f"program {program_id}"
        )

    return _format_lines_with_ochiai_comment(
        lines[start_line - 1:end_line],
        fault_line_number - start_line,
        "Python",
    )


def _format_lines_with_ochiai_comment(
    lines: list[str],
    fault_line_index: int,
    language: str,
) -> str:
    if not 0 <= fault_line_index < len(lines):
        raise ProgramFunctionLookupError("Fault line is outside the extracted faulty function")

    marked_lines = list(lines)
    marked_lines[fault_line_index] = _append_ochiai_comment(
        marked_lines[fault_line_index],
        language,
    )
    return "\n".join(marked_lines).rstrip()


def _append_ochiai_comment(line: str, language: str) -> str:
    marker = "//" if language == "Java" else "#"
    comment = f"{marker} {OCHIAI_REFERENCE_TEXT}"
    if not line:
        return comment
    return f"{line} {comment}"


def _read_ochiai_fault_line_number(path: Path) -> int:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ProgramFunctionLookupError(f"Invalid {OCHIAI_FL_FILENAME} value in {path}: {raw!r}") from exc


def run_one_repair(
    context: ProgramContext,
    task_information_cache: dict[str, str],
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    (
        generate_flawed_specification_result,
        generate_fix_specification_result,
        generate_fix_code_result,
    ) = import_viberepair_stage_functions()

    faulty_function, _fault_line_number = get_ochiai_faulty_function_with_reference_comment(context)
    if context.task_name not in task_information_cache:
        task_information_cache[context.task_name] = get_task_sections(context.task_name)
    task_information = task_information_cache[context.task_name]

    model = build_chat_model(model_name=model_name, temperature=temperature)

    flawed_specification_result = generate_flawed_specification_result(
        buggy_code=faulty_function,
        model_name=model_name,
        temperature=temperature,
        model=model,
    )

    fix_specification_input = build_fix_specification_input(
        task_information=task_information,
        flawed_specification=flawed_specification_result.flawed_specification,
    )
    fix_specification_result = generate_fix_specification_result(
        information=fix_specification_input,
        model_name=model_name,
        temperature=temperature,
        model=model,
    )

    fix_code_result = generate_fix_code_result(
        fix_specification=fix_specification_result.fix_specification,
        buggy_code=faulty_function,
        model_name=model_name,
        temperature=temperature,
        model=model,
    )

    return {
        "status": "success",
        "programId": context.program_id,
        "repair_index": None,
        "task_name": context.task_name,
        "language": context.language,
        "flawed_specification": flawed_specification_result.flawed_specification,
        "fix_specification_reasoning": {
            "intended_behavior": fix_specification_result.intended_behavior,
            "diagnosis_and_repair_directions": (
                fix_specification_result.diagnosis_and_repair_directions
            ),
            "raw_response": fix_specification_result.raw_response,
        },
        "fixed_specification": fix_specification_result.fix_specification,
        "fix_code": fix_code_result.replacement_code,
        "estimated_cost": {
            "buggy_code_to_flawed_specification": (
                flawed_specification_result.estimated_cost_usd
            ),
            "fix_specification": fix_specification_result.estimated_cost_usd,
            "fix_specification_and_buggy_code_to_fix_code": (
                fix_code_result.estimated_cost_usd
            ),
            "total": _sum_costs(
                flawed_specification_result.estimated_cost_usd,
                fix_specification_result.estimated_cost_usd,
                fix_code_result.estimated_cost_usd,
            ),
        },
        "usage": {
            "buggy_code_to_flawed_specification": _usage_dict(
                flawed_specification_result.usage
            ),
            "fix_specification": _usage_dict(fix_specification_result.usage),
            "fix_specification_and_buggy_code_to_fix_code": _usage_dict(
                fix_code_result.usage
            ),
        },
    }


def _usage_dict(usage: Any) -> dict[str, int | None]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _sum_costs(*costs: float | None) -> float | None:
    if any(cost is None for cost in costs):
        return None
    return sum(cost for cost in costs if cost is not None)


def error_record(
    context: ProgramContext,
    repair_index: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "status": "error",
        "programId": context.program_id,
        "repair_index": repair_index,
        "task_name": context.task_name,
        "language": context.language,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("Error: --runs must be at least 1", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1 when provided", file=sys.stderr)
        return 1

    try:
        import_viberepair_stage_functions()
        build_chat_model(model_name=args.model, temperature=args.temperature)
    except (EnvironmentError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        contexts = iter_program_contexts(args.dataset_root, args.program_ids)
    except (ProgramFunctionLookupError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        contexts = contexts[: args.limit]

    output_dir = Path(args.output_dir)
    task_information_cache: dict[str, str] = {}
    total_runs = len(contexts) * args.runs
    completed = 0
    skipped = 0
    failed = 0

    print(
        f"Processing {len(contexts)} program ids x {args.runs} repairs "
        f"with model={args.model}, temperature={args.temperature}"
    )
    print(f"Reading fault lines from {OCHIAI_FL_FILENAME}")
    print(f"Writing JSON results to {output_dir}")

    for context in contexts:
        for repair_index in range(1, args.runs + 1):
            output_path = result_path(output_dir, context, repair_index)
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[skip] {output_path}")
                continue

            try:
                payload = run_one_repair(
                    context=context,
                    task_information_cache=task_information_cache,
                    model_name=args.model,
                    temperature=args.temperature,
                )
                payload["repair_index"] = repair_index
                write_json(output_path, payload)
                completed += 1
                print(f"[ok] {context.program_id} repair {repair_index}/{args.runs}")
            except Exception as exc:
                failed += 1
                write_json(
                    output_path,
                    error_record(
                        context=context,
                        repair_index=repair_index,
                        error=exc,
                    ),
                )
                print(
                    f"[error] {context.program_id} repair {repair_index}/{args.runs}: {exc}",
                    file=sys.stderr,
                )
                if args.stop_on_error:
                    print_summary(completed, skipped, failed, total_runs)
                    return 1

            if args.sleep_seconds > 0:
                sleep(args.sleep_seconds)

    print_summary(completed, skipped, failed, total_runs)
    return 1 if failed and args.stop_on_error else 0


def print_summary(completed: int, skipped: int, failed: int, total_runs: int) -> None:
    print(
        f"Finished: completed={completed}, skipped={skipped}, "
        f"failed={failed}, total_planned={total_runs}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
