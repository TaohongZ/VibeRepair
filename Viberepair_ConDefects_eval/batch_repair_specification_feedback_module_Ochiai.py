from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ConDefects.util.get_faulty_function_by_program_id import (  # noqa: E402
    ProgramFunctionLookupError,
)
from demo_pipeline import ProgramContext, resolve_program_context  # noqa: E402
import full_demo_pipeline as full_demo  # noqa: E402
from viberepair.langchain_support import DEFAULT_MODEL_NAME, build_chat_model  # noqa: E402


DEFAULT_DATASET_ROOT = REPO_ROOT / "ConDefects" / "datasetForExperiment"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "eval_specification_feedback_module_Ochiai"
DEFAULT_WORK_ROOT = Path("/tmp") / "viberepair_specification_feedback_module_Ochiai"
DEFAULT_PRIORITY_DATASET_ROOT = DEFAULT_DATASET_ROOT
DEFAULT_OUTER_REPAIR_RUNS = 5
DEFAULT_FEEDBACK_LOOPS = 5
DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
DEFAULT_TEMPERATURE = 0.2
LANGUAGES = ("Python", "Java")
OCHIAI_FL_FILENAME = "Ochiai_FL.txt"
OCHIAI_REFERENCE_TEXT = (
    "Ochiai fault localization identifies this line as having the highest suspiciousness "
    "score (T0). It may be related to the bug, but this is only a reference and should "
    "not be considered conclusive evidence."
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched iterative VibeRepair repairs for ConDefects datasetForExperiment. "
            "Each program is repaired with the full feedback pipeline; if it fails, "
            "the temporary run is removed and the full pipeline is restarted from scratch."
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
        help="Directory where per-program JSON result files are written.",
    )
    parser.add_argument(
        "--work-root",
        default=str(DEFAULT_WORK_ROOT),
        help="Root directory used for temporary full-pipeline runs.",
    )
    parser.add_argument(
        "--priority-dataset-root",
        default=str(DEFAULT_PRIORITY_DATASET_ROOT),
        help=(
            "Programs that also exist under this dataset root are repaired before "
            "the remaining datasetForExperiment programs. Default: ConDefects/datasetForExperiment."
        ),
    )
    parser.add_argument(
        "--outer-runs",
        type=int,
        default=DEFAULT_OUTER_REPAIR_RUNS,
        help=(
            "Maximum number of full restarts per program. "
            f"Default: {DEFAULT_OUTER_REPAIR_RUNS}"
        ),
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=DEFAULT_FEEDBACK_LOOPS,
        help=(
            "Maximum feedback loops inside each full pipeline run. "
            f"Default: {DEFAULT_FEEDBACK_LOOPS}"
        ),
    )
    parser.add_argument(
        "--test-timeout-seconds",
        type=float,
        default=DEFAULT_TEST_TIMEOUT_SECONDS,
        help=(
            "Timeout used by each validation compile/run command. "
            f"Default: {DEFAULT_TEST_TIMEOUT_SECONDS:g}"
        ),
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
        "--test-limit",
        action="append",
        dest="test_limit",
        help="Optional test case filename to run. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between programs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-program JSON files instead of skipping them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first program error.",
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


def collect_program_keys(dataset_root: str | Path) -> set[tuple[str, str, str]]:
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        return set()

    keys: set[tuple[str, str, str]] = set()
    for task_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for language in LANGUAGES:
            language_dir = task_dir / language
            if not language_dir.is_dir():
                continue
            for program_dir in sorted(path for path in language_dir.iterdir() if path.is_dir()):
                if _has_faulty_version(program_dir):
                    keys.add((task_dir.name, language, program_dir.name))
    return keys


def prioritize_contexts(
    contexts: list[ProgramContext],
    priority_keys: set[tuple[str, str, str]],
) -> list[ProgramContext]:
    if not priority_keys:
        return contexts
    return sorted(
        contexts,
        key=lambda context: 0 if context_key(context) in priority_keys else 1,
    )


def context_key(context: ProgramContext) -> tuple[str, str, str]:
    return (context.task_name, context.language, context.program_id)


def _has_faulty_version(program_dir: Path) -> bool:
    return (program_dir / "faultyVersion.py").exists() or (program_dir / "faultyVersion.java").exists()


def result_path(output_dir: str | Path, context: ProgramContext) -> Path:
    filename = (
        f"{_safe_filename(context.task_name)}_"
        f"{_safe_filename(context.language)}_"
        f"{_safe_filename(context.program_id)}.json"
    )
    return Path(output_dir) / filename


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def locate_preferred_faulty_function(context: ProgramContext) -> full_demo.OchiaiFunctionLocation:
    fault_line_number = read_ochiai_fault_line(context.program_dir)
    full_demo.OCHIAI_FL_FILENAME = OCHIAI_FL_FILENAME

    java_path = context.program_dir / "faultyVersion.java"
    if java_path.exists():
        return locate_java_function(context, java_path, fault_line_number, OCHIAI_FL_FILENAME)

    python_path = context.program_dir / "faultyVersion.py"
    if python_path.exists():
        return locate_python_function(context, python_path, fault_line_number, OCHIAI_FL_FILENAME)

    raise ProgramFunctionLookupError(
        f"Program {context.program_id} does not contain faultyVersion.java or faultyVersion.py: "
        f"{context.program_dir}"
    )


def read_ochiai_fault_line(program_dir: Path) -> int:
    ochiai_path = program_dir / OCHIAI_FL_FILENAME
    if not ochiai_path.exists():
        raise ProgramFunctionLookupError(
            f"Program {program_dir.name} does not contain {OCHIAI_FL_FILENAME}: "
            f"{program_dir}"
        )
    return read_fault_line_number(ochiai_path)


def read_fault_line_number(path: Path) -> int:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ProgramFunctionLookupError(f"Invalid fault line value in {path}: {raw!r}") from exc


def locate_java_function(
    context: ProgramContext,
    java_path: Path,
    fault_line_number: int,
    fault_source: str,
) -> full_demo.OchiaiFunctionLocation:
    lines = java_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not 1 <= fault_line_number <= len(lines):
        raise ProgramFunctionLookupError(
            f"Fault line {fault_line_number} from {fault_source} is out of range for {java_path}"
        )

    target_index = fault_line_number - 1
    for start_index, end_index in full_demo._find_method_ranges(lines):
        if start_index <= target_index <= end_index:
            return full_demo.OchiaiFunctionLocation(
                program_id=context.program_id,
                program_dir=context.program_dir,
                source_path=java_path,
                task_name=context.task_name,
                language=context.language,
                fault_line_number=fault_line_number,
                start_line_index=start_index,
                end_line_index=end_index,
                faulty_function=format_lines_with_fault_comment(
                    lines[start_index : end_index + 1],
                    target_index - start_index,
                    fault_source,
                    context.language,
                ),
            )

    raise ProgramFunctionLookupError(
        f"Could not find an enclosing Java method for program {context.program_id} "
        f"at line {fault_line_number} from {fault_source}"
    )


def locate_python_function(
    context: ProgramContext,
    python_path: Path,
    fault_line_number: int,
    fault_source: str,
) -> full_demo.OchiaiFunctionLocation:
    source = python_path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    if not 1 <= fault_line_number <= len(lines):
        raise ProgramFunctionLookupError(
            f"Fault line {fault_line_number} from {fault_source} is out of range for {python_path}"
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramFunctionLookupError(
            f"Could not parse faultyVersion.py for program {context.program_id}: {exc}"
        ) from exc

    target_node = full_demo._find_smallest_enclosing_python_function(tree, fault_line_number)
    if target_node is None:
        start_line_index = 0
        end_line_index = len(lines) - 1
    else:
        start_line_index = full_demo._python_node_start_line(target_node) - 1
        end_line = getattr(target_node, "end_lineno", None)
        if end_line is None:
            raise ProgramFunctionLookupError(
                f"Could not determine the end line of the enclosing Python function for "
                f"program {context.program_id}"
            )
        end_line_index = end_line - 1

    return full_demo.OchiaiFunctionLocation(
        program_id=context.program_id,
        program_dir=context.program_dir,
        source_path=python_path,
        task_name=context.task_name,
        language=context.language,
        fault_line_number=fault_line_number,
        start_line_index=start_line_index,
        end_line_index=end_line_index,
        faulty_function=format_lines_with_fault_comment(
            lines[start_line_index : end_line_index + 1],
            fault_line_number - 1 - start_line_index,
            fault_source,
            context.language,
        ),
    )


def format_lines_with_fault_comment(
    lines: list[str],
    fault_line_index: int,
    fault_source: str,
    language: str,
) -> str:
    if not 0 <= fault_line_index < len(lines):
        raise ProgramFunctionLookupError("Fault line is outside the extracted function")

    marked_lines = list(lines)
    marked_lines[fault_line_index] = append_fault_comment(
        marked_lines[fault_line_index],
        fault_source=fault_source,
        language=language,
    )
    return "\n".join(marked_lines).rstrip()


def append_fault_comment(line: str, fault_source: str, language: str) -> str:
    marker = "//" if language == "Java" else "#"
    comment = f"{marker} {OCHIAI_REFERENCE_TEXT}"
    if not line:
        return comment
    return f"{line} {comment}"


def install_full_demo_overrides(test_timeout_seconds: float) -> None:
    full_demo.TEST_TIMEOUT_SECONDS = test_timeout_seconds
    full_demo.locate_ochiai_faulty_function = locate_preferred_faulty_function


def run_program_repair(context: ProgramContext, args: argparse.Namespace) -> dict[str, Any]:
    output_path = result_path(args.output_dir, context)
    program_work_root = Path(args.work_root) / _safe_filename(context.program_id)
    final_summary: dict[str, Any] | None = None
    outer_records: list[dict[str, Any]] = []

    for outer_run_index in range(1, args.outer_runs + 1):
        outer_work_root = program_work_root / f"outer_run_{outer_run_index:02d}"
        if outer_work_root.exists():
            shutil.rmtree(outer_work_root)
        outer_work_root.mkdir(parents=True, exist_ok=True)

        print(
            f"[start] programId={context.program_id} outer_run={outer_run_index}/"
            f"{args.outer_runs}"
        )
        started_at = time.monotonic()
        try:
            pipeline_args = argparse.Namespace(
                program_id=context.program_id,
                dataset_root=args.dataset_root,
                model=args.model,
                temperature=args.temperature,
                max_loops=args.max_loops,
                work_root=str(outer_work_root),
                test_limit=args.test_limit,
                output_json=None,
                use_structure_adapter=False,
            )
            summary = full_demo.run_full_demo_pipeline(pipeline_args)
            summary["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
            summary["outer_run_index"] = outer_run_index
            write_json(Path(summary["work_dir"]) / "summary.json", build_chain_summary(summary))
            final_summary = summary

            outer_records.append(
                {
                    "outer_run_index": outer_run_index,
                    "status": summary.get("status"),
                    "work_dir": summary.get("work_dir"),
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "attempts": len(summary.get("attempts") or []),
                }
            )

            if summary.get("status") == "success":
                result = build_program_result(
                    context=context,
                    args=args,
                    summary=summary,
                    outer_records=outer_records,
                )
                write_json(output_path, result)
                print(f"[pass] programId={context.program_id} outer_run={outer_run_index}")
                return result

            if outer_run_index < args.outer_runs:
                print(
                    f"[retry] programId={context.program_id} failed in outer_run="
                    f"{outer_run_index}; deleting {outer_work_root}"
                )
                shutil.rmtree(outer_work_root, ignore_errors=True)
        except Exception as exc:
            elapsed_seconds = round(time.monotonic() - started_at, 3)
            error_summary = build_error_summary(
                context=context,
                args=args,
                outer_run_index=outer_run_index,
                outer_work_root=outer_work_root,
                elapsed_seconds=elapsed_seconds,
                error=exc,
            )
            final_summary = error_summary
            outer_records.append(
                {
                    "outer_run_index": outer_run_index,
                    "status": error_summary["status"],
                    "work_dir": error_summary["work_dir"],
                    "elapsed_seconds": elapsed_seconds,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            write_json(outer_work_root / "summary.json", build_chain_summary(error_summary))

            if outer_run_index < args.outer_runs:
                print(
                    f"[retry-error] programId={context.program_id} outer_run={outer_run_index}: "
                    f"{exc}; deleting {outer_work_root}",
                    file=sys.stderr,
                )
                shutil.rmtree(outer_work_root, ignore_errors=True)
            else:
                print(
                    f"[error] programId={context.program_id} final outer_run={outer_run_index}: "
                    f"{exc}",
                    file=sys.stderr,
                )

        if args.sleep_seconds > 0:
            sleep(args.sleep_seconds)

    if final_summary is None:
        final_summary = build_error_summary(
            context=context,
            args=args,
            outer_run_index=0,
            outer_work_root=program_work_root,
            elapsed_seconds=0.0,
            error=RuntimeError("No repair run was executed."),
        )

    result = build_program_result(
        context=context,
        args=args,
        summary=final_summary,
        outer_records=outer_records,
    )
    write_json(output_path, result)
    print(f"[failed] programId={context.program_id} after {args.outer_runs} full restarts")
    return result


def build_program_result(
    context: ProgramContext,
    args: argparse.Namespace,
    summary: dict[str, Any],
    outer_records: list[dict[str, Any]],
) -> dict[str, Any]:
    attempts = build_attempt_records(summary.get("attempts") or [])
    selected_attempt = find_selected_attempt(summary)
    flattened = flatten_selected_chain(summary, selected_attempt)
    result = {
        "status": "success" if summary.get("status") == "success" else "failed",
        "pass": summary.get("status") == "success",
        "programId": context.program_id,
        "repair_index": summary.get("outer_run_index"),
        "task_name": context.task_name,
        "language": context.language,
        "outer_runs": args.outer_runs,
        "max_loops": args.max_loops,
        "test_timeout_seconds": args.test_timeout_seconds,
        "validated_attempt_index": summary.get("validated_attempt_index"),
        "work_dir": summary.get("work_dir"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "outer_run_history": outer_records,
        "attempts": attempts,
        "last_chain_summary": build_chain_summary(summary),
        **flattened,
    }
    result["usage"] = build_usage_summary(summary, selected_attempt)
    result["estimated_cost"] = build_cost_summary(summary, selected_attempt)
    return result


def find_selected_attempt(summary: dict[str, Any]) -> dict[str, Any] | None:
    attempts = summary.get("attempts") or []
    validated_attempt_index = summary.get("validated_attempt_index")
    if validated_attempt_index is not None:
        for attempt in attempts:
            if attempt.get("attempt_index") == validated_attempt_index:
                return attempt
    if attempts:
        return attempts[-1]
    return None


def flatten_selected_chain(
    summary: dict[str, Any],
    selected_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "faulty_function": summary.get("faulty_function"),
        "flawed_specification": summary.get("flawed_specification"),
        "flawed_specification_usage": summary.get("flawed_specification_usage"),
        "flawed_specification_estimated_cost": summary.get(
            "flawed_specification_estimated_cost"
        ),
    }
    if not selected_attempt:
        return payload

    payload.update(
        {
            "selected_inner_attempt_index": selected_attempt.get("attempt_index"),
            "selected_inner_attempt_status": selected_attempt.get("status"),
            "pass_rate": selected_attempt.get("pass_rate"),
            "test_records": selected_attempt.get("test_records"),
            "test_error": selected_attempt.get("test_error"),
            "fix_specification_reasoning": selected_attempt.get(
                "fix_specification_reasoning"
            ),
            "fixed_specification": selected_attempt.get("fixed_specification"),
            "fix_code": selected_attempt.get("fix_code"),
            "fix_code_raw_response": selected_attempt.get("fix_code_raw_response"),
        }
    )
    return payload


def build_usage_summary(
    summary: dict[str, Any],
    selected_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    usage = {
        "buggy_code_to_flawed_specification": summary.get("flawed_specification_usage"),
        "fix_specification": None,
        "fix_specification_and_buggy_code_to_fix_code": None,
    }
    if selected_attempt:
        usage["fix_specification"] = selected_attempt.get("fix_specification_usage")
        usage["fix_specification_and_buggy_code_to_fix_code"] = selected_attempt.get(
            "fix_code_usage"
        )
    return usage


def build_cost_summary(
    summary: dict[str, Any],
    selected_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    cost = {
        "buggy_code_to_flawed_specification": summary.get(
            "flawed_specification_estimated_cost"
        ),
        "fix_specification": None,
        "fix_specification_and_buggy_code_to_fix_code": None,
    }
    if selected_attempt:
        cost["fix_specification"] = selected_attempt.get(
            "fix_specification_estimated_cost"
        )
        cost["fix_specification_and_buggy_code_to_fix_code"] = selected_attempt.get(
            "fix_code_estimated_cost"
        )
    return cost


def build_error_summary(
    context: ProgramContext,
    args: argparse.Namespace,
    outer_run_index: int,
    outer_work_root: Path,
    elapsed_seconds: float,
    error: Exception,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "programId": context.program_id,
        "task_name": context.task_name,
        "language": context.language,
        "work_dir": str(outer_work_root),
        "outer_run_index": outer_run_index,
        "max_loops": args.max_loops,
        "test_timeout_seconds": args.test_timeout_seconds,
        "validated_attempt_index": None,
        "attempts": [],
        "elapsed_seconds": elapsed_seconds,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def build_chain_summary(summary: dict[str, Any]) -> dict[str, Any]:
    payload = copy_present(
        summary,
        [
            "status",
            "generated_at",
            "programId",
            "task_name",
            "language",
            "work_dir",
            "max_loops",
            "use_structure_adapter",
            "faulty_function",
            "flawed_specification",
            "flawed_specification_usage",
            "flawed_specification_estimated_cost",
            "validated_attempt_index",
            "final_fix_code",
            "outer_run_index",
            "elapsed_seconds",
            "error_type",
            "error",
            "traceback",
        ],
    )
    payload["attempts"] = build_attempt_records(summary.get("attempts") or [])
    return payload


def build_attempt_records(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        copy_present(
            attempt,
            [
                "attempt_index",
                "status",
                "pass",
                "pass_rate",
                "attempt_dir",
                "verification_dir",
                "temp_dataset_root",
                "temp_program_dir",
                "temp_source_path",
                "test_records",
                "test_error",
                "fix_specification_reasoning",
                "fixed_specification",
                "fix_specification_usage",
                "fix_specification_estimated_cost",
                "fix_code",
                "fix_code_raw_response",
                "fix_code_usage",
                "fix_code_estimated_cost",
                "fix_code_before_structure_adapter",
                "fix_code_structure_adapter_changed",
                "fix_code_structure_adapter_reason",
                "fix_code_structure_adapter_raw_response",
                "fix_code_structure_adapter_usage",
                "fix_code_structure_adapter_estimated_cost",
                "fix_code_structure_adapter_deterministic_fallback_applied",
            ],
        )
        for attempt in attempts
    ]


def copy_present(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_batch_summary(path: str | Path, summary: dict[str, Any]) -> None:
    write_json(path, summary)


def main() -> int:
    args = parse_args()
    if args.outer_runs < 1:
        print("Error: --outer-runs must be at least 1", file=sys.stderr)
        return 1
    if args.max_loops < 1:
        print("Error: --max-loops must be at least 1", file=sys.stderr)
        return 1
    if args.test_timeout_seconds <= 0:
        print("Error: --test-timeout-seconds must be positive", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1 when provided", file=sys.stderr)
        return 1

    install_full_demo_overrides(args.test_timeout_seconds)

    try:
        full_demo.import_viberepair_full_pipeline_functions()
        build_chat_model(model_name=args.model, temperature=args.temperature)
    except (EnvironmentError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        contexts = iter_program_contexts(args.dataset_root, args.program_ids)
    except (ProgramFunctionLookupError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    priority_keys = collect_program_keys(args.priority_dataset_root)
    contexts = prioritize_contexts(contexts, priority_keys)

    if args.limit is not None:
        contexts = contexts[: args.limit]

    priority_program_count = sum(
        1 for context in contexts if context_key(context) in priority_keys
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_summary_path = output_dir / "batch_summary.json"
    batch_summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "priority_programs": priority_program_count,
        "remaining_programs": len(contexts) - priority_program_count,
        "output_dir": str(output_dir.resolve()),
        "work_root": str(Path(args.work_root).resolve()),
        "outer_runs": args.outer_runs,
        "max_loops": args.max_loops,
        "test_timeout_seconds": args.test_timeout_seconds,
        "total_programs": len(contexts),
        "completed": 0,
        "skipped": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "programs": [],
    }
    write_batch_summary(batch_summary_path, batch_summary)

    print(
        f"Processing {len(contexts)} program ids with model={args.model}, "
        f"temperature={args.temperature}, outer_runs={args.outer_runs}, "
        f"max_loops={args.max_loops}, test_timeout={args.test_timeout_seconds:g}s"
    )
    print(
        f"Priority dataset root: {args.priority_dataset_root} "
        f"({priority_program_count} priority, "
        f"{len(contexts) - priority_program_count} remaining)"
    )
    print(f"Writing JSON results to {output_dir}")
    print(f"Writing temporary runs to {args.work_root}")
    print(f"Reading fault lines from {OCHIAI_FL_FILENAME}")

    for index, context in enumerate(contexts, start=1):
        output_path = result_path(output_dir, context)
        if output_path.exists() and not args.overwrite:
            batch_summary["skipped"] += 1
            batch_summary["programs"].append(
                {
                    "programId": context.program_id,
                    "task_name": context.task_name,
                    "language": context.language,
                    "status": "skipped",
                    "result_json": str(output_path),
                }
            )
            write_batch_summary(batch_summary_path, batch_summary)
            print(f"[skip] {index}/{len(contexts)} {output_path}")
            continue

        print(
            f"\n[{index}/{len(contexts)}] repairing programId={context.program_id} "
            f"task={context.task_name} language={context.language}"
        )
        try:
            result = run_program_repair(context, args)
            batch_summary["completed"] += 1
            if result.get("pass") is True:
                batch_summary["passed"] += 1
            else:
                batch_summary["failed"] += 1
            batch_summary["programs"].append(
                {
                    "programId": context.program_id,
                    "task_name": context.task_name,
                    "language": context.language,
                    "status": result.get("status"),
                    "pass": result.get("pass"),
                    "repair_index": result.get("repair_index"),
                    "validated_attempt_index": result.get("validated_attempt_index"),
                    "result_json": str(output_path),
                    "work_dir": result.get("work_dir"),
                }
            )
        except Exception as exc:
            batch_summary["errors"] += 1
            error_payload = {
                "status": "error",
                "pass": False,
                "programId": context.program_id,
                "task_name": context.task_name,
                "language": context.language,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(output_path, error_payload)
            batch_summary["programs"].append(
                {
                    "programId": context.program_id,
                    "task_name": context.task_name,
                    "language": context.language,
                    "status": "error",
                    "result_json": str(output_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[error] programId={context.program_id}: {exc}", file=sys.stderr)
            if args.stop_on_error:
                write_batch_summary(batch_summary_path, batch_summary)
                print_final_batch_summary(batch_summary, batch_summary_path)
                return 1

        write_batch_summary(batch_summary_path, batch_summary)
        if args.sleep_seconds > 0:
            sleep(args.sleep_seconds)

    print_final_batch_summary(batch_summary, batch_summary_path)
    return 1 if batch_summary["errors"] and args.stop_on_error else 0


def print_final_batch_summary(summary: dict[str, Any], summary_path: Path) -> None:
    print(
        "\nFinished: "
        f"completed={summary['completed']}, skipped={summary['skipped']}, "
        f"passed={summary['passed']}, failed={summary['failed']}, "
        f"errors={summary['errors']}, total={summary['total_programs']}"
    )
    print(f"Batch summary written to {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
