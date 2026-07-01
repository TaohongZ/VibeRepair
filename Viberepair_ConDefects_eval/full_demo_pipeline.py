from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ConDefects.util.get_faulty_function_by_program_id import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    ProgramFunctionLookupError,
    _find_method_ranges,
    _find_smallest_enclosing_python_function,
    _python_node_start_line,
)
from ConDefects.util.get_task_sections import (  # noqa: E402
    TaskSectionFetchError,
    get_task_sections,
)
from eval_ConDefects.demo_pipeline import (  # noqa: E402
    ProgramContext,
    build_fix_specification_input,
    resolve_program_context,
)
from viberepair.langchain_support import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
)
CONDEFECTS_ROOT = REPO_ROOT / "ConDefects"
DEFAULT_WORK_ROOT = Path("/tmp") / "viberepair_full_demo_pipeline"
DEFAULT_MAX_LOOPS = 5
MAX_TEST_TEXT_CHARS = 4000
MAX_FAILURE_CASES_FOR_FEEDBACK = 3
TEST_TIMEOUT_SECONDS = 60
PYTHON3_EXECUTABLE = os.environ.get("CONDEFECTS_PYTHON3", "python3")
OCHIAI_FL_FILENAME = "Ochiai_FL.txt"
OCHIAI_REFERENCE_TEXT = (
    "Ochiai fault localization identifies this line as having the highest suspiciousness "
    "score (T0). It may be related to the bug, but this is only a reference and should "
    "not be considered conclusive evidence."
)


@dataclass(slots=True, frozen=True)
class OchiaiFunctionLocation:
    program_id: str
    program_dir: Path
    source_path: Path
    task_name: str
    language: str
    fault_line_number: int
    start_line_index: int
    end_line_index: int
    faulty_function: str


@dataclass(slots=True, frozen=True)
class ValidationAttempt:
    attempt_index: int
    attempt_dir: Path
    verification_dir: Path
    temp_dataset_root: Path
    temp_program_dir: Path
    temp_source_path: Path
    passed: bool
    pass_rate: float | None
    records: list[dict[str, Any]]
    error: str | None = None


@dataclass(slots=True, frozen=True)
class StructureAdapterAttempt:
    adapted_fix_code: str
    changed: bool = False
    reason: str | None = None
    raw_response: str = ""
    usage: Any = None
    estimated_cost_usd: float | None = None
    deterministic_fallback_applied: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full iterative VibeRepair demo for one ConDefects program id: "
            "Ochiai function -> flawed specification -> fixed specification -> fix code "
            "-> tests -> verification feedback loop."
        )
    )
    parser.add_argument("program_id", help="ConDefects program id, for example 53706270")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Path to the ConDefects dataset directory. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Chat model name used by VibeRepair stages. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature used by VibeRepair stages. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=DEFAULT_MAX_LOOPS,
        help=f"Maximum fix/test attempts. Default: {DEFAULT_MAX_LOOPS}",
    )
    parser.add_argument(
        "--work-root",
        default=str(DEFAULT_WORK_ROOT),
        help="Root directory used for temporary full-demo runs.",
    )
    parser.add_argument(
        "--test-limit",
        action="append",
        dest="test_limit",
        help="Optional test case filename to run. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to copy the final summary JSON.",
    )
    parser.add_argument(
        "--disable-structure-adapter",
        action="store_true",
        help="Skip the post-generation fix-code structure adapter stage.",
    )
    return parser.parse_args()


def import_viberepair_full_pipeline_functions() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from viberepair.modules.buggy_code_to_flawed_specification import (
            generate_flawed_specification_result,
        )
        from viberepair.modules.fix_code_structure_adapter import (
            FixCodeStructureAdapter,
        )
        from viberepair.modules.fix_specification import FixSpecificationSession
        from viberepair.modules.fix_specification_and_buggy_code_to_fix_code import (
            FixCodeGenerator,
        )
        from viberepair.modules.verification_bundle_to_test_failure_specifications import (
            generate_test_failure_specifications_result,
        )
    except ImportError as exc:
        raise ImportError(
            "VibeRepair full demo requires langchain>=1.0.0, langchain-core, "
            "and langchain-openai. Install them with: "
            "pip install -U langchain langchain-core langchain-openai."
        ) from exc

    return (
        generate_flawed_specification_result,
        FixSpecificationSession,
        FixCodeGenerator,
        FixCodeStructureAdapter,
        generate_test_failure_specifications_result,
    )


def locate_ochiai_faulty_function(context: ProgramContext) -> OchiaiFunctionLocation:
    ochiai_path = context.program_dir / OCHIAI_FL_FILENAME
    if not ochiai_path.exists():
        raise ProgramFunctionLookupError(
            f"Program {context.program_id} does not contain {OCHIAI_FL_FILENAME}: "
            f"{context.program_dir}"
        )

    fault_line_number = _read_ochiai_fault_line_number(ochiai_path)
    java_path = context.program_dir / "faultyVersion.java"
    if java_path.exists():
        return _locate_java_ochiai_function(context, java_path, fault_line_number)

    python_path = context.program_dir / "faultyVersion.py"
    if python_path.exists():
        return _locate_python_ochiai_function(context, python_path, fault_line_number)

    raise ProgramFunctionLookupError(
        f"Program {context.program_id} does not contain faultyVersion.java or faultyVersion.py: "
        f"{context.program_dir}"
    )


def _locate_java_ochiai_function(
    context: ProgramContext,
    java_path: Path,
    fault_line_number: int,
) -> OchiaiFunctionLocation:
    lines = java_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not 1 <= fault_line_number <= len(lines):
        raise ProgramFunctionLookupError(
            f"Ochiai fault line {fault_line_number} is out of range for {java_path}"
        )

    target_index = fault_line_number - 1
    for start_index, end_index in _find_method_ranges(lines):
        if start_index <= target_index <= end_index:
            return OchiaiFunctionLocation(
                program_id=context.program_id,
                program_dir=context.program_dir,
                source_path=java_path,
                task_name=context.task_name,
                language=context.language,
                fault_line_number=fault_line_number,
                start_line_index=start_index,
                end_line_index=end_index,
                faulty_function=_format_lines_with_ochiai_comment(
                    lines[start_index : end_index + 1],
                    target_index - start_index,
                    context.language,
                ),
            )

    raise ProgramFunctionLookupError(
        f"Could not find an enclosing Java method for program {context.program_id} "
        f"at Ochiai line {fault_line_number}"
    )


def _locate_python_ochiai_function(
    context: ProgramContext,
    python_path: Path,
    fault_line_number: int,
) -> OchiaiFunctionLocation:
    source = python_path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    if not 1 <= fault_line_number <= len(lines):
        raise ProgramFunctionLookupError(
            f"Ochiai fault line {fault_line_number} is out of range for {python_path}"
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramFunctionLookupError(
            f"Could not parse faultyVersion.py for program {context.program_id}: {exc}"
        ) from exc

    target_node = _find_smallest_enclosing_python_function(tree, fault_line_number)
    if target_node is None:
        start_line_index = 0
        end_line_index = len(lines) - 1
    else:
        start_line_index = _python_node_start_line(target_node) - 1
        end_line = getattr(target_node, "end_lineno", None)
        if end_line is None:
            raise ProgramFunctionLookupError(
                f"Could not determine the end line of the enclosing Python function for "
                f"program {context.program_id}"
            )
        end_line_index = end_line - 1

    return OchiaiFunctionLocation(
        program_id=context.program_id,
        program_dir=context.program_dir,
        source_path=python_path,
        task_name=context.task_name,
        language=context.language,
        fault_line_number=fault_line_number,
        start_line_index=start_line_index,
        end_line_index=end_line_index,
        faulty_function=_format_lines_with_ochiai_comment(
            lines[start_line_index : end_line_index + 1],
            fault_line_number - 1 - start_line_index,
            context.language,
        ),
    )


def _read_ochiai_fault_line_number(path: Path) -> int:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ProgramFunctionLookupError(f"Invalid {OCHIAI_FL_FILENAME} value in {path}: {raw!r}") from exc


def _format_lines_with_ochiai_comment(
    lines: list[str],
    fault_line_index: int,
    language: str,
) -> str:
    if not 0 <= fault_line_index < len(lines):
        raise ProgramFunctionLookupError("Ochiai fault line is outside the extracted function")

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


def run_full_demo_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_loops < 1:
        raise ValueError("--max-loops must be at least 1")

    (
        generate_flawed_specification_result,
        FixSpecificationSession,
        FixCodeGenerator,
        FixCodeStructureAdapter,
        generate_test_failure_specifications_result,
    ) = import_viberepair_full_pipeline_functions()

    context = resolve_program_context(args.program_id, args.dataset_root)
    location = locate_ochiai_faulty_function(context)
    run_dir = make_run_dir(args.work_root, context.program_id)
    use_structure_adapter = bool(getattr(args, "use_structure_adapter", True))
    if getattr(args, "disable_structure_adapter", False):
        use_structure_adapter = False

    model = build_chat_model(model_name=args.model, temperature=args.temperature)
    task_information = get_task_sections(context.task_name)

    print(f"Program ID: {context.program_id}")
    print(f"Task: {context.task_name}")
    print(f"Language: {context.language}")
    print(f"Ochiai line: {location.fault_line_number}")
    print(f"Run directory: {run_dir}")
    print(f"Structure adapter enabled: {use_structure_adapter}")

    print("\nGenerating flawed specification")
    flawed_specification_result = generate_flawed_specification_result(
        buggy_code=location.faulty_function,
        model_name=args.model,
        temperature=args.temperature,
        model=model,
    )
    print_block("Flawed Specification", flawed_specification_result.flawed_specification)

    fix_specification_session = FixSpecificationSession(
        model_name=args.model,
        temperature=args.temperature,
        model=model,
    )
    fix_code_generator = FixCodeGenerator(
        model_name=args.model,
        temperature=args.temperature,
        model=model,
    )
    fix_code_structure_adapter = (
        FixCodeStructureAdapter(
            model_name=args.model,
            temperature=args.temperature,
            model=model,
        )
        if use_structure_adapter
        else None
    )

    fix_specification_input = build_fix_specification_input(
        task_information=task_information,
        flawed_specification=flawed_specification_result.flawed_specification,
    )
    print("\nNow fixing specification")
    current_specification_result = fix_specification_session.start(fix_specification_input)
    print_block("Fix Specification", current_specification_result.fix_specification)

    print("\nGenerating fix code")
    current_fix_code_result = fix_code_generator.generate(
        fix_specification=current_specification_result.fix_specification,
        buggy_code=location.faulty_function,
    )
    current_structure_adapter_result = prepare_fix_code_for_validation(
        fix_code_structure_adapter=fix_code_structure_adapter,
        use_structure_adapter=use_structure_adapter,
        buggy_code=location.faulty_function,
        fix_code=current_fix_code_result.replacement_code,
        fix_specification=current_specification_result.fix_specification,
    )

    summary: dict[str, Any] = {
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "programId": context.program_id,
        "task_name": context.task_name,
        "language": context.language,
        "program_dir": str(context.program_dir),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "work_dir": str(run_dir),
        "model": args.model,
        "temperature": args.temperature,
        "max_loops": args.max_loops,
        "use_structure_adapter": use_structure_adapter,
        "fault_location_source": OCHIAI_FL_FILENAME,
        "fault_line_number": location.fault_line_number,
        "source_range": {
            "source_path": str(location.source_path),
            "start_line": location.start_line_index + 1,
            "end_line": location.end_line_index + 1,
        },
        "faulty_function": location.faulty_function,
        "flawed_specification": flawed_specification_result.flawed_specification,
        "flawed_specification_usage": _usage_dict(flawed_specification_result.usage),
        "flawed_specification_estimated_cost": flawed_specification_result.estimated_cost_usd,
        "attempts": [],
    }
    write_json(run_dir / "summary.json", summary)
    write_text(run_dir / "task_information.txt", task_information)
    write_text(run_dir / "faulty_function_with_ochiai_comment.txt", location.faulty_function)
    write_text(run_dir / "flawed_specification.txt", flawed_specification_result.flawed_specification)

    for attempt_index in range(1, args.max_loops + 1):
        print(f"\nTesting attempt {attempt_index}/{args.max_loops}")
        validation = validate_fix_code_attempt(
            location=location,
            fix_code=current_structure_adapter_result.adapted_fix_code,
            run_dir=run_dir,
            attempt_index=attempt_index,
            test_limit=args.test_limit,
        )

        attempt_payload = {
            "attempt_index": attempt_index,
            "status": "pass" if validation.passed else "fail",
            "pass": validation.passed,
            "pass_rate": validation.pass_rate,
            "attempt_dir": str(validation.attempt_dir),
            "verification_dir": str(validation.verification_dir),
            "temp_dataset_root": str(validation.temp_dataset_root),
            "temp_program_dir": str(validation.temp_program_dir),
            "temp_source_path": str(validation.temp_source_path),
            "test_records": validation.records,
            "test_error": validation.error,
            "fix_specification_reasoning": {
                "intended_behavior": current_specification_result.intended_behavior,
                "diagnosis_and_repair_directions": (
                    current_specification_result.diagnosis_and_repair_directions
                ),
                "raw_response": current_specification_result.raw_response,
            },
            "fixed_specification": current_specification_result.fix_specification,
            "fix_specification_usage": _usage_dict(current_specification_result.usage),
            "fix_specification_estimated_cost": current_specification_result.estimated_cost_usd,
            "fix_code": current_structure_adapter_result.adapted_fix_code,
            "fix_code_raw_response": current_fix_code_result.raw_response,
            "fix_code_usage": _usage_dict(current_fix_code_result.usage),
            "fix_code_estimated_cost": current_fix_code_result.estimated_cost_usd,
        }
        if use_structure_adapter:
            attempt_payload.update(
                {
                    "fix_code_before_structure_adapter": current_fix_code_result.replacement_code,
                    "fix_code_structure_adapter_changed": current_structure_adapter_result.changed,
                    "fix_code_structure_adapter_reason": current_structure_adapter_result.reason,
                    "fix_code_structure_adapter_raw_response": current_structure_adapter_result.raw_response,
                    "fix_code_structure_adapter_usage": _usage_dict(current_structure_adapter_result.usage),
                    "fix_code_structure_adapter_estimated_cost": (
                        current_structure_adapter_result.estimated_cost_usd
                    ),
                    "fix_code_structure_adapter_deterministic_fallback_applied": (
                        current_structure_adapter_result.deterministic_fallback_applied
                    ),
                }
            )
        summary["attempts"].append(attempt_payload)
        write_json(run_dir / "summary.json", summary)

        if validation.passed:
            print(f"[pass] attempt {attempt_index} passed all tests")
            summary["status"] = "success"
            summary["validated_attempt_index"] = attempt_index
            summary["final_fix_code"] = current_structure_adapter_result.adapted_fix_code
            write_json(run_dir / "summary.json", summary)
            copy_summary_if_requested(summary, args.output_json)
            return summary

        print(f"[fail] attempt {attempt_index} pass_rate={validation.pass_rate}")
        if attempt_index >= args.max_loops:
            break

        print("\nGenerating Test Failure Specification")
        feedback_result = generate_test_failure_specifications_result(
            verification_result_path=str(validation.verification_dir),
            model_name=args.model,
            temperature=args.temperature,
            workspace_root=str(REPO_ROOT),
            buggy_code=location.faulty_function,
            fix_code=current_structure_adapter_result.adapted_fix_code,
            fix_specification=current_specification_result.fix_specification,
        )
        print_block(
            "Test Failure Specification",
            feedback_result.test_failure_specifications,
        )
        write_text(
            validation.verification_dir / "test_failure_specifications.txt",
            feedback_result.test_failure_specifications,
        )
        attempt_payload["test_failure_specifications"] = feedback_result.test_failure_specifications
        attempt_payload["test_failure_specifications_usage"] = _usage_dict(
            feedback_result.usage
        )
        attempt_payload["test_failure_specifications_estimated_cost"] = (
            feedback_result.estimated_cost_usd
        )
        write_json(run_dir / "summary.json", summary)

        print("\nNow fixing specification")
        current_specification_result = fix_specification_session.add_information(
            build_verification_feedback_input(
                attempt_index=attempt_index,
                validation=validation,
                test_failure_specifications=feedback_result.test_failure_specifications,
            )
        )
        print_block("Fix Specification", current_specification_result.fix_specification)

        print("\nGenerating fix code")
        current_fix_code_result = fix_code_generator.generate(
            fix_specification=current_specification_result.fix_specification,
            buggy_code=location.faulty_function,
        )
        current_structure_adapter_result = prepare_fix_code_for_validation(
            fix_code_structure_adapter=fix_code_structure_adapter,
            use_structure_adapter=use_structure_adapter,
            buggy_code=location.faulty_function,
            fix_code=current_fix_code_result.replacement_code,
            fix_specification=current_specification_result.fix_specification,
        )

    print(f"[failed] no generated fix passed within {args.max_loops} attempts")
    summary["status"] = "failed"
    summary["validated_attempt_index"] = None
    summary["final_fix_code"] = current_structure_adapter_result.adapted_fix_code
    write_json(run_dir / "summary.json", summary)
    copy_summary_if_requested(summary, args.output_json)
    return summary



def prepare_fix_code_for_validation(
    fix_code_structure_adapter: Any | None,
    use_structure_adapter: bool,
    buggy_code: str,
    fix_code: str,
    fix_specification: str,
) -> StructureAdapterAttempt:
    if not use_structure_adapter:
        print_block("Fix Code", fix_code)
        return StructureAdapterAttempt(adapted_fix_code=fix_code)

    if fix_code_structure_adapter is None:
        raise ValueError("Structure adapter is enabled but was not initialized.")

    print_block("Fix Code (Before Structure Adapter)", fix_code)
    print("\nAdapting fix code structure")
    result = fix_code_structure_adapter.adapt(
        buggy_code=buggy_code,
        fix_code=fix_code,
        fix_specification=fix_specification,
    )
    print_block("Structure-Compatible Fix Code", result.adapted_fix_code)
    return StructureAdapterAttempt(
        adapted_fix_code=result.adapted_fix_code,
        changed=result.changed,
        reason=result.reason,
        raw_response=result.raw_response,
        usage=result.usage,
        estimated_cost_usd=result.estimated_cost_usd,
        deterministic_fallback_applied=result.deterministic_fallback_applied,
    )

def make_run_dir(work_root: str | Path, program_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{program_id}_{timestamp}"
    if not base.exists():
        base.mkdir(parents=True)
        return base

    suffix = 1
    while True:
        candidate = root / f"{program_id}_{timestamp}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def validate_fix_code_attempt(
    location: OchiaiFunctionLocation,
    fix_code: str,
    run_dir: Path,
    attempt_index: int,
    test_limit: list[str] | None,
) -> ValidationAttempt:
    attempt_dir = run_dir / f"attempt_{attempt_index:02d}"
    verification_dir = attempt_dir / "verification_bundle"
    temp_dataset_root = attempt_dir / "dataset"
    temp_program_dir = temp_dataset_root / location.task_name / location.language / location.program_id

    if verification_dir.exists():
        shutil.rmtree(verification_dir)
    verification_dir.mkdir(parents=True, exist_ok=True)

    if temp_program_dir.exists():
        shutil.rmtree(temp_program_dir)
    shutil.copytree(location.program_dir, temp_program_dir)

    temp_source_path = temp_program_dir / location.source_path.name
    replace_source_range(
        source_path=temp_source_path,
        start_line_index=location.start_line_index,
        end_line_index=location.end_line_index,
        replacement=fix_code,
    )

    failed_cases, error_cases, pass_rate, records = run_minimal_test_cases(
        location=location,
        temp_source_path=temp_source_path,
        attempt_dir=attempt_dir,
        test_limit=test_limit,
    )
    write_minimal_verification_bundle(
        verification_dir=verification_dir,
        failed_cases=failed_cases,
        error_cases=error_cases,
    )

    passed = pass_rate == 1.0 and not error_cases
    error_text = build_error_text(error_cases)

    return ValidationAttempt(
        attempt_index=attempt_index,
        attempt_dir=attempt_dir,
        verification_dir=verification_dir,
        temp_dataset_root=temp_dataset_root,
        temp_program_dir=temp_program_dir,
        temp_source_path=temp_source_path,
        passed=passed,
        pass_rate=pass_rate if isinstance(pass_rate, float) else None,
        records=records,
        error=error_text,
    )


def run_minimal_test_cases(
    location: OchiaiFunctionLocation,
    temp_source_path: Path,
    attempt_dir: Path,
    test_limit: list[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None, list[dict[str, Any]]]:
    deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
    test_names = resolve_test_case_names(location.task_name, test_limit)
    if not test_names:
        error_cases = [
            build_error_case(
                name=f"{location.task_name}/test_discovery",
                phase="test_discovery",
                message=f"No ConDefects test cases were found for task {location.task_name}.",
            )
        ]
        return [], error_cases, None, build_validation_records(location, None, [], error_cases, 0)

    if location.language == "Java":
        failed_cases, error_cases, pass_count = run_java_test_cases(
            location=location,
            temp_source_path=temp_source_path,
            attempt_dir=attempt_dir,
            test_names=test_names,
            deadline=deadline,
        )
    else:
        failed_cases, error_cases, pass_count = run_python_test_cases(
            location=location,
            temp_source_path=temp_source_path,
            test_names=test_names,
            deadline=deadline,
        )

    pass_rate = pass_count / len(test_names) if test_names else None
    records = build_validation_records(
        location=location,
        pass_rate=pass_rate,
        failed_cases=failed_cases,
        error_cases=error_cases,
        tests_run=len(test_names),
    )
    return failed_cases, error_cases, pass_rate, records


def run_python_test_cases(
    location: OchiaiFunctionLocation,
    temp_source_path: Path,
    test_names: list[str],
    deadline: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    source_lines = temp_source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    executable = "python2" if any("print " in line for line in source_lines) else PYTHON3_EXECUTABLE
    failed_cases: list[dict[str, Any]] = []
    error_cases: list[dict[str, Any]] = []
    pass_count = 0

    for test_name in test_names:
        remaining_timeout = remaining_validation_seconds(deadline)
        if remaining_timeout <= 0:
            error_cases.append(build_validation_timeout_case(location))
            break

        input_path = resolve_test_case_path(location.task_name, "in", test_name)
        expected_output_path = resolve_test_case_path(location.task_name, "out", test_name)
        if input_path is None or expected_output_path is None:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="test_setup",
                    message="Could not resolve input or expected output file for this test case.",
                )
            )
            continue

        input_text = read_limited_text(input_path, limit=None)
        expected_output = read_limited_text(expected_output_path, limit=None)
        try:
            completed = subprocess.run(
                [executable, str(temp_source_path)],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=remaining_timeout,
            )
        except subprocess.TimeoutExpired:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="timeout",
                    message=(
                        "Python execution timed out before the validation attempt "
                        f"{TEST_TIMEOUT_SECONDS:g}-second budget was exhausted."
                    ),
                )
            )
            break
        except Exception as exc:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="runtime",
                    message=f"Python execution failed: {exc}",
                )
            )
            continue

        actual_output = completed.stdout
        if completed.returncode != 0:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="runtime",
                    message="Python fix code exited with a non-zero status.",
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
        elif outputs_match(actual_output, expected_output):
            pass_count += 1
        else:
            failed_cases.append(
                build_output_failure_case(
                    location=location,
                    test_name=test_name,
                    input_text=input_text,
                    expected_output=expected_output,
                    actual_output=actual_output,
                )
            )

    return failed_cases, error_cases, pass_count


def run_java_test_cases(
    location: OchiaiFunctionLocation,
    temp_source_path: Path,
    attempt_dir: Path,
    test_names: list[str],
    deadline: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    execution_dir = attempt_dir / "execution"
    if execution_dir.exists():
        shutil.rmtree(execution_dir)
    execution_dir.mkdir(parents=True, exist_ok=True)

    main_java_path = execution_dir / "Main.java"
    shutil.copy(temp_source_path, main_java_path)
    remaining_timeout = remaining_validation_seconds(deadline)
    if remaining_timeout <= 0:
        shutil.rmtree(execution_dir, ignore_errors=True)
        return [], [build_validation_timeout_case(location)], 0

    try:
        compile_result = subprocess.run(
            ["javac", "Main.java"],
            cwd=execution_dir,
            capture_output=True,
            text=True,
            timeout=remaining_timeout,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(execution_dir, ignore_errors=True)
        error_cases = [
            build_error_case(
                name=f"{location.task_name}/compile_timeout",
                phase="timeout",
                message=(
                    "Java compilation timed out after "
                    f"{TEST_TIMEOUT_SECONDS:g} seconds."
                ),
            )
        ]
        return [], error_cases, 0

    if compile_result.returncode != 0:
        shutil.rmtree(execution_dir, ignore_errors=True)
        error_cases = [
            build_error_case(
                name=f"{location.task_name}/compile",
                phase="compile",
                message="Java fix code failed to compile.",
                returncode=compile_result.returncode,
                stdout=compile_result.stdout,
                stderr=compile_result.stderr,
            )
        ]
        return [], error_cases, 0

    failed_cases: list[dict[str, Any]] = []
    error_cases: list[dict[str, Any]] = []
    pass_count = 0

    for test_name in test_names:
        remaining_timeout = remaining_validation_seconds(deadline)
        if remaining_timeout <= 0:
            error_cases.append(build_validation_timeout_case(location))
            break

        input_path = resolve_test_case_path(location.task_name, "in", test_name)
        expected_output_path = resolve_test_case_path(location.task_name, "out", test_name)
        if input_path is None or expected_output_path is None:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="test_setup",
                    message="Could not resolve input or expected output file for this test case.",
                )
            )
            continue

        input_text = read_limited_text(input_path, limit=None)
        expected_output = read_limited_text(expected_output_path, limit=None)
        try:
            completed = subprocess.run(
                ["java", "Main"],
                cwd=execution_dir,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=remaining_timeout,
            )
        except subprocess.TimeoutExpired:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="timeout",
                    message=(
                        "Java execution timed out before the validation attempt "
                        f"{TEST_TIMEOUT_SECONDS:g}-second budget was exhausted."
                    ),
                )
            )
            break
        except Exception as exc:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="runtime",
                    message=f"Java execution failed: {exc}",
                )
            )
            continue

        actual_output = completed.stdout
        if completed.returncode != 0:
            error_cases.append(
                build_error_case(
                    name=f"{location.task_name}/{test_name}",
                    phase="runtime",
                    message="Java fix code exited with a non-zero status.",
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
        elif outputs_match(actual_output, expected_output):
            pass_count += 1
        else:
            failed_cases.append(
                build_output_failure_case(
                    location=location,
                    test_name=test_name,
                    input_text=input_text,
                    expected_output=expected_output,
                    actual_output=actual_output,
                )
            )

    shutil.rmtree(execution_dir, ignore_errors=True)
    return failed_cases, error_cases, pass_count


def write_minimal_verification_bundle(
    verification_dir: Path,
    failed_cases: list[dict[str, Any]],
    error_cases: list[dict[str, Any]],
) -> None:
    if error_cases:
        write_json(
            verification_dir / "error.json",
            {"tests": error_cases[:MAX_FAILURE_CASES_FOR_FEEDBACK]},
        )
        return
    if failed_cases:
        write_json(
            verification_dir / "test_failures.json",
            {"tests": failed_cases[:MAX_FAILURE_CASES_FOR_FEEDBACK]},
        )


def build_output_failure_case(
    location: OchiaiFunctionLocation,
    test_name: str,
    input_text: str,
    expected_output: str,
    actual_output: str,
) -> dict[str, Any]:
    newline = chr(10)
    truncated_input = truncate_text(input_text)
    truncated_expected = truncate_text(expected_output)
    truncated_actual = truncate_text(actual_output)
    return {
        "name": f"{location.task_name}/{test_name}",
        "status": "failed",
        "message": (
            "ConDefects test case output did not match the expected output."
            f"{newline}{newline}Input:{newline}{truncated_input}"
            f"{newline}{newline}Expected output:{newline}{truncated_expected}"
            f"{newline}{newline}Actual output:{newline}{truncated_actual}"
        ),
        "test_case": test_name,
        "input": truncated_input,
        "expected_output": truncated_expected,
        "actual_output": truncated_actual,
    }


def remaining_validation_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def build_validation_timeout_case(location: OchiaiFunctionLocation) -> dict[str, Any]:
    return build_error_case(
        name=f"{location.task_name}/validation_timeout",
        phase="timeout",
        message=(
            "Validation timed out after "
            f"{TEST_TIMEOUT_SECONDS:g} seconds before all test cases finished."
        ),
    )


def build_error_case(
    name: str,
    phase: str,
    message: str,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    newline = chr(10)
    details = [message]
    if returncode is not None:
        details.append(f"returncode = {returncode}")
    if stdout:
        details.append(f"stdout:{newline}{truncate_text(stdout)}")
    if stderr:
        details.append(f"stderr:{newline}{truncate_text(stderr)}")
    return {
        "name": name,
        "status": "error",
        "phase": phase,
        "message": f"{newline}{newline}".join(details),
    }


def build_error_text(error_cases: list[dict[str, Any]]) -> str | None:
    if not error_cases:
        return None
    separator = f"{chr(10)}{chr(10)}"
    return separator.join(str(case.get("message", "")).rstrip() for case in error_cases).strip()


def build_validation_records(
    location: OchiaiFunctionLocation,
    pass_rate: float | None,
    failed_cases: list[dict[str, Any]],
    error_cases: list[dict[str, Any]],
    tests_run: int,
) -> list[dict[str, Any]]:
    return [
        {
            "task": location.task_name,
            "language": location.language,
            "program_id": location.program_id,
            "pass_rate": pass_rate,
            "tests_run": tests_run,
            "failed_tests": [case.get("test_case") or case.get("name") for case in failed_cases],
            "error_tests": [case.get("name") for case in error_cases],
        }
    ]


def resolve_test_case_names(task_name: str, test_limit: list[str] | None) -> list[str]:
    input_dir = resolve_test_case_dir(task_name, "in")
    if input_dir is None:
        return []
    test_names = sorted(path.name for path in input_dir.iterdir() if path.is_file())
    if test_limit is not None:
        wanted = set(test_limit)
        test_names = [name for name in test_names if name in wanted]
    return test_names


def resolve_test_case_dir(task_name: str, direction: str) -> Path | None:
    contest_name, problem_name = task_name.split("_", 1)
    contest_path = resolve_case_insensitive_child(CONDEFECTS_ROOT / "Test", contest_name)
    if not contest_path.is_dir():
        return None
    for candidate in (problem_name.upper(), problem_name, "Ex"):
        case_dir = resolve_case_insensitive_child(contest_path, candidate) / direction
        if case_dir.exists():
            return case_dir
    return None


def outputs_match(actual_output: str, expected_output: str) -> bool:
    return actual_output.splitlines() == expected_output.splitlines()


def replace_source_range(
    source_path: str | Path,
    start_line_index: int,
    end_line_index: int,
    replacement: str,
) -> None:
    source_path = Path(source_path)
    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    if not 0 <= start_line_index <= end_line_index < len(lines):
        raise ProgramFunctionLookupError("Ochiai function range is outside source file")

    suffix = "".join(lines[end_line_index + 1 :])
    replacement_text = replacement
    if suffix and replacement_text and not replacement_text.endswith(("\n", "\r")):
        replacement_text += infer_line_ending(lines[end_line_index])

    patched = "".join(lines[:start_line_index]) + replacement_text + suffix
    source_path.write_text(patched, encoding="utf-8")


def infer_line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    return "\n"


def resolve_test_case_path(task_name: str, direction: str, test_name: str) -> Path | None:
    contest_name, problem_name = task_name.split("_", 1)
    contest_path = resolve_case_insensitive_child(CONDEFECTS_ROOT / "Test", contest_name)
    if not contest_path.is_dir():
        return None
    for candidate in (problem_name.upper(), problem_name, "Ex"):
        case_dir = resolve_case_insensitive_child(contest_path, candidate) / direction
        test_path = case_dir / test_name
        if test_path.exists():
            return test_path
    return None


def resolve_case_insensitive_child(parent: Path, child: str) -> Path:
    if not parent.is_dir():
        return parent / child
    for entry in parent.iterdir():
        if entry.name.lower() == child.lower():
            return entry
    return parent / child


def truncate_text(text: str, limit: int = MAX_TEST_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}{chr(10)}{chr(10)}[truncated {omitted} characters]"


def read_limited_text(path: str | Path, limit: int | None = MAX_TEST_TEXT_CHARS) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if limit is None:
        return text
    return truncate_text(text, limit=limit)


def build_verification_feedback_input(
    attempt_index: int,
    validation: ValidationAttempt,
    test_failure_specifications: str,
) -> str:
    return (
        f"Verification feedback after failed generated fix attempt {attempt_index}.\n"
        f"Verification bundle: {validation.verification_dir}\n"
        f"Pass rate: {validation.pass_rate}\n\n"
        "Test Failure Specifications\n"
        f"{test_failure_specifications.strip()}"
    )


def _usage_dict(usage: Any) -> dict[str, int | None]:
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def copy_summary_if_requested(summary: dict[str, Any], output_json: str | None) -> None:
    if not output_json:
        return
    write_json(output_json, summary)


def print_block(title: str, content: str) -> None:
    print(f"\n=== {title} ===")
    print((content or "").rstrip())


def print_final_summary(summary: dict[str, Any]) -> None:
    print("\n=== Full Demo Summary ===")
    print(f"Status: {summary['status']}")
    print(f"Program ID: {summary['programId']}")
    print(f"Task: {summary['task_name']}")
    print(f"Language: {summary['language']}")
    print(f"Attempts: {len(summary['attempts'])}/{summary['max_loops']}")
    print(f"Work dir: {summary['work_dir']}")
    print(f"Summary JSON: {Path(summary['work_dir']) / 'summary.json'}")


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    try:
        summary = run_full_demo_pipeline(args)
    except (
        EnvironmentError,
        ImportError,
        ProgramFunctionLookupError,
        TaskSectionFetchError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
    write_json(Path(summary["work_dir"]) / "summary.json", summary)
    copy_summary_if_requested(summary, args.output_json)
    print_final_summary(summary)
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
