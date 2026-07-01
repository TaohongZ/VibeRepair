from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ConDefects.util.get_task_sections import TaskSectionFetchError  # noqa: E402
import full_demo_pipeline as full_demo  # noqa: E402
from viberepair.langchain_support import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
)


DEFAULT_DATASET_ROOT = full_demo.DEFAULT_DATASET_ROOT
CONDEFECTS_ROOT = full_demo.CONDEFECTS_ROOT
DEFAULT_WORK_ROOT = Path("/tmp") / "viberepair_full_demo_plus_reasoning_pipeline"
DEFAULT_MAX_LOOPS = full_demo.DEFAULT_MAX_LOOPS
TEST_TIMEOUT_SECONDS = full_demo.TEST_TIMEOUT_SECONDS
OCHIAI_FL_FILENAME = full_demo.OCHIAI_FL_FILENAME
OCHIAI_REFERENCE_TEXT = full_demo.OCHIAI_REFERENCE_TEXT

OchiaiFunctionLocation = full_demo.OchiaiFunctionLocation
ValidationAttempt = full_demo.ValidationAttempt
StructureAdapterAttempt = full_demo.StructureAdapterAttempt
ProgramFunctionLookupError = full_demo.ProgramFunctionLookupError

_find_method_ranges = full_demo._find_method_ranges
_find_smallest_enclosing_python_function = full_demo._find_smallest_enclosing_python_function
_python_node_start_line = full_demo._python_node_start_line
locate_ochiai_faulty_function = full_demo.locate_ochiai_faulty_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full iterative VibeRepair demo with the reasoning module inserted before "
            "fix_specification: Ochiai function -> flawed specification -> reasoning suggestion "
            "-> fixed specification -> fix code -> tests -> verification feedback loop."
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


def import_viberepair_full_pipeline_functions() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        (
            generate_flawed_specification_result,
            FixSpecificationSession,
            FixCodeGenerator,
            FixCodeStructureAdapter,
            generate_test_failure_specifications_result,
        ) = full_demo.import_viberepair_full_pipeline_functions()
        from viberepair.modules.reasoning import generate_repair_suggestion_result
    except ImportError as exc:
        raise ImportError(
            "VibeRepair full demo plus reasoning requires langchain>=1.0.0, "
            "langchain-core, and langchain-openai. Install them with: "
            "pip install -U langchain langchain-core langchain-openai."
        ) from exc

    return (
        generate_flawed_specification_result,
        generate_repair_suggestion_result,
        FixSpecificationSession,
        FixCodeGenerator,
        FixCodeStructureAdapter,
        generate_test_failure_specifications_result,
    )


def build_fix_specification_input_with_reasoning(
    task_information: str,
    flawed_specification: str,
    reasoning_repair_suggestion: str,
) -> str:
    return (
        full_demo.build_fix_specification_input(
            task_information=task_information,
            flawed_specification=flawed_specification,
        )
        + "\n\nReasoning Module Repair Suggestion\n"
        + reasoning_repair_suggestion.strip()
    )


def run_full_demo_plus_reasoning_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_loops < 1:
        raise ValueError("--max-loops must be at least 1")

    full_demo.TEST_TIMEOUT_SECONDS = TEST_TIMEOUT_SECONDS
    full_demo.CONDEFECTS_ROOT = CONDEFECTS_ROOT

    (
        generate_flawed_specification_result,
        generate_repair_suggestion_result,
        FixSpecificationSession,
        FixCodeGenerator,
        FixCodeStructureAdapter,
        generate_test_failure_specifications_result,
    ) = import_viberepair_full_pipeline_functions()

    context = full_demo.resolve_program_context(args.program_id, args.dataset_root)
    location = locate_ochiai_faulty_function(context)
    run_dir = full_demo.make_run_dir(args.work_root, context.program_id)
    use_structure_adapter = bool(getattr(args, "use_structure_adapter", True))
    if getattr(args, "disable_structure_adapter", False):
        use_structure_adapter = False

    model = build_chat_model(model_name=args.model, temperature=args.temperature)
    task_information = full_demo.get_task_sections(context.task_name)

    print(f"Program ID: {context.program_id}")
    print(f"Task: {context.task_name}")
    print(f"Language: {context.language}")
    print(f"Ochiai line: {location.fault_line_number}")
    print(f"Run directory: {run_dir}")
    print(f"Structure adapter enabled: {use_structure_adapter}")
    print("Reasoning module enabled: True")

    print("\nGenerating flawed specification")
    flawed_specification_result = generate_flawed_specification_result(
        buggy_code=location.faulty_function,
        model_name=args.model,
        temperature=args.temperature,
        model=model,
    )
    full_demo.print_block("Flawed Specification", flawed_specification_result.flawed_specification)

    print("\nGenerating reasoning repair suggestion")
    reasoning_result = generate_repair_suggestion_result(
        program_id=context.program_id,
        buggy_code=location.faulty_function,
        buggy_file_path=str(location.source_path),
        language=context.language,
        fault_line_number=location.fault_line_number,
        additional_context=task_information,
        model_name=args.model,
        temperature=args.temperature,
        dataset_root=args.dataset_root,
    )
    full_demo.print_block("Reasoning Repair Suggestion", reasoning_result.repair_suggestion)

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

    fix_specification_input = build_fix_specification_input_with_reasoning(
        task_information=task_information,
        flawed_specification=flawed_specification_result.flawed_specification,
        reasoning_repair_suggestion=reasoning_result.repair_suggestion,
    )
    print("\nNow fixing specification")
    current_specification_result = fix_specification_session.start(fix_specification_input)
    full_demo.print_block("Fix Specification", current_specification_result.fix_specification)

    print("\nGenerating fix code")
    current_fix_code_result = fix_code_generator.generate(
        fix_specification=current_specification_result.fix_specification,
        buggy_code=location.faulty_function,
    )
    current_structure_adapter_result = full_demo.prepare_fix_code_for_validation(
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
        "use_reasoning_module": True,
        "fault_location_source": OCHIAI_FL_FILENAME,
        "fault_line_number": location.fault_line_number,
        "source_range": {
            "source_path": str(location.source_path),
            "start_line": location.start_line_index + 1,
            "end_line": location.end_line_index + 1,
        },
        "faulty_function": location.faulty_function,
        "flawed_specification": flawed_specification_result.flawed_specification,
        "flawed_specification_usage": full_demo._usage_dict(flawed_specification_result.usage),
        "flawed_specification_estimated_cost": flawed_specification_result.estimated_cost_usd,
        "reasoning": {
            "root_cause": reasoning_result.root_cause,
            "similar_fix_examples": reasoning_result.similar_fix_examples,
            "repair_suggestion": reasoning_result.repair_suggestion,
            "raw_response": reasoning_result.raw_response,
        },
        "reasoning_repair_suggestion": reasoning_result.repair_suggestion,
        "reasoning_usage": full_demo._usage_dict(reasoning_result.usage),
        "reasoning_estimated_cost": reasoning_result.estimated_cost_usd,
        "fix_specification_initial_input_includes_reasoning": True,
        "attempts": [],
    }
    full_demo.write_json(run_dir / "summary.json", summary)
    full_demo.write_text(run_dir / "task_information.txt", task_information)
    full_demo.write_text(run_dir / "faulty_function_with_ochiai_comment.txt", location.faulty_function)
    full_demo.write_text(run_dir / "flawed_specification.txt", flawed_specification_result.flawed_specification)
    full_demo.write_text(run_dir / "reasoning_repair_suggestion.txt", reasoning_result.repair_suggestion)

    for attempt_index in range(1, args.max_loops + 1):
        print(f"\nTesting attempt {attempt_index}/{args.max_loops}")
        validation = full_demo.validate_fix_code_attempt(
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
            "fix_specification_usage": full_demo._usage_dict(current_specification_result.usage),
            "fix_specification_estimated_cost": current_specification_result.estimated_cost_usd,
            "fix_code": current_structure_adapter_result.adapted_fix_code,
            "fix_code_raw_response": current_fix_code_result.raw_response,
            "fix_code_usage": full_demo._usage_dict(current_fix_code_result.usage),
            "fix_code_estimated_cost": current_fix_code_result.estimated_cost_usd,
        }
        if use_structure_adapter:
            attempt_payload.update(
                {
                    "fix_code_before_structure_adapter": current_fix_code_result.replacement_code,
                    "fix_code_structure_adapter_changed": current_structure_adapter_result.changed,
                    "fix_code_structure_adapter_reason": current_structure_adapter_result.reason,
                    "fix_code_structure_adapter_raw_response": current_structure_adapter_result.raw_response,
                    "fix_code_structure_adapter_usage": full_demo._usage_dict(current_structure_adapter_result.usage),
                    "fix_code_structure_adapter_estimated_cost": (
                        current_structure_adapter_result.estimated_cost_usd
                    ),
                    "fix_code_structure_adapter_deterministic_fallback_applied": (
                        current_structure_adapter_result.deterministic_fallback_applied
                    ),
                }
            )
        summary["attempts"].append(attempt_payload)
        full_demo.write_json(run_dir / "summary.json", summary)

        if validation.passed:
            print(f"[pass] attempt {attempt_index} passed all tests")
            summary["status"] = "success"
            summary["validated_attempt_index"] = attempt_index
            summary["final_fix_code"] = current_structure_adapter_result.adapted_fix_code
            full_demo.write_json(run_dir / "summary.json", summary)
            full_demo.copy_summary_if_requested(summary, args.output_json)
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
        full_demo.print_block(
            "Test Failure Specification",
            feedback_result.test_failure_specifications,
        )
        full_demo.write_text(
            validation.verification_dir / "test_failure_specifications.txt",
            feedback_result.test_failure_specifications,
        )
        attempt_payload["test_failure_specifications"] = feedback_result.test_failure_specifications
        attempt_payload["test_failure_specifications_usage"] = full_demo._usage_dict(
            feedback_result.usage
        )
        attempt_payload["test_failure_specifications_estimated_cost"] = (
            feedback_result.estimated_cost_usd
        )
        full_demo.write_json(run_dir / "summary.json", summary)

        print("\nNow fixing specification")
        current_specification_result = fix_specification_session.add_information(
            full_demo.build_verification_feedback_input(
                attempt_index=attempt_index,
                validation=validation,
                test_failure_specifications=feedback_result.test_failure_specifications,
            )
        )
        full_demo.print_block("Fix Specification", current_specification_result.fix_specification)

        print("\nGenerating fix code")
        current_fix_code_result = fix_code_generator.generate(
            fix_specification=current_specification_result.fix_specification,
            buggy_code=location.faulty_function,
        )
        current_structure_adapter_result = full_demo.prepare_fix_code_for_validation(
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
    full_demo.write_json(run_dir / "summary.json", summary)
    full_demo.copy_summary_if_requested(summary, args.output_json)
    return summary


run_full_demo_pipeline = run_full_demo_plus_reasoning_pipeline
write_json = full_demo.write_json
write_text = full_demo.write_text
print_block = full_demo.print_block
print_final_summary = full_demo.print_final_summary
copy_summary_if_requested = full_demo.copy_summary_if_requested


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    try:
        summary = run_full_demo_plus_reasoning_pipeline(args)
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
    full_demo.write_json(Path(summary["work_dir"]) / "summary.json", summary)
    full_demo.copy_summary_if_requested(summary, args.output_json)
    full_demo.print_final_summary(summary)
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
