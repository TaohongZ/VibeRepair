from __future__ import annotations

import argparse
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
import full_demoPlusReasoning_pipeline as full_demo  # noqa: E402
from viberepair.langchain_support import DEFAULT_MODEL_NAME, build_chat_model  # noqa: E402


DEFAULT_DATASET_ROOT = REPO_ROOT / "ConDefects" / "datasetForExperiment"
DEFAULT_FEEDBACK_RESULTS_DIR = (
    Path(__file__).resolve().parent / "eval_specification_feedback_module_Ochiai"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "eval_specification_feedback_reasoning_module_Ochiai"
)
DEFAULT_WORK_ROOT = Path("/tmp") / "viberepair_specification_feedback_reasoning_module_Ochiai"
DEFAULT_OUTER_REPAIR_RUNS = 5
DEFAULT_FEEDBACK_LOOPS = 5
DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
DEFAULT_TEMPERATURE = 0.2
LANGUAGES = ("Python", "Java")
OCHIAI_FL_FILENAME = "Ochiai_FL.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched iterative VibeRepair repairs with the reasoning module for "
            "programs whose feedback-Ochiai result did not pass."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the ConDefects datasetForExperiment directory.",
    )
    parser.add_argument(
        "--feedback-results-dir",
        default=str(DEFAULT_FEEDBACK_RESULTS_DIR),
        help=(
            "Directory containing eval_specification_feedback_module_Ochiai JSON files. "
            "Programs with pass != true are selected for repair."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where successful per-program JSON result files are written.",
    )
    parser.add_argument(
        "--work-root",
        default=str(DEFAULT_WORK_ROOT),
        help="Root directory used for temporary full-pipeline runs.",
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
        help="Optional maximum number of selected program ids to process.",
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
        help="Optional delay between programs and outer runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing successful per-program JSON files instead of skipping them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first unexpected program error.",
    )
    return parser.parse_args()


def collect_failed_feedback_program_ids(results_dir: str | Path) -> set[str]:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise ValueError(f"Feedback results directory is not a directory: {results_dir}")

    program_ids: set[str] = set()
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "batch_summary.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse feedback result JSON {path}: {exc}") from exc
        if not isinstance(data, dict):
            continue
        program_id = data.get("programId")
        if program_id is None:
            continue
        if data.get("pass") is not True:
            program_ids.add(str(program_id))
    return program_ids


def iter_program_contexts(
    dataset_root: str | Path,
    selected_program_ids: set[str],
    program_ids: list[str] | None = None,
) -> list[ProgramContext]:
    dataset_root = Path(dataset_root)
    if program_ids:
        explicit_program_ids = {str(program_id) for program_id in program_ids}
        return [
            resolve_program_context(program_id, dataset_root)
            for program_id in sorted(explicit_program_ids & selected_program_ids)
        ]

    contexts: list[ProgramContext] = []
    for task_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for language in LANGUAGES:
            language_dir = task_dir / language
            if not language_dir.is_dir():
                continue
            for program_dir in sorted(path for path in language_dir.iterdir() if path.is_dir()):
                if program_dir.name not in selected_program_ids:
                    continue
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


def result_path(output_dir: str | Path, context: ProgramContext) -> Path:
    filename = (
        f"{_safe_filename(context.task_name)}_"
        f"{_safe_filename(context.language)}_"
        f"{_safe_filename(context.program_id)}.json"
    )
    return Path(output_dir) / filename


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def install_full_demo_overrides(test_timeout_seconds: float) -> None:
    full_demo.TEST_TIMEOUT_SECONDS = test_timeout_seconds
    if hasattr(full_demo, "full_demo"):
        full_demo.full_demo.TEST_TIMEOUT_SECONDS = test_timeout_seconds


def run_program_repair(context: ProgramContext, args: argparse.Namespace) -> dict[str, Any] | None:
    output_path = result_path(args.output_dir, context)
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    program_work_root = Path(args.work_root) / _safe_filename(context.program_id)
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
            summary = full_demo.run_full_demo_plus_reasoning_pipeline(pipeline_args)
            summary["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
            summary["outer_run_index"] = outer_run_index
            write_json(Path(summary["work_dir"]) / "summary.json", build_chain_summary(summary))

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

            print(
                f"[failed-run] programId={context.program_id} outer_run="
                f"{outer_run_index}/{args.outer_runs}"
            )
        except Exception as exc:
            elapsed_seconds = round(time.monotonic() - started_at, 3)
            outer_records.append(
                {
                    "outer_run_index": outer_run_index,
                    "status": "error",
                    "work_dir": str(outer_work_root),
                    "elapsed_seconds": elapsed_seconds,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            write_json(
                outer_work_root / "summary.json",
                {
                    "status": "error",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "programId": context.program_id,
                    "task_name": context.task_name,
                    "language": context.language,
                    "work_dir": str(outer_work_root),
                    "outer_run_index": outer_run_index,
                    "max_loops": args.max_loops,
                    "test_timeout_seconds": args.test_timeout_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "attempts": [],
                },
            )
            print(
                f"[error-run] programId={context.program_id} outer_run={outer_run_index}/"
                f"{args.outer_runs}: {exc}",
                file=sys.stderr,
            )

        if outer_run_index < args.outer_runs:
            shutil.rmtree(outer_work_root, ignore_errors=True)
        if args.sleep_seconds > 0:
            sleep(args.sleep_seconds)

    print(f"[no-save] programId={context.program_id} did not pass after {args.outer_runs} runs")
    return None


def build_program_result(
    context: ProgramContext,
    args: argparse.Namespace,
    summary: dict[str, Any],
    outer_records: list[dict[str, Any]],
) -> dict[str, Any]:
    attempts = build_attempt_records(summary.get("attempts") or [])
    selected_attempt = find_selected_attempt(summary, attempts)
    flattened = flatten_selected_chain(summary, selected_attempt)
    result = {
        "status": "success",
        "pass": True,
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
        "faulty_function": summary.get("faulty_function"),
        "flawed_specification": summary.get("flawed_specification"),
        "flawed_specification_usage": summary.get("flawed_specification_usage"),
        "flawed_specification_estimated_cost": summary.get("flawed_specification_estimated_cost"),
        "reasoning": summary.get("reasoning"),
        "reasoning_repair_suggestion": summary.get("reasoning_repair_suggestion"),
        "reasoning_usage": summary.get("reasoning_usage"),
        "reasoning_estimated_cost": summary.get("reasoning_estimated_cost"),
        **flattened,
    }
    result["usage"] = build_usage_summary(summary, selected_attempt)
    result["estimated_cost"] = build_cost_summary(summary, selected_attempt)
    return result


def find_selected_attempt(
    summary: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    validated_attempt_index = summary.get("validated_attempt_index")
    if validated_attempt_index is not None:
        for attempt in attempts:
            if attempt.get("attempt_index") == validated_attempt_index:
                return attempt
    return attempts[-1] if attempts else None


def flatten_selected_chain(
    summary: dict[str, Any],
    selected_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    if not selected_attempt:
        return {}

    return copy_present(
        selected_attempt,
        [
            "selected_inner_attempt_index",
            "selected_inner_attempt_status",
            "pass_rate",
            "test_records",
            "test_error",
            "fix_specification_reasoning",
            "fixed_specification",
            "fix_code",
            "fix_code_raw_response",
        ],
        aliases={
            "selected_inner_attempt_index": "attempt_index",
            "selected_inner_attempt_status": "status",
        },
    )


def build_usage_summary(
    summary: dict[str, Any],
    selected_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "buggy_code_to_flawed_specification": summary.get("flawed_specification_usage"),
        "reasoning": summary.get("reasoning_usage"),
        "fix_specification": selected_attempt.get("fix_specification_usage") if selected_attempt else None,
        "fix_specification_and_buggy_code_to_fix_code": (
            selected_attempt.get("fix_code_usage") if selected_attempt else None
        ),
    }


def build_cost_summary(
    summary: dict[str, Any],
    selected_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "buggy_code_to_flawed_specification": summary.get(
            "flawed_specification_estimated_cost"
        ),
        "reasoning": summary.get("reasoning_estimated_cost"),
        "fix_specification": (
            selected_attempt.get("fix_specification_estimated_cost") if selected_attempt else None
        ),
        "fix_specification_and_buggy_code_to_fix_code": (
            selected_attempt.get("fix_code_estimated_cost") if selected_attempt else None
        ),
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
            "reasoning",
            "reasoning_repair_suggestion",
            "reasoning_usage",
            "reasoning_estimated_cost",
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
        if isinstance(attempt, dict)
    ]


def copy_present(
    source: dict[str, Any],
    keys: list[str],
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    aliases = aliases or {}
    for output_key in keys:
        source_key = aliases.get(output_key, output_key)
        if source_key in source:
            payload[output_key] = source[source_key]
    return payload


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
        selected_program_ids = collect_failed_feedback_program_ids(args.feedback_results_dir)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        contexts = iter_program_contexts(
            args.dataset_root,
            selected_program_ids=selected_program_ids,
            program_ids=args.program_ids,
        )
    except (ProgramFunctionLookupError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        contexts = contexts[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_summary_path = output_dir / "batch_summary.json"
    batch_summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feedback_results_dir": str(Path(args.feedback_results_dir).resolve()),
        "selected_feedback_failed_programs": len(selected_program_ids),
        "planned_programs": len(contexts),
        "output_dir": str(output_dir.resolve()),
        "work_root": str(Path(args.work_root).resolve()),
        "outer_runs": args.outer_runs,
        "max_loops": args.max_loops,
        "test_timeout_seconds": args.test_timeout_seconds,
        "completed": 0,
        "skipped": 0,
        "passed": 0,
        "failed_without_result": 0,
        "errors_without_result": 0,
        "programs": [],
    }
    write_batch_summary(batch_summary_path, batch_summary)

    print(
        f"Selected {len(selected_program_ids)} program ids with pass != true from "
        f"{args.feedback_results_dir}"
    )
    print(f"Processing {len(contexts)} selected program ids")
    print(f"Writing successful JSON results to {output_dir}")
    print(f"Writing temporary runs to {args.work_root}")
    print(f"Reading fault lines from {OCHIAI_FL_FILENAME}")

    if not contexts:
        print("No selected program ids to repair.")
        print_final_batch_summary(batch_summary, batch_summary_path)
        return 0

    try:
        full_demo.import_viberepair_full_pipeline_functions()
        build_chat_model(model_name=args.model, temperature=args.temperature)
    except (EnvironmentError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for index, context in enumerate(contexts, start=1):
        output_path = result_path(output_dir, context)
        if output_path.exists() and not args.overwrite:
            batch_summary["skipped"] += 1
            batch_summary["programs"].append(
                {
                    "programId": context.program_id,
                    "task_name": context.task_name,
                    "language": context.language,
                    "status": "skipped_existing_success_result",
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
        except Exception as exc:
            batch_summary["errors_without_result"] += 1
            batch_summary["programs"].append(
                {
                    "programId": context.program_id,
                    "task_name": context.task_name,
                    "language": context.language,
                    "status": "error_no_result_saved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[error] programId={context.program_id}: {exc}", file=sys.stderr)
            if args.stop_on_error:
                write_batch_summary(batch_summary_path, batch_summary)
                print_final_batch_summary(batch_summary, batch_summary_path)
                return 1
        else:
            batch_summary["completed"] += 1
            if result is None:
                batch_summary["failed_without_result"] += 1
                batch_summary["programs"].append(
                    {
                        "programId": context.program_id,
                        "task_name": context.task_name,
                        "language": context.language,
                        "status": "failed_no_result_saved",
                        "pass": False,
                    }
                )
            else:
                batch_summary["passed"] += 1
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

        write_batch_summary(batch_summary_path, batch_summary)
        if args.sleep_seconds > 0:
            sleep(args.sleep_seconds)

    print_final_batch_summary(batch_summary, batch_summary_path)
    return 0


def print_final_batch_summary(summary: dict[str, Any], summary_path: Path) -> None:
    print(
        "\nFinished: "
        f"completed={summary['completed']}, skipped={summary['skipped']}, "
        f"passed={summary['passed']}, failed_without_result={summary['failed_without_result']}, "
        f"errors_without_result={summary['errors_without_result']}, planned={summary['planned_programs']}"
    )
    print(f"Batch summary written to {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
