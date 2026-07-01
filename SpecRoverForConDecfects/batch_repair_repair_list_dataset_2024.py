from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpecRoverForConDecfects.config import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_PATCH_RETRIES,
    DEFAULT_TEMPERATURE,
)
from SpecRoverForConDecfects.condefects_dataset import (  # noqa: E402
    OCHIAI_FL_FILENAME,
    ProgramContext,
    ProgramFunctionLookupError,
    iter_program_contexts,
)
from SpecRoverForConDecfects.specrover_condefects import (  # noqa: E402
    RepairRunOptions,
    run_one_repair,
    write_json,
)
from SpecRoverForConDecfects.validation import (  # noqa: E402
    DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
    DEFAULT_VALIDATION_TIMEOUT_SECONDS,
)


BATCH_MODEL_NAME = DEFAULT_MODEL_NAME
DEFAULT_FINAL_REPORT_DIR = (
    PACKAGE_ROOT / "output" / "eval_specrover_validation_feedback_module_Ochiai"
)
DEFAULT_WORK_OUTPUT_ROOT = (
    PACKAGE_ROOT / "output" / "batch_repair_datasetForExperiment_work"
)
DEFAULT_SUMMARY_PATH = (
    PACKAGE_ROOT / "output" / "batch_repair_datasetForExperiment_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SpecRover ConDefects baseline on ConDefects/datasetForExperiment. "
            "Validation is always enabled and Ochiai_FL.txt is the only fault location source."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to ConDefects/datasetForExperiment.",
    )
    parser.add_argument(
        "--final-report-dir",
        default=str(DEFAULT_FINAL_REPORT_DIR),
        help="Directory for one final report JSON per repaired program.",
    )
    parser.add_argument(
        "--work-output-root",
        default=str(DEFAULT_WORK_OUTPUT_ROOT),
        help="Directory for per-run conversations, patches, and validation logs.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to write the batch progress summary JSON.",
    )
    parser.add_argument(
        "--task-info-cache",
        default=str(PACKAGE_ROOT / "output" / "task_information_cache.json"),
        help="JSON cache for AtCoder task Problem Statement and Constraints.",
    )
    parser.add_argument(
        "--program-id",
        action="append",
        dest="program_ids",
        help="Optionally run a subset of datasetForExperiment program IDs. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Optionally limit to a language directory, for example Python or Java.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected programs to process.",
    )
    parser.add_argument(
        "--model",
        default=BATCH_MODEL_NAME,
        help=f"Chat model to use. Default: {BATCH_MODEL_NAME}.",
    )
    parser.add_argument(
        "--max-independent-runs",
        type=int,
        default=5,
        help="Maximum completely independent SpecRover repair attempts per program. Default: 5.",
    )
    parser.add_argument(
        "--review-rounds",
        type=int,
        default=5,
        help="Maximum reviewer/validation rounds inside one independent repair. Default: 5.",
    )
    parser.add_argument(
        "--max-patch-retries",
        type=int,
        default=DEFAULT_PATCH_RETRIES,
        help=f"Tagged-edit retry count per patch generation. Default: {DEFAULT_PATCH_RETRIES}.",
    )
    parser.add_argument(
        "--test-timeout-seconds",
        type=float,
        default=DEFAULT_VALIDATION_TIMEOUT_SECONDS,
        help=(
            "Total timeout budget for one validation attempt. Timeout means "
            f"validation fails. Default: {DEFAULT_VALIDATION_TIMEOUT_SECONDS:g}."
        ),
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
        help=(
            "Maximum RSS for one validation process tree before the attempt is "
            f"killed and marked failed. Use 0 to disable. Default: {DEFAULT_VALIDATION_MEMORY_LIMIT_MB}."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Model temperature. Default: {DEFAULT_TEMPERATURE}.",
    )
    parser.add_argument(
        "--reproduce-and-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the SpecRover-style reproducer + reviewer loop. Default: true.",
    )
    parser.add_argument(
        "--max-reproducer-tests",
        type=int,
        default=None,
        help="Optional maximum number of ConDefects test cases to scan for a reproducer.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay after each independent repair run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate final reports that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare prompts and metadata without invoking the model.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first unexpected exception.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_error = validate_args(args)
    if validation_error:
        print(f"Error: {validation_error}", file=sys.stderr)
        return 1

    final_report_dir = Path(args.final_report_dir)
    work_output_root = Path(args.work_output_root)
    final_report_dir.mkdir(parents=True, exist_ok=True)
    work_output_root.mkdir(parents=True, exist_ok=True)

    try:
        contexts, skipped_info = select_contexts(args)
    except (OSError, ValueError, ProgramFunctionLookupError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    task_information_cache = load_task_information_cache(args.task_info_cache)
    summary = build_initial_summary(args, contexts, skipped_info)
    write_json(args.summary_output, summary)

    print(
        f"Processing {len(contexts)} datasetForExperiment programs with "
        f"model={args.model}, validation=enabled, "
        f"max_independent_runs={args.max_independent_runs}",
        flush=True,
    )
    print(f"Final report dir: {final_report_dir}", flush=True)
    print(f"Work output root: {work_output_root}", flush=True)

    options = RepairRunOptions(
        model_name=args.model,
        temperature=args.temperature,
        max_patch_retries=args.max_patch_retries,
        dry_run=args.dry_run,
        reproduce_and_review=args.reproduce_and_review,
        enable_validation=True,
        review_rounds=args.review_rounds,
        max_reproducer_tests=args.max_reproducer_tests,
        validation_timeout_seconds=args.test_timeout_seconds,
        validation_memory_limit_mb=args.memory_limit_mb,
    )

    for program_index, context in enumerate(contexts, start=1):
        report_path = final_report_path(final_report_dir, context)
        existing_report = load_existing_report(report_path) if report_path.exists() else None
        if existing_report is not None and existing_report.get("pass") is True and not args.overwrite:
            summary["skipped_existing_reports"] += 1
            summary["programs"].append(
                {
                    "programId": context.program_id,
                    "task_name": context.task_name,
                    "language": context.language,
                    "status": "skipped_existing_success_report",
                    "existing_status": existing_report.get("status"),
                    "existing_pass": existing_report.get("pass"),
                    "report_path": str(report_path),
                }
            )
            write_json(args.summary_output, summary)
            print(
                f"[{program_index}/{len(contexts)}] [skip existing success] "
                f"{context.program_id} -> {report_path}",
                flush=True,
            )
            continue
        if report_path.exists():
            report_path.unlink()

        print(
            f"[{program_index}/{len(contexts)}] repairing "
            f"{context.task_name}/{context.language}/{context.program_id}",
            flush=True,
        )
        try:
            report = repair_program_with_independent_runs(
                context=context,
                args=args,
                options=options,
                work_output_root=work_output_root,
                task_information_cache=task_information_cache,
                report_path=report_path,
            )
        except Exception as exc:
            report = build_exception_report(context, args, exc)
            print(f"  -> exception {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.stop_on_error:
                summary["finished_at"] = datetime.now().isoformat()
                summary["programs"].append(program_summary_from_report(report, report_path))
                write_json(args.summary_output, summary)
                save_task_information_cache(args.task_info_cache, task_information_cache)
                return 1

        save_task_information_cache(args.task_info_cache, task_information_cache)
        if report.get("pass") is True:
            summary["programs_passed"] += 1
        elif report.get("status") == "dry_run":
            summary["programs_dry_run"] += 1
        else:
            summary["programs_failed"] += 1
        summary["programs_completed"] += 1
        summary["programs"].append(program_summary_from_report(report, report_path))
        write_json(args.summary_output, summary)

    summary["finished_at"] = datetime.now().isoformat()
    write_json(args.summary_output, summary)
    save_task_information_cache(args.task_info_cache, task_information_cache)
    return 0


def load_existing_report(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_args(args: argparse.Namespace) -> str | None:
    if args.limit is not None and args.limit < 1:
        return "--limit must be at least 1 when provided"
    if args.max_independent_runs < 1:
        return "--max-independent-runs must be at least 1"
    if args.review_rounds < 1:
        return "--review-rounds must be at least 1"
    if args.max_patch_retries < 1:
        return "--max-patch-retries must be at least 1"
    if args.test_timeout_seconds <= 0:
        return "--test-timeout-seconds must be positive"
    if args.memory_limit_mb < 0:
        return "--memory-limit-mb must be zero or positive"
    if args.max_reproducer_tests is not None and args.max_reproducer_tests < 1:
        return "--max-reproducer-tests must be at least 1 when provided"
    if args.sleep_seconds < 0:
        return "--sleep-seconds must be zero or positive"
    return None


def repair_program_with_independent_runs(
    context: ProgramContext,
    args: argparse.Namespace,
    options: RepairRunOptions,
    work_output_root: Path,
    task_information_cache: dict[str, str],
    report_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    report = build_program_report(context, args)
    max_runs = 1 if args.dry_run else args.max_independent_runs
    last_payload: dict[str, Any] | None = None

    for outer_run_index in range(1, max_runs + 1):
        run_work_dir = independent_run_work_dir(
            work_output_root,
            context.program_id,
            outer_run_index,
        )
        run_started = time.monotonic()
        try:
            payload = run_one_repair(
                context=context,
                options=options,
                work_dir=run_work_dir,
                task_information_cache=task_information_cache,
            )
            elapsed = round(time.monotonic() - run_started, 2)
            payload["outer_run_index"] = outer_run_index
            payload["batch_elapsed_seconds"] = elapsed
            last_payload = payload
            report["outer_run_history"].append(
                summarize_run_payload(payload, outer_run_index, elapsed)
            )

            if args.dry_run:
                finalize_report(
                    report,
                    status="dry_run",
                    passed=False,
                    started=started,
                    selected_payload=payload,
                    selected_outer_run_index=outer_run_index,
                )
                return report

            if payload_passed_validation(payload):
                finalize_report(
                    report,
                    status="success",
                    passed=True,
                    started=started,
                    selected_payload=payload,
                    selected_outer_run_index=outer_run_index,
                )
                write_json(report_path, report)
                return report
        except Exception as exc:
            elapsed = round(time.monotonic() - run_started, 2)
            report["outer_run_history"].append(
                {
                    "outer_run_index": outer_run_index,
                    "status": "exception",
                    "pass": False,
                    "elapsed_seconds": elapsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            report["last_exception"] = {
                "outer_run_index": outer_run_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if args.stop_on_error:
                raise

        finalize_report(
            report,
            status="running",
            passed=False,
            started=started,
            selected_payload=last_payload,
            selected_outer_run_index=None,
        )
        save_task_information_cache(args.task_info_cache, task_information_cache)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    finalize_report(
        report,
        status="failed",
        passed=False,
        started=started,
        selected_payload=last_payload,
        selected_outer_run_index=None,
    )
    return report


def build_program_report(context: ProgramContext, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running",
        "pass": False,
        "programId": context.program_id,
        "task_name": context.task_name,
        "language": context.language,
        "outer_runs": args.max_independent_runs,
        "max_independent_runs": args.max_independent_runs,
        "review_rounds": args.review_rounds,
        "max_validation_feedbacks_to_patch_agent": max(0, args.review_rounds - 1),
        "max_patch_retries": args.max_patch_retries,
        "test_timeout_seconds": args.test_timeout_seconds,
        "memory_limit_mb": args.memory_limit_mb,
        "enable_validation": True,
        "reproduce_and_review": args.reproduce_and_review,
        "max_reproducer_tests": args.max_reproducer_tests,
        "generated_at": datetime.now().isoformat(),
        "finished_at": None,
        "elapsed_seconds": None,
        "selected_outer_run_index": None,
        "outer_run_history": [],
    }


def finalize_report(
    report: dict[str, Any],
    status: str,
    passed: bool,
    started: float,
    selected_payload: dict[str, Any] | None,
    selected_outer_run_index: int | None,
) -> None:
    report["status"] = status
    report["pass"] = passed
    report["finished_at"] = datetime.now().isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["selected_outer_run_index"] = selected_outer_run_index

    if not isinstance(selected_payload, dict):
        return

    report["faulty_function"] = selected_payload.get("faulty_function")
    report["fix_code"] = selected_payload.get("fix_code")
    report["fix_code_raw_response"] = selected_payload.get("raw_patch_response")
    report["patch_handle"] = selected_payload.get("patch_handle")
    report["workflow"] = selected_payload.get("workflow")
    report["usage"] = selected_payload.get("usage")
    report["estimated_cost"] = selected_payload.get("estimated_cost")

    validation = public_validation_json(selected_payload.get("validation"))
    if validation is not None:
        report["validation"] = validation
        report["pass_rate"] = validation.get("pass_rate")


def public_validation_json(validation: Any) -> dict[str, Any] | None:
    if not isinstance(validation, dict):
        return None
    return {
        "passed": validation.get("passed"),
        "message": validation.get("message"),
        "pass_rate": validation.get("pass_rate"),
        "timed_out": validation.get("timed_out"),
        "memory_exceeded": validation.get("memory_exceeded"),
    }


def summarize_run_payload(
    payload: dict[str, Any],
    outer_run_index: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    validation = payload.get("validation")
    pass_rate = validation.get("pass_rate") if isinstance(validation, dict) else None
    validation_passed = validation.get("passed") if isinstance(validation, dict) else None
    timed_out = validation.get("timed_out") if isinstance(validation, dict) else None
    memory_exceeded = validation.get("memory_exceeded") if isinstance(validation, dict) else None
    return {
        "outer_run_index": outer_run_index,
        "status": payload.get("status"),
        "pass": payload_passed_validation(payload),
        "validation_passed": validation_passed,
        "timed_out": timed_out,
        "memory_exceeded": memory_exceeded,
        "pass_rate": pass_rate,
        "patch_handle": payload.get("patch_handle"),
        "workflow": payload.get("workflow"),
        "elapsed_seconds": elapsed_seconds,
        "estimated_cost": payload.get("estimated_cost"),
    }


def payload_passed_validation(payload: dict[str, Any]) -> bool:
    validation = payload.get("validation")
    return (
        payload.get("status") == "success"
        and isinstance(validation, dict)
        and validation.get("passed") is True
    )


def build_exception_report(
    context: ProgramContext,
    args: argparse.Namespace,
    error: Exception,
) -> dict[str, Any]:
    report = build_program_report(context, args)
    report["status"] = "exception"
    report["pass"] = False
    report["finished_at"] = datetime.now().isoformat()
    report["error_type"] = type(error).__name__
    report["error"] = str(error)
    return report


def program_summary_from_report(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    validation = report.get("validation")
    timed_out = validation.get("timed_out") if isinstance(validation, dict) else None
    memory_exceeded = validation.get("memory_exceeded") if isinstance(validation, dict) else None
    return {
        "programId": report.get("programId"),
        "task_name": report.get("task_name"),
        "language": report.get("language"),
        "status": report.get("status"),
        "pass": report.get("pass"),
        "timed_out": timed_out,
        "memory_exceeded": memory_exceeded,
        "selected_outer_run_index": report.get("selected_outer_run_index"),
        "outer_runs_executed": len(report.get("outer_run_history") or []),
        "pass_rate": report.get("pass_rate"),
        "report_path": str(report_path) if report.get("pass") is True else None,
    }


def select_contexts(args: argparse.Namespace) -> tuple[list[ProgramContext], dict[str, Any]]:
    all_contexts = iter_program_contexts(
        dataset_root=args.dataset_root,
        program_ids=args.program_ids,
        require_fault_location=False,
        languages=args.languages,
    )
    eligible_contexts = [
        context for context in all_contexts if (context.program_dir / OCHIAI_FL_FILENAME).exists()
    ]
    contexts = sorted(
        eligible_contexts,
        key=lambda context: (context.task_name, context.language, context.program_id),
    )
    if args.limit is not None:
        contexts = contexts[: args.limit]

    all_ids = {context.program_id for context in all_contexts}
    eligible_ids = {context.program_id for context in eligible_contexts}
    skipped_info = {
        "dataset_program_count": len(all_contexts),
        "programs_with_ochiai_count": len(eligible_contexts),
        "missing_ochiai_program_ids": sorted(all_ids - eligible_ids),
        "selected_program_count": len(contexts),
    }
    return contexts, skipped_info


def build_initial_summary(
    args: argparse.Namespace,
    contexts: list[ProgramContext],
    skipped_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(),
        "finished_at": None,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "final_report_dir": str(Path(args.final_report_dir).resolve()),
        "work_output_root": str(Path(args.work_output_root).resolve()),
        "task_info_cache": str(Path(args.task_info_cache).resolve()),
        "model": args.model,
        "temperature": args.temperature,
        "enable_validation": True,
        "reproduce_and_review": args.reproduce_and_review,
        "max_independent_runs": args.max_independent_runs,
        "review_rounds": args.review_rounds,
        "max_validation_feedbacks_to_patch_agent": max(0, args.review_rounds - 1),
        "max_patch_retries": args.max_patch_retries,
        "test_timeout_seconds": args.test_timeout_seconds,
        "memory_limit_mb": args.memory_limit_mb,
        "max_reproducer_tests": args.max_reproducer_tests,
        "dry_run": args.dry_run,
        "total_programs": len(contexts),
        "programs_completed": 0,
        "programs_passed": 0,
        "programs_failed": 0,
        "programs_dry_run": 0,
        "skipped_existing_reports": 0,
        "skipped_info": skipped_info,
        "programs": [],
    }


def final_report_path(final_report_dir: str | Path, context: ProgramContext) -> Path:
    filename = (
        f"{safe_filename(context.task_name)}_"
        f"{safe_filename(context.language)}_"
        f"{safe_filename(context.program_id)}.json"
    )
    return Path(final_report_dir) / filename


def independent_run_work_dir(
    work_output_root: str | Path,
    program_id: str,
    outer_run_index: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path(work_output_root)
        / safe_filename(program_id)
        / f"outer_run_{outer_run_index:02d}"
        / f"{safe_filename(program_id)}_{timestamp}"
    )


def load_task_information_cache(path: str | Path) -> dict[str, str]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if isinstance(value, str)
    }


def save_task_information_cache(path: str | Path, cache: dict[str, str]) -> None:
    write_json(path, cache)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
