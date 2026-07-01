from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ReinFixForConDecfects.src.config import (  # noqa: E402
    DEFAULT_AGENT_MODEL_NAME,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PATCH_MODEL_NAME,
    DEFAULT_TEST_ROOT,
)
from ReinFixForConDecfects.src.condefects_dataset import (  # noqa: E402
    ProgramContext,
    ProgramFunctionLookupError,
    get_ochiai_faulty_function_with_reference_comment,
    iter_program_contexts,
)


DEFAULT_WORK_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "batch_repair_dataset_2024_work"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "batch_repair_dataset_2024_fix_code_results"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "batch_repair_dataset_2024_summary.json"
DEFAULT_REPAIR_RUNS = 5
DEFAULT_RAG_HOST = "127.0.0.1"
DEFAULT_RAG_PORT = 5000
DEFAULT_RAG_STARTUP_TIMEOUT_SECONDS = 900.0


class Logger:
    def __init__(self, filename: str, stream=sys.stdout):
        self.terminal = stream
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filename, "a", encoding="utf-8")

    def _log(self, level: str, message: str) -> None:
        if message.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"[{timestamp}] [{level.upper()}] {message}\n"
            self.terminal.write(message)
            self.log.write(message)

    def info(self, message: str) -> None:
        self._log("info", message)

    def error(self, message: str) -> None:
        self._log("error", message)

    def warning(self, message: str) -> None:
        self._log("warning", message)

    def write(self, message: str) -> None:
        if message.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"[{timestamp}] {message}"
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.flush()
        self.log.close()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def load_reinfix_runtime() -> dict[str, Any]:
    from ReinFixForConDecfects.src.react_sf_gen_patch import (  # noqa: WPS433
        ensure_environment,
        run_patch_generation,
        run_solution_generation,
    )

    return {
        "ensure_environment": ensure_environment,
        "run_patch_generation": run_patch_generation,
        "run_solution_generation": run_solution_generation,
    }


def build_rag_search_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/search"


def rag_health_url(search_url: str) -> str:
    parsed = urllib.parse.urlparse(search_url)
    return urllib.parse.urlunparse(parsed._replace(path="/health", query="", fragment=""))


def configure_rag_search_url(args: argparse.Namespace) -> str:
    search_url = build_rag_search_url(args.rag_host, args.rag_port)
    os.environ["REINFIX_CONDEFECTS_SEARCH_URL"] = search_url
    args.rag_search_url = search_url
    try:
        from ReinFixForConDecfects.src import config as reinfix_config  # noqa: WPS433

        reinfix_config.RAG_SEARCH_URL = search_url
    except Exception:
        pass
    return search_url


def is_rag_server_ready(search_url: str, timeout_seconds: float = 2.0) -> bool:
    request = urllib.request.Request(rag_health_url(search_url), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("status") == "ok"
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False


def ensure_rag_server(args: argparse.Namespace) -> tuple[subprocess.Popen[str] | None, Any | None]:
    search_url = configure_rag_search_url(args)
    if not args.start_rag_server:
        print(f"Using RAG search URL without auto-start: {search_url}")
        return None, None

    if is_rag_server_ready(search_url):
        print(f"RAG server already healthy: {search_url}")
        return None, None

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    log_path = Path(args.rag_log_path) if args.rag_log_path else Path(args.work_output_root) / "rag_search_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "ReinFixForConDecfects.src.rag_search_server",
        "--host",
        args.rag_host,
        "--port",
        str(args.rag_port),
    ]
    print(f"Starting RAG server: {' '.join(cmd)}")
    print(f"RAG server log: {log_path}")
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ.copy(), "PYTHONUNBUFFERED": "1"},
    )
    atexit.register(stop_rag_server, process, log_handle)

    deadline = time.monotonic() + args.rag_startup_timeout_seconds
    while time.monotonic() < deadline:
        if is_rag_server_ready(search_url):
            print(f"RAG server ready: {search_url}")
            return process, log_handle
        returncode = process.poll()
        if returncode is not None:
            stop_rag_server(process, log_handle)
            raise RuntimeError(
                f"RAG server exited before becoming healthy with code {returncode}. "
                f"See log: {log_path}"
            )
        time.sleep(2.0)

    stop_rag_server(process, log_handle)
    raise TimeoutError(
        f"Timed out waiting for RAG server after {args.rag_startup_timeout_seconds:g}s. "
        f"See log: {log_path}"
    )


def stop_rag_server(process: subprocess.Popen[str] | None, log_handle: Any | None = None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    if log_handle is not None and not getattr(log_handle, "closed", True):
        log_handle.flush()
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched ReInFix repairs for ConDefects/datasetForExperiment. "
            "Ochiai_FL.txt is the only fault location source."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to ConDefects/datasetForExperiment.",
    )
    parser.add_argument(
        "--test-root",
        default=str(DEFAULT_TEST_ROOT),
        help="Path to ConDefects/Test.",
    )
    parser.add_argument(
        "--work-output-root",
        default=str(DEFAULT_WORK_OUTPUT_ROOT),
        help="Directory for per-run ReInFix intermediate solutions, patches, and logs.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=(
            "Directory where consolidated per-patch JSON files are written. Pass this "
            "path to eval_ConDefects/validate_fix_code_results_ochiai.py --results-dir."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to write a batch progress summary JSON.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_REPAIR_RUNS,
        help=f"Number of independent repair runs per program ID. Default: {DEFAULT_REPAIR_RUNS}.",
    )
    parser.add_argument(
        "--solution-runs",
        type=int,
        default=1,
        help="Number of ReInFix solution-generation rounds inside each independent repair run.",
    )
    parser.add_argument(
        "--patch-samples",
        type=int,
        default=1,
        help=(
            "Number of OpenAI patch completions sampled for each root-cause/suggestion pair "
            "inside one independent repair run."
        ),
    )
    parser.add_argument(
        "--program-id",
        action="append",
        dest="program_ids",
        help=(
            "Limit the batch to one datasetForExperiment program ID. Repeat this option "
            "for multiple IDs."
        ),
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Limit the batch to a language directory, for example Python or Java.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected datasetForExperiment programs to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay after each independent repair run.",
    )
    parser.add_argument(
        "--start-rag-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the local example-patch RAG server before repair. Default: true.",
    )
    parser.add_argument(
        "--rag-host",
        default=DEFAULT_RAG_HOST,
        help=f"Host used for the local RAG server. Default: {DEFAULT_RAG_HOST}.",
    )
    parser.add_argument(
        "--rag-port",
        type=int,
        default=DEFAULT_RAG_PORT,
        help=f"Port used for the local RAG server. Default: {DEFAULT_RAG_PORT}.",
    )
    parser.add_argument(
        "--rag-startup-timeout-seconds",
        type=float,
        default=DEFAULT_RAG_STARTUP_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for the RAG server to load the retrieval database and "
            f"become healthy. Default: {DEFAULT_RAG_STARTUP_TIMEOUT_SECONDS:g}."
        ),
    )
    parser.add_argument(
        "--rag-log-path",
        default=None,
        help="Optional log file for the auto-started RAG server.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate runs that already have consolidated result JSON files.",
    )
    parser.add_argument(
        "--no-reinfix-marker",
        action="store_true",
        help="Do not add the original ReInFix /* bug is here */ marker to the faulty function.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first program/run error.",
    )
    return parser.parse_args()



def select_contexts(args: argparse.Namespace) -> tuple[list[ProgramContext], dict[str, Any]]:
    contexts = iter_program_contexts(
        dataset_root=args.dataset_root,
        program_ids=args.program_ids,
        require_ochiai=True,
        languages=args.languages,
    )
    dataset_context_count = len(contexts)
    if args.limit is not None:
        contexts = contexts[: args.limit]

    selection = {
        "dataset_context_count": dataset_context_count,
        "selected_count": len(contexts),
    }
    return contexts, selection

def result_files_for_program(results_dir: str | Path, program_id: str) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []

    files: list[Path] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("programId") or "") == program_id:
            files.append(path)
    return files


def existing_repair_runs(results_dir: str | Path, program_id: str) -> set[int]:
    runs: set[int] = set()
    for path in result_files_for_program(results_dir, program_id):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        repair_run = payload.get("repair_run")
        is_success_patch = payload.get("status") == "success" and isinstance(payload.get("fix_code"), str)
        is_watchdog_skip = payload.get("status") == "error" and payload.get("watchdog_skip") is True
        if isinstance(repair_run, int) and (is_success_patch or is_watchdog_skip):
            runs.add(repair_run)
    return runs


def next_repair_index(results_dir: str | Path, program_id: str) -> int:
    max_index = 0
    for path in result_files_for_program(results_dir, program_id):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        repair_index = payload.get("repair_index")
        if isinstance(repair_index, int):
            max_index = max(max_index, repair_index)
    return max_index + 1


def delete_run_results(results_dir: str | Path, program_id: str, repair_run: int) -> None:
    for path in result_files_for_program(results_dir, program_id):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("repair_run") == repair_run:
            path.unlink()


def consolidated_result_path(results_dir: str | Path, context: ProgramContext, repair_index: int) -> Path:
    filename = (
        f"{safe_filename(context.task_name)}_"
        f"{safe_filename(context.language)}_"
        f"{safe_filename(context.program_id)}_"
        f"repair_{repair_index:03d}.json"
    )
    return Path(results_dir) / filename


def write_consolidated_patch_results(
    context: ProgramContext,
    temp_fix_result_paths: list[Path],
    results_dir: str | Path,
    repair_run: int,
    attempt_root: Path,
) -> int:
    selected_path = first_successful_patch_result(temp_fix_result_paths)
    if selected_path is None:
        return 0

    payload = read_json(selected_path)
    source_repair_index = payload.get("repair_index")
    repair_index = next_repair_index(results_dir, context.program_id)
    for key in ("program_dir", "fault_location_source", "fault_line_number"):
        payload.pop(key, None)
    payload["repair_index"] = repair_index
    payload["repair_run"] = repair_run
    payload["source_repair_index"] = source_repair_index
    payload["discarded_patch_count_in_same_run"] = max(0, len(temp_fix_result_paths) - 1)
    payload["batch_attempt_work_dir"] = str(attempt_root)
    payload["batch_generated_at"] = datetime.now().isoformat()
    output_path = consolidated_result_path(results_dir, context, repair_index)
    write_json(output_path, payload)
    return 1


def first_successful_patch_result(temp_fix_result_paths: list[Path]) -> Path | None:
    for temp_path in sorted(temp_fix_result_paths, key=repair_result_sort_key):
        try:
            payload = read_json(temp_path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "success" and isinstance(payload.get("fix_code"), str):
            return temp_path
    return None


def repair_result_sort_key(path: Path) -> tuple[int, str]:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return (10**9, path.name)
    repair_index = payload.get("repair_index")
    if isinstance(repair_index, int):
        return (repair_index, path.name)
    return (10**9, path.name)


def write_error_result(
    context: ProgramContext,
    results_dir: str | Path,
    repair_run: int,
    error: Exception | str,
    attempt_root: Path | None = None,
) -> Path:
    repair_index = next_repair_index(results_dir, context.program_id)
    output_path = consolidated_result_path(results_dir, context, repair_index)
    payload = {
        "status": "error",
        "programId": context.program_id,
        "repair_index": repair_index,
        "repair_run": repair_run,
        "task_name": context.task_name,
        "language": context.language,
        "agent_model": DEFAULT_AGENT_MODEL_NAME,
        "patch_model": DEFAULT_PATCH_MODEL_NAME,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "NoPatchGenerated",
        "error": str(error),
        "batch_attempt_work_dir": str(attempt_root) if attempt_root else None,
        "batch_generated_at": datetime.now().isoformat(),
    }
    write_json(output_path, payload)
    return output_path


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def build_initial_summary(
    args: argparse.Namespace,
    selection: dict[str, Any],
    contexts: list[ProgramContext],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(),
        "finished_at": None,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "work_output_root": str(Path(args.work_output_root).resolve()),
        "results_dir": str(Path(args.results_dir).resolve()),
        "runs_per_program": args.runs,
        "solution_runs_per_independent_run": args.solution_runs,
        "patch_samples": args.patch_samples,
        "rag_search_url": getattr(args, "rag_search_url", build_rag_search_url(args.rag_host, args.rag_port)),
        "start_rag_server": args.start_rag_server,
        "rag_startup_timeout_seconds": args.rag_startup_timeout_seconds,
        "agent_model": DEFAULT_AGENT_MODEL_NAME,
        "patch_model": DEFAULT_PATCH_MODEL_NAME,
        "selection": selection,
        "total_programs": len(contexts),
        "completed_runs": 0,
        "skipped_runs": 0,
        "failed_runs": 0,
        "written_patch_jsons": 0,
        "programs": [],
    }


def update_program_record(summary: dict[str, Any], record: dict[str, Any]) -> None:
    programs = summary.setdefault("programs", [])
    for index, existing in enumerate(programs):
        if existing.get("programId") == record.get("programId"):
            programs[index] = record
            return
    programs.append(record)


def run_one_independent_repair(
    context: ProgramContext,
    repair_run: int,
    args: argparse.Namespace,
    logger: Logger,
    runtime: dict[str, Any],
) -> tuple[int, Path]:
    attempt_root = (
        Path(args.work_output_root)
        / "attempts"
        / safe_filename(context.program_id)
        / f"repair_{repair_run:02d}"
    )
    attempt_root.mkdir(parents=True, exist_ok=True)

    faulty_function, fault_line_number = get_ochiai_faulty_function_with_reference_comment(
        context,
        include_reinfix_marker=not args.no_reinfix_marker,
    )
    solution_path = runtime["run_solution_generation"](
        context=context,
        faulty_function=faulty_function,
        fault_line_number=fault_line_number,
        output_root=attempt_root,
        args=args,
        logger=logger,
    )
    runtime["run_patch_generation"](
        context=context,
        faulty_function=faulty_function,
        fault_line_number=fault_line_number,
        solution_path=solution_path,
        output_root=attempt_root,
        args=args,
        logger=logger,
    )
    temp_fix_results_dir = attempt_root / "fix_code_results"
    temp_fix_result_paths = sorted(temp_fix_results_dir.glob("*.json")) if temp_fix_results_dir.is_dir() else []
    written = write_consolidated_patch_results(
        context=context,
        temp_fix_result_paths=temp_fix_result_paths,
        results_dir=args.results_dir,
        repair_run=repair_run,
        attempt_root=attempt_root,
    )
    if written == 0:
        write_error_result(
            context=context,
            results_dir=args.results_dir,
            repair_run=repair_run,
            error="ReInFix completed this run but produced no parseable patch.",
            attempt_root=attempt_root,
        )
    return written, attempt_root


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("Error: --runs must be at least 1", file=sys.stderr)
        return 1
    if args.solution_runs < 1:
        print("Error: --solution-runs must be at least 1", file=sys.stderr)
        return 1
    if args.patch_samples < 1:
        print("Error: --patch-samples must be at least 1", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1 when provided", file=sys.stderr)
        return 1
    if args.sleep_seconds < 0:
        print("Error: --sleep-seconds must be zero or positive", file=sys.stderr)
        return 1

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.work_output_root).mkdir(parents=True, exist_ok=True)

    rag_process = None
    rag_log_handle = None
    try:
        rag_process, rag_log_handle = ensure_rag_server(args)
        runtime = load_reinfix_runtime()
        runtime["ensure_environment"]()
        contexts, selection = select_contexts(args)
    except (
        EnvironmentError,
        ImportError,
        ProgramFunctionLookupError,
        RuntimeError,
        TimeoutError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        stop_rag_server(rag_process, rag_log_handle)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary = build_initial_summary(args, selection, contexts)
    write_summary(args.summary_output, summary)

    print(
        f"Processing {len(contexts)} datasetForExperiment programs x {args.runs} independent runs "
        f"with agent_model={DEFAULT_AGENT_MODEL_NAME}, patch_model={DEFAULT_PATCH_MODEL_NAME}"
    )
    print(f"RAG search URL: {args.rag_search_url}")
    print(f"Consolidated fix_code JSON results: {args.results_dir}")
    print(
        "Validation command template: "
        "python3 eval_ConDefects/validate_fix_code_results_ochiai.py "
        f"--dataset-root {args.dataset_root} --results-dir {args.results_dir}"
    )

    original_stdout = sys.stdout

    for program_index, context in enumerate(contexts, start=1):
        program_record = {
            "programId": context.program_id,
            "task_name": context.task_name,
            "language": context.language,
            "runs": [],
        }
        existing_runs = existing_repair_runs(args.results_dir, context.program_id)

        for repair_run in range(1, args.runs + 1):
            if repair_run in existing_runs and not args.overwrite:
                summary["skipped_runs"] += 1
                run_record = {
                    "repair_run": repair_run,
                    "status": "skipped",
                    "reason": "consolidated result JSON already exists for this repair_run",
                }
                program_record["runs"].append(run_record)
                update_program_record(summary, program_record)
                write_summary(args.summary_output, summary)
                print(
                    f"[{program_index}/{len(contexts)}] [skip] "
                    f"{context.program_id} repair_run={repair_run}"
                )
                continue

            if args.overwrite:
                delete_run_results(args.results_dir, context.program_id, repair_run)

            log_path = (
                Path(args.work_output_root)
                / "logs"
                / f"{safe_filename(context.program_id)}_repair_{repair_run:02d}.log"
            )
            logger = Logger(str(log_path), original_stdout)
            sys.stdout = logger
            started_at = time.monotonic()
            attempt_root: Path | None = None
            try:
                logger.info(
                    f"[{program_index}/{len(contexts)}] repair programId={context.program_id}, "
                    f"repair_run={repair_run}/{args.runs}"
                )
                written_patch_jsons, attempt_root = run_one_independent_repair(
                    context=context,
                    repair_run=repair_run,
                    args=args,
                    logger=logger,
                    runtime=runtime,
                )
                elapsed_seconds = round(time.monotonic() - started_at, 3)
                summary["completed_runs"] += 1
                summary["written_patch_jsons"] += written_patch_jsons
                run_record = {
                    "repair_run": repair_run,
                    "status": "success" if written_patch_jsons else "no_patch",
                    "written_patch_jsons": written_patch_jsons,
                    "attempt_work_dir": str(attempt_root),
                    "log_path": str(log_path),
                    "elapsed_seconds": elapsed_seconds,
                }
                logger.info(
                    f"[ok] {context.program_id} repair_run={repair_run}/{args.runs}, "
                    f"written_patch_jsons={written_patch_jsons}, elapsed={elapsed_seconds}s"
                )
            except Exception as exc:
                elapsed_seconds = round(time.monotonic() - started_at, 3)
                summary["failed_runs"] += 1
                error_path = write_error_result(
                    context=context,
                    results_dir=args.results_dir,
                    repair_run=repair_run,
                    error=exc,
                    attempt_root=attempt_root,
                )
                run_record = {
                    "repair_run": repair_run,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_result_json": str(error_path),
                    "attempt_work_dir": str(attempt_root) if attempt_root else None,
                    "log_path": str(log_path),
                    "elapsed_seconds": elapsed_seconds,
                }
                logger.error(
                    f"[error] {context.program_id} repair_run={repair_run}/{args.runs}: {exc}"
                )
                if args.stop_on_error:
                    program_record["runs"].append(run_record)
                    update_program_record(summary, program_record)
                    summary["finished_at"] = datetime.now().isoformat()
                    write_summary(args.summary_output, summary)
                    stop_rag_server(rag_process, rag_log_handle)
                    return 1
            finally:
                sys.stdout = original_stdout
                logger.close()

            program_record["runs"].append(run_record)
            update_program_record(summary, program_record)
            write_summary(args.summary_output, summary)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    summary["finished_at"] = datetime.now().isoformat()
    write_summary(args.summary_output, summary)
    print(
        "Finished batch repair: "
        f"completed_runs={summary['completed_runs']}, skipped_runs={summary['skipped_runs']}, "
        f"failed_runs={summary['failed_runs']}, written_patch_jsons={summary['written_patch_jsons']}, "
        f"programs={summary['total_programs']}"
    )
    print(f"Summary written to {args.summary_output}")
    print(f"Patch JSON directory for validation: {args.results_dir}")
    stop_rag_server(rag_process, rag_log_handle)
    return 1 if summary["failed_runs"] and args.stop_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
