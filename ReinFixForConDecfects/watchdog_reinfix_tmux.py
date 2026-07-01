from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
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
    DEFAULT_PATCH_MODEL_NAME,
)
from ReinFixForConDecfects.src.condefects_dataset import (  # noqa: E402
    ProgramFunctionLookupError,
    iter_program_contexts,
    resolve_program_context,
)


DEFAULT_SESSION = "reinfix"
DEFAULT_BATCH_SCRIPT = PACKAGE_ROOT / "batch_repair_dataset_2024.py"
DEFAULT_VENV_ACTIVATE = REPO_ROOT / ".venv" / "bin" / "activate"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "output"
DEFAULT_WORK_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "batch_repair_dataset_2024_work"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "batch_repair_dataset_2024_fix_code_results"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "batch_repair_dataset_2024_summary.json"
DEFAULT_WATCHDOG_LOG = DEFAULT_OUTPUT_ROOT / "reinfix_watchdog.log"
DEFAULT_INTERVAL_SECONDS = 60.0
DEFAULT_RUNS = 5


@dataclass(slots=True)
class FailureTarget:
    program_id: str
    repair_run: int | None
    source: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch tmux session 'reinfix' and restart ConDefects ReInFix batch repair "
            "when the session is missing or the batch script is no longer running."
        )
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true", help="Run one health check and exit.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--venv-activate", default=str(DEFAULT_VENV_ACTIVATE))
    parser.add_argument("--batch-script", default=str(DEFAULT_BATCH_SCRIPT))
    parser.add_argument(
        "--batch-args",
        default="",
        help=(
            "Extra args for batch_repair_dataset_2024.py. The default is empty so restarts "
            "continue from existing successful/watchdog-skipped runs."
        ),
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--work-output-root", default=str(DEFAULT_WORK_OUTPUT_ROOT))
    parser.add_argument("--watchdog-log", default=str(DEFAULT_WATCHDOG_LOG))
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--restart-after-finished",
        action="store_true",
        help="Restart even when the summary has finished_at. Default: false.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval_seconds <= 0:
        print("Error: --interval-seconds must be positive", file=sys.stderr)
        return 1
    if args.runs < 1:
        print("Error: --runs must be positive", file=sys.stderr)
        return 1

    while True:
        try:
            keep_running = check_once(args)
        except Exception as exc:  # Keep the watchdog alive even if one check fails.
            log(args, f"[watchdog-error] {type(exc).__name__}: {exc}")
            keep_running = True

        if args.once or not keep_running:
            return 0
        time.sleep(args.interval_seconds)


def check_once(args: argparse.Namespace) -> bool:
    if batch_finished(args) and not args.restart_after_finished:
        log(args, "batch summary has finished_at; watchdog is exiting")
        return False

    session_exists = tmux_session_exists(args.session)
    script_running = session_exists and batch_script_running_in_session(args)
    if session_exists and script_running:
        log(args, f"healthy: tmux session {args.session!r} exists and batch script is running")
        return True

    reason = (
        f"tmux session {args.session!r} does not exist"
        if not session_exists
        else f"batch script is not running in tmux session {args.session!r}"
    )
    log(args, f"unhealthy: {reason}")

    captured = capture_tmux(args.session) if session_exists else ""
    target = infer_failure_target(args, captured)
    if target is not None:
        write_watchdog_error_json(args, target, reason)
    else:
        log(args, "could not infer failing program; restarting without writing error JSON")

    recreate_tmux_session(args)
    return True


def tmux_session_exists(session: str) -> bool:
    result = run_command(["tmux", "has-session", "-t", session])
    return result.returncode == 0


def batch_script_running_in_session(args: argparse.Namespace) -> bool:
    pane_pids_result = run_command(["tmux", "list-panes", "-t", args.session, "-F", "#{pane_pid}"])
    if pane_pids_result.returncode != 0:
        return False
    pane_pids = {
        int(line.strip())
        for line in pane_pids_result.stdout.splitlines()
        if line.strip().isdigit()
    }
    if not pane_pids:
        return False

    processes = read_process_table()
    descendants = descendant_pids(pane_pids, processes)
    script_name = Path(args.batch_script).name
    for pid in descendants:
        command = processes.get(pid, {}).get("args", "")
        if script_name in command and "python" in command:
            return True
    return False


def read_process_table() -> dict[int, dict[str, Any]]:
    result = run_command(["ps", "-eo", "pid=,ppid=,args="])
    processes: dict[int, dict[str, Any]] = {}
    if result.returncode != 0:
        return processes
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        processes[int(parts[0])] = {
            "ppid": int(parts[1]),
            "args": parts[2] if len(parts) == 3 else "",
        }
    return processes


def descendant_pids(root_pids: set[int], processes: dict[int, dict[str, Any]]) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, info in processes.items():
        children.setdefault(int(info["ppid"]), []).append(pid)

    found: set[int] = set(root_pids)
    stack = list(root_pids)
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def capture_tmux(session: str) -> str:
    result = run_command(["tmux", "capture-pane", "-p", "-t", session, "-S", "-2000"])
    return result.stdout if result.returncode == 0 else ""


def infer_failure_target(args: argparse.Namespace, captured: str) -> FailureTarget | None:
    for source_name, text in (
        ("tmux_capture", captured),
        ("latest_log", latest_log_text(args)),
    ):
        target = infer_from_text(text, source_name)
        if target is not None:
            if target.repair_run is None:
                target.repair_run = infer_repair_run(args, target.program_id)
            return target

    target = infer_from_summary(args)
    if target is not None:
        return target

    first_program = first_dataset_program(args)
    if first_program:
        return FailureTarget(
            program_id=first_program,
            repair_run=infer_repair_run(args, first_program),
            source="dataset",
            detail="fallback to first datasetForExperiment program",
        )
    return None


def infer_from_text(text: str, source: str) -> FailureTarget | None:
    if not text:
        return None
    patterns = [
        r"repair programId=(?P<pid>\d+),\s*repair_run=(?P<run>\d+)/(?P<runs>\d+)",
        r"\[error\]\s+(?P<pid>\d+)\s+repair_run=(?P<run>\d+)/(?P<runs>\d+)",
        r"(?P<pid>\d+)\s+repair_run=(?P<run>\d+)/(?P<runs>\d+)",
        r"Generating Solution for:\s*(?P<pid>\d+),\s*Round:\s*(?P<round>\d+)",
    ]
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text))
    if not matches:
        return None
    match = max(matches, key=lambda item: item.start())
    repair_run = match.groupdict().get("run")
    return FailureTarget(
        program_id=match.group("pid"),
        repair_run=int(repair_run) if repair_run and repair_run.isdigit() else None,
        source=source,
        detail=match.group(0),
    )


def latest_log_text(args: argparse.Namespace) -> str:
    logs_dir = Path(args.work_output_root) / "logs"
    if not logs_dir.is_dir():
        return ""
    logs = sorted(logs_dir.glob("*_repair_*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return ""
    latest = logs[-1]
    filename_match = re.search(r"(?P<pid>\d+)_repair_(?P<run>\d+)\.log$", latest.name)
    prefix = ""
    if filename_match:
        prefix = (
            f"repair programId={filename_match.group('pid')}, "
            f"repair_run={int(filename_match.group('run'))}/{get_runs_per_program(args)}\n"
        )
    try:
        return prefix + latest.read_text(encoding="utf-8", errors="ignore")[-20000:]
    except OSError:
        return prefix


def infer_from_summary(args: argparse.Namespace) -> FailureTarget | None:
    summary = read_json_if_exists(args.summary_output)
    if not summary:
        return None
    programs = [record for record in summary.get("programs", []) if isinstance(record, dict)]
    runs_per_program = int(summary.get("runs_per_program") or args.runs)
    if programs:
        last = programs[-1]
        program_id = str(last.get("programId") or "").strip()
        if program_id:
            runs = [run for run in last.get("runs", []) if isinstance(run, dict)]
            next_run = min(len(runs) + 1, runs_per_program)
            return FailureTarget(
                program_id=program_id,
                repair_run=next_run,
                source="summary",
                detail="last program record in summary",
            )

    first_program = first_dataset_program(args)
    if first_program:
        return FailureTarget(
            program_id=first_program,
            repair_run=1,
            source="summary",
            detail="empty summary; fallback to first datasetForExperiment program",
        )
    return None


def first_dataset_program(args: argparse.Namespace) -> str | None:
    try:
        contexts = iter_program_contexts(dataset_root=args.dataset_root, require_ochiai=True)
    except (ProgramFunctionLookupError, OSError, ValueError):
        return None
    return contexts[0].program_id if contexts else None


def infer_repair_run(args: argparse.Namespace, program_id: str) -> int:
    completed = completed_or_watchdog_skipped_runs(args, program_id)
    for repair_run in range(1, get_runs_per_program(args) + 1):
        if repair_run not in completed:
            return repair_run
    return get_runs_per_program(args)


def get_runs_per_program(args: argparse.Namespace) -> int:
    summary = read_json_if_exists(args.summary_output)
    if summary and isinstance(summary.get("runs_per_program"), int):
        return int(summary["runs_per_program"])
    return args.runs


def completed_or_watchdog_skipped_runs(args: argparse.Namespace, program_id: str) -> set[int]:
    runs: set[int] = set()
    for payload in result_payloads_for_program(args, program_id):
        repair_run = payload.get("repair_run")
        if not isinstance(repair_run, int):
            continue
        is_success = payload.get("status") == "success" and isinstance(payload.get("fix_code"), str)
        is_watchdog_skip = payload.get("status") == "error" and payload.get("watchdog_skip") is True
        if is_success or is_watchdog_skip:
            runs.add(repair_run)
    return runs


def result_payloads_for_program(args: argparse.Namespace, program_id: str) -> list[dict[str, Any]]:
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        return []
    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        payload = read_json_if_exists(path)
        if payload and str(payload.get("programId") or "") == program_id:
            payloads.append(payload)
    return payloads


def write_watchdog_error_json(args: argparse.Namespace, target: FailureTarget, reason: str) -> None:
    program_id = target.program_id
    repair_run = target.repair_run or infer_repair_run(args, program_id)
    for payload in result_payloads_for_program(args, program_id):
        if payload.get("repair_run") == repair_run and payload.get("watchdog_skip") is True:
            log(args, f"watchdog error already exists for programId={program_id} repair_run={repair_run}")
            return
        if payload.get("repair_run") == repair_run and payload.get("status") == "success":
            log(args, f"success already exists for programId={program_id} repair_run={repair_run}; no error JSON written")
            return

    repair_index = next_repair_index(args, program_id)
    context_payload = context_metadata(args, program_id)
    output_path = result_path(args, context_payload, repair_index)
    payload = {
        "status": "error",
        "programId": program_id,
        "repair_index": repair_index,
        "repair_run": repair_run,
        "task_name": context_payload.get("task_name"),
        "language": context_payload.get("language"),
        "agent_model": DEFAULT_AGENT_MODEL_NAME,
        "patch_model": DEFAULT_PATCH_MODEL_NAME,
        "error_type": "WatchdogDetectedBatchExit",
        "error": reason,
        "watchdog_skip": True,
        "watchdog_detected_at": datetime.now().isoformat(),
        "watchdog_inferred_from": target.source,
        "watchdog_inference_detail": target.detail,
    }
    write_json(output_path, payload)
    log(args, f"wrote watchdog error JSON: {output_path}")


def context_metadata(args: argparse.Namespace, program_id: str) -> dict[str, str | None]:
    try:
        context = resolve_program_context(program_id, dataset_root=args.dataset_root, require_ochiai=False)
        return {
            "program_id": program_id,
            "task_name": context.task_name,
            "language": context.language,
        }
    except (ProgramFunctionLookupError, ValueError):
        return {"program_id": program_id, "task_name": None, "language": None}


def next_repair_index(args: argparse.Namespace, program_id: str) -> int:
    max_index = 0
    for payload in result_payloads_for_program(args, program_id):
        repair_index = payload.get("repair_index")
        if isinstance(repair_index, int):
            max_index = max(max_index, repair_index)
    return max_index + 1


def result_path(args: argparse.Namespace, context_payload: dict[str, str | None], repair_index: int) -> Path:
    task_name = safe_filename(context_payload.get("task_name") or "unknown_task")
    language = safe_filename(context_payload.get("language") or "unknown_language")
    program_id = safe_filename(str(context_payload.get("program_id") or ""))
    if not program_id:
        program_id = "unknown_program"
    return Path(args.results_dir) / f"{task_name}_{language}_{program_id}_repair_{repair_index:03d}.json"


def recreate_tmux_session(args: argparse.Namespace) -> None:
    if tmux_session_exists(args.session):
        run_command(["tmux", "kill-session", "-t", args.session])

    command = build_batch_command(args)
    result = run_command(["tmux", "new-session", "-d", "-s", args.session, command])
    if result.returncode != 0:
        raise RuntimeError(f"could not create tmux session {args.session!r}: {result.stderr.strip()}")
    log(args, f"recreated tmux session {args.session!r}: {command}")


def build_batch_command(args: argparse.Namespace) -> str:
    batch_parts = [
        "python3",
        str(Path(args.batch_script).relative_to(args.repo_root)),
        "--dataset-root",
        str(args.dataset_root),
        "--results-dir",
        str(args.results_dir),
        "--summary-output",
        str(args.summary_output),
        "--work-output-root",
        str(args.work_output_root),
        "--runs",
        str(args.runs),
    ]
    if args.batch_args.strip():
        batch_parts.extend(shlex.split(args.batch_args))
    batch_command = " ".join(shlex.quote(part) for part in batch_parts)
    return (
        "bash -lc "
        + shlex.quote(
            f"cd {shlex.quote(args.repo_root)} "
            f"&& source {shlex.quote(args.venv_activate)} "
            f"&& {batch_command}"
        )
    )


def batch_finished(args: argparse.Namespace) -> bool:
    summary = read_json_if_exists(args.summary_output)
    return bool(summary and summary.get("finished_at"))


def read_json_if_exists(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path = Path(args.watchdog_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
