#!/usr/bin/env python3
"""Watch and restart SpecRover ConDefects datasetForExperiment batch in tmux.

Typical usage:

  nohup python3 SpecRoverForConDecfects/watchdog_specrover_tmux.py \
    > SpecRoverForConDecfects/output/specrover_datasetForExperiment_watchdog.nohup.log 2>&1 &

Extra batch arguments can be supplied after "--":

  python3 SpecRoverForConDecfects/watchdog_specrover_tmux.py -- \
    --limit 20 --sleep-seconds 1
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpecRoverForConDecfects.batch_repair_repair_list_dataset_2024 import (  # noqa: E402
    safe_filename,
)
from SpecRoverForConDecfects.condefects_dataset import (  # noqa: E402
    ProgramFunctionLookupError,
    resolve_program_context,
)
from SpecRoverForConDecfects.config import DEFAULT_DATASET_ROOT  # noqa: E402
from SpecRoverForConDecfects.validation import (  # noqa: E402
    DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
    DEFAULT_VALIDATION_TIMEOUT_SECONDS,
)


DEFAULT_SESSION = "specrover_datasetForExperiment"
DEFAULT_BATCH_SCRIPT = PACKAGE_ROOT / "batch_repair_repair_list_dataset_2024.py"
DEFAULT_VENV_ACTIVATE = REPO_ROOT / ".venv" / "bin" / "activate"
DEFAULT_PYTHON = (
    REPO_ROOT / ".venv" / "bin" / "python"
    if (REPO_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "output"
DEFAULT_FINAL_REPORT_DIR = DEFAULT_OUTPUT_ROOT / "eval_specrover_validation_feedback_module_Ochiai"
DEFAULT_WORK_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "batch_repair_datasetForExperiment_work"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "batch_repair_datasetForExperiment_summary.json"
DEFAULT_TASK_INFO_CACHE = DEFAULT_OUTPUT_ROOT / "task_information_cache.json"
DEFAULT_WATCHDOG_LOG = DEFAULT_OUTPUT_ROOT / "specrover_datasetForExperiment_watchdog.log"
DEFAULT_RUN_LOG = DEFAULT_OUTPUT_ROOT / "specrover_datasetForExperiment_run.log"
DEFAULT_INTERVAL_SECONDS = 60.0
DEFAULT_GRACE_SECONDS = 5.0


@dataclass(slots=True)
class FailureTarget:
    program_id: str
    task_name: str | None
    language: str | None
    source: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor a tmux session running SpecRoverForConDecfects/"
            "batch_repair_repair_list_dataset_2024.py. If the session disappears "
            "or the batch script is no longer running, write a watchdog failure "
            "report for the inferred current program and restart the batch."
        )
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--grace-seconds", type=float, default=DEFAULT_GRACE_SECONDS)
    parser.add_argument("--restart-delay", type=float, default=5.0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--once", action="store_true", help="Run one health check and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without changing tmux or reports.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--venv-activate", default=str(DEFAULT_VENV_ACTIVATE))
    parser.add_argument("--no-venv-activate", action="store_true")
    parser.add_argument("--batch-script", default=str(DEFAULT_BATCH_SCRIPT))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--final-report-dir", default=str(DEFAULT_FINAL_REPORT_DIR))
    parser.add_argument("--work-output-root", default=str(DEFAULT_WORK_OUTPUT_ROOT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--task-info-cache", default=str(DEFAULT_TASK_INFO_CACHE))
    parser.add_argument("--watchdog-log", default=str(DEFAULT_WATCHDOG_LOG))
    parser.add_argument("--run-log", default=str(DEFAULT_RUN_LOG))
    parser.add_argument("--state-file", default=None)
    parser.add_argument(
        "--test-timeout-seconds",
        type=float,
        default=DEFAULT_VALIDATION_TIMEOUT_SECONDS,
        help=f"Forwarded to the batch script. Default: {DEFAULT_VALIDATION_TIMEOUT_SECONDS:g}.",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
        help=f"Forwarded to the batch script. Default: {DEFAULT_VALIDATION_MEMORY_LIMIT_MB}.",
    )
    parser.add_argument(
        "--restart-after-finished",
        action="store_true",
        help="Restart even when the summary has finished_at. Default: false.",
    )
    parser.add_argument(
        "--restart-on-clean-exit",
        action="store_true",
        help="Restart even when the tmux wrapper recorded exit_status=0.",
    )
    parser.add_argument(
        "--no-watchdog-error-report",
        action="store_true",
        help="Restart without writing a watchdog error report for the inferred current program.",
    )
    parser.add_argument(
        "batch_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the batch script after a literal '--'.",
    )
    args = parser.parse_args()
    if args.batch_args and args.batch_args[0] == "--":
        args.batch_args = args.batch_args[1:]
    return args


def main() -> int:
    args = parse_args()
    validate_args(args)
    restarts = 0

    while True:
        try:
            action = check_once(args)
        except Exception as exc:  # Keep the watchdog alive even if one check fails.
            log(args, f"[watchdog-error] {type(exc).__name__}: {exc}")
            action = "keep_running"

        if action == "restarted":
            restarts += 1
            if args.max_restarts and restarts >= args.max_restarts:
                log(args, f"max restarts reached: {args.max_restarts}; watchdog is exiting")
                return 0
            if args.restart_delay:
                time.sleep(args.restart_delay)

        if args.once or action == "stop":
            return 0
        time.sleep(args.interval_seconds)


def validate_args(args: argparse.Namespace) -> None:
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if args.grace_seconds < 0:
        raise SystemExit("--grace-seconds cannot be negative")
    if args.restart_delay < 0:
        raise SystemExit("--restart-delay cannot be negative")
    if args.max_restarts < 0:
        raise SystemExit("--max-restarts cannot be negative")
    if args.test_timeout_seconds <= 0:
        raise SystemExit("--test-timeout-seconds must be positive")
    if args.memory_limit_mb < 0:
        raise SystemExit("--memory-limit-mb cannot be negative")

    forwarded_options = {item.split("=", 1)[0] for item in args.batch_args}
    blocked_options = {
        "--dataset-root",
        "--final-report-dir",
        "--work-output-root",
        "--summary-output",
        "--task-info-cache",
        "--test-timeout-seconds",
        "--memory-limit-mb",
    }
    overlap = sorted(blocked_options & forwarded_options)
    if overlap:
        raise SystemExit(
            "Use watchdog options instead of passing these after '--': "
            + ", ".join(overlap)
        )
    if "--overwrite" in forwarded_options:
        raise SystemExit(
            "Do not pass --overwrite through the watchdog; restarts rely on "
            "existing final reports to resume safely."
        )


def check_once(args: argparse.Namespace) -> str:
    if batch_finished(args) and not args.restart_after_finished:
        log(args, "batch summary has finished_at; watchdog is exiting")
        return "stop"

    session_exists_now = tmux_session_exists(args.session)
    script_running = session_exists_now and batch_script_running_in_session(args)
    if session_exists_now and script_running:
        log(args, f"healthy: tmux session {args.session!r} exists and batch script is running")
        return "keep_running"

    if args.grace_seconds:
        time.sleep(args.grace_seconds)
        session_exists_now = tmux_session_exists(args.session)
        script_running = session_exists_now and batch_script_running_in_session(args)
        if session_exists_now and script_running:
            log(args, f"healthy after grace period: tmux session {args.session!r}")
            return "keep_running"

    reason = (
        f"tmux session {args.session!r} does not exist"
        if not session_exists_now
        else f"batch script is not running in tmux session {args.session!r}"
    )
    state = read_state_file(args)
    if state:
        reason = f"{reason}; last_exit={state.get('status')} ended_at={state.get('ended_at')}"
        if (
            state.get("status") == "0"
            and not args.restart_on_clean_exit
            and batch_finished(args)
            and not args.restart_after_finished
        ):
            log(args, f"clean exit with finished summary; watchdog is exiting ({reason})")
            return "stop"

    log(args, f"unhealthy: {reason}")

    captured = capture_tmux(args.session) if session_exists_now else ""
    target = infer_failure_target(args, captured)
    if target and not args.no_watchdog_error_report:
        write_watchdog_error_report(args, target, reason)
    elif not target:
        log(args, "could not infer current program; restarting without writing an error report")

    recreate_tmux_session(args)
    return "restarted"


def tmux_session_exists(session: str) -> bool:
    return run_command(["tmux", "has-session", "-t", session]).returncode == 0


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
        ("run_log", tail_text(args.run_log)),
    ):
        target = infer_from_text(text, source_name)
        if target:
            return enrich_target(args, target)

    target = infer_from_running_report(args)
    if target:
        return target

    return infer_from_summary(args)


def infer_from_text(text: str, source: str) -> FailureTarget | None:
    if not text:
        return None

    patterns = [
        (
            r"\[(?P<idx>\d+)/(?P<total>\d+)\]\s+repairing\s+"
            r"(?P<task>[^/\s]+)/(?P<language>[^/\s]+)/(?P<pid>\d+)"
        ),
        r"programId[=:]\s*(?P<pid>\d+)",
        r"program_id[=:]\s*(?P<pid>\d+)",
    ]
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text))
    if not matches:
        return None
    match = max(matches, key=lambda item: item.start())
    return FailureTarget(
        program_id=match.group("pid"),
        task_name=match.groupdict().get("task"),
        language=match.groupdict().get("language"),
        source=source,
        detail=match.group(0),
    )


def infer_from_running_report(args: argparse.Namespace) -> FailureTarget | None:
    report_dir = Path(args.final_report_dir)
    if not report_dir.is_dir():
        return None
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in report_dir.glob("*.json"):
        payload = read_json_if_exists(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "running" or payload.get("finished_at") is None:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((mtime, path, payload))
    if not candidates:
        return None

    _, path, payload = max(candidates, key=lambda item: item[0])
    program_id = str(payload.get("programId") or "").strip()
    if not program_id:
        return None
    return FailureTarget(
        program_id=program_id,
        task_name=as_optional_str(payload.get("task_name")),
        language=as_optional_str(payload.get("language")),
        source="running_report",
        detail=str(path),
    )


def infer_from_summary(args: argparse.Namespace) -> FailureTarget | None:
    summary = read_json_if_exists(args.summary_output)
    if not isinstance(summary, dict):
        return None
    programs = [record for record in summary.get("programs", []) if isinstance(record, dict)]
    if not programs:
        return None

    last = programs[-1]
    program_id = str(last.get("programId") or "").strip()
    if not program_id:
        return None
    return FailureTarget(
        program_id=program_id,
        task_name=as_optional_str(last.get("task_name")),
        language=as_optional_str(last.get("language")),
        source="summary",
        detail="last completed or skipped program in summary",
    )


def enrich_target(args: argparse.Namespace, target: FailureTarget) -> FailureTarget:
    if target.task_name and target.language:
        return target
    metadata = context_metadata(args, target.program_id)
    return FailureTarget(
        program_id=target.program_id,
        task_name=target.task_name or metadata.get("task_name"),
        language=target.language or metadata.get("language"),
        source=target.source,
        detail=target.detail,
    )


def write_watchdog_error_report(args: argparse.Namespace, target: FailureTarget, reason: str) -> None:
    metadata = context_metadata(args, target.program_id)
    task_name = target.task_name or metadata.get("task_name") or "unknown_task"
    language = target.language or metadata.get("language") or "unknown_language"
    report_path = final_report_path(args.final_report_dir, task_name, language, target.program_id)
    existing = read_json_if_exists(report_path)
    if isinstance(existing, dict):
        if existing.get("pass") is True or existing.get("status") == "success":
            log(args, f"success report already exists for programId={target.program_id}; no error report written")
            return
        if existing.get("status") not in {None, "running", "watchdog_error"}:
            log(
                args,
                f"non-running report already exists for programId={target.program_id} "
                f"status={existing.get('status')!r}; no error report written",
            )
            return

    log(
        args,
        "watchdog detected an unfinished program but final reports are success-only; "
        f"no watchdog error JSON written for programId={target.program_id}: {reason}",
    )


def context_metadata(args: argparse.Namespace, program_id: str) -> dict[str, str | None]:
    try:
        context = resolve_program_context(
            program_id,
            dataset_root=args.dataset_root,
            require_fault_location=False,
        )
        return {
            "program_id": program_id,
            "task_name": context.task_name,
            "language": context.language,
        }
    except (ProgramFunctionLookupError, ValueError):
        return {"program_id": program_id, "task_name": None, "language": None}


def recreate_tmux_session(args: argparse.Namespace) -> None:
    if args.dry_run:
        log(args, f"dry-run: would recreate tmux session {args.session!r}")
        log(args, f"dry-run command: {build_tmux_command(args)}")
        return
    if tmux_session_exists(args.session):
        run_command(["tmux", "kill-session", "-t", args.session])

    command = build_tmux_command(args)
    result = run_command(["tmux", "new-session", "-d", "-s", args.session, command])
    if result.returncode != 0:
        raise RuntimeError(f"could not create tmux session {args.session!r}: {result.stderr.strip()}")
    log(args, f"recreated tmux session {args.session!r}: {command}")


def build_tmux_command(args: argparse.Namespace) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_command = " ".join(shlex.quote(part) for part in build_batch_argv(args))
    repo_root = shlex.quote(str(args.repo_root))
    run_log = shlex.quote(str(args.run_log))
    state_file = shlex.quote(str(state_file_path(args)))
    output_dir = shlex.quote(str(Path(args.run_log).parent))

    venv_block = ""
    if not args.no_venv_activate:
        activate = shlex.quote(str(args.venv_activate))
        venv_block = (
            f"if [ -f {activate} ]; then . {activate}; "
            f"else echo '[watchdog] venv activate not found: {activate}'; fi; "
        )

    inner = (
        f"mkdir -p {output_dir}; "
        "{ "
        f"echo '[watchdog] run_id={run_id} session={args.session} started_at='$(date -Is); "
        f"cd {repo_root}; "
        f"{venv_block}"
        f"{batch_command}; "
        "status=$?; "
        f"echo '[watchdog] run_id={run_id} exit_status='${{status}}' ended_at='$(date -Is); "
        "printf 'run_id=%s\\nsession=%s\\nstatus=%s\\nended_at=%s\\n' "
        f"{shlex.quote(run_id)} {shlex.quote(args.session)} "
        "\"$status\" \"$(date -Is)\" "
        f"> {state_file}; "
        "exit \"$status\"; "
        f"}} >> {run_log} 2>&1"
    )
    return "bash -lc " + shlex.quote(inner)


def build_batch_argv(args: argparse.Namespace) -> list[str]:
    try:
        script = str(Path(args.batch_script).resolve().relative_to(Path(args.repo_root).resolve()))
    except ValueError:
        script = str(Path(args.batch_script).resolve())
    argv = [
        str(args.python),
        script,
        "--dataset-root",
        str(args.dataset_root),
        "--final-report-dir",
        str(args.final_report_dir),
        "--work-output-root",
        str(args.work_output_root),
        "--summary-output",
        str(args.summary_output),
        "--task-info-cache",
        str(args.task_info_cache),
        "--test-timeout-seconds",
        f"{args.test_timeout_seconds:g}",
        "--memory-limit-mb",
        str(args.memory_limit_mb),
    ]
    argv.extend(args.batch_args)
    return argv


def batch_finished(args: argparse.Namespace) -> bool:
    summary = read_json_if_exists(args.summary_output)
    return bool(isinstance(summary, dict) and summary.get("finished_at"))


def state_file_path(args: argparse.Namespace) -> Path:
    if args.state_file:
        return Path(args.state_file)
    return Path(args.summary_output).parent / f".{args.session}.last_exit"


def read_state_file(args: argparse.Namespace) -> dict[str, str] | None:
    path = state_file_path(args)
    if not path.exists():
        return None
    state: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                state[key] = value
    except OSError:
        return None
    return state


def final_report_path(
    final_report_dir: str | Path,
    task_name: str,
    language: str,
    program_id: str,
) -> Path:
    filename = (
        f"{safe_filename(task_name)}_"
        f"{safe_filename(language)}_"
        f"{safe_filename(program_id)}.json"
    )
    return Path(final_report_dir) / filename


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


def tail_text(path: str | Path, limit: int = 40000) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[-limit:]
    except OSError:
        return ""


def as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if args.dry_run:
        return
    path = Path(args.watchdog_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
