from __future__ import annotations

import contextlib
import json
import os
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATASET_ROOT, DEFAULT_TEST_ROOT
from .condefects_dataset import (
    OCHIAI_FL_FILENAME,
    FaultyFunctionLocation,
    ProgramContext,
)
from .data_structures import ReproResult


DEFAULT_VALIDATION_TIMEOUT_SECONDS = float(
    os.environ.get("SPECROVER_CONDEFECTS_VALIDATION_TIMEOUT_SECONDS", "300")
)
DEFAULT_VALIDATION_MEMORY_LIMIT_MB = int(
    os.environ.get("SPECROVER_CONDEFECTS_VALIDATION_MEMORY_LIMIT_MB", "4096")
)
MEMORY_ERROR_MARKERS = (
    "memoryerror",
    "outofmemoryerror",
    "cannot allocate memory",
    "could not reserve enough space",
    "could not allocate metaspace",
    "native memory allocation",
    "std::bad_alloc",
    "memory limit exceeded",
)
MAX_FAILURE_TEXT_LENGTH = 4000
MAX_SAVED_VALIDATION_FAILURE_CASES = 3
MONITOR_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class TestCase:
    name: str
    input_path: Path
    expected_output_path: Path
    input_text: str
    expected_output: str

    def to_prompt(self, actual_output: str | None = None) -> str:
        parts = [
            f"ConDefects test case: {self.name}",
            "",
            "Input:",
            truncate_text(self.input_text),
            "",
            "Expected output:",
            truncate_text(self.expected_output),
        ]
        if actual_output is not None:
            parts.extend(
                [
                    "",
                    "Actual output on the buggy program:",
                    truncate_text(actual_output),
                ]
            )
        return "\n".join(parts)


@dataclass(frozen=True)
class GuardedProcessResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    memory_exceeded: bool
    peak_rss_mb: float


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    message: str
    pass_rate: float | None
    result_dir: str | None
    raw_record: dict[str, Any]
    timed_out: bool = False
    memory_exceeded: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "message": self.message,
            "pass_rate": self.pass_rate,
            "result_dir": self.result_dir,
            "timed_out": self.timed_out,
            "memory_exceeded": self.memory_exceeded,
            "raw_record": self.raw_record,
        }


def evaluate_patch(
    context: ProgramContext,
    location: FaultyFunctionLocation,
    fix_code: str,
    output_dir: str | Path,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    timeout_seconds: float = DEFAULT_VALIDATION_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
) -> ValidationResult:
    temp_root = Path(tempfile.mkdtemp(prefix=f"specrover_val_{context.program_id}_"))
    result_output_dir = Path(output_dir) / "validation"
    result_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _, temp_program_dir = copy_program_to_temp_dataset(
            context,
            dataset_root,
            temp_root,
        )
        temp_source_path = temp_program_dir / location.source_path.name
        replace_source_range(
            temp_source_path,
            location.start_line_index,
            location.end_line_index,
            fix_code,
        )

        record = run_validation_test_cases(
            context=context,
            temp_source_path=temp_source_path,
            temp_program_dir=temp_program_dir,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )
        record["result_dir"] = str(result_output_dir)
        write_validation_artifacts(result_output_dir, record)

        pass_rate = record.get("pass_rate")
        passed = record.get("pass") is True
        message = validation_message(record)
        return ValidationResult(
            passed=passed,
            message=message,
            pass_rate=pass_rate if isinstance(pass_rate, float) else None,
            result_dir=str(result_output_dir),
            raw_record=record,
            timed_out=record.get("timed_out") is True,
            memory_exceeded=record.get("memory_exceeded") is True,
        )
    except Exception as exc:
        memory_exceeded = is_memory_exception(exc)
        record = {
            "task": context.task_name,
            "language": context.language,
            "program_id": context.program_id,
            "status": "memory_exceeded" if memory_exceeded else "error",
            "pass": False,
            "pass_rate": None,
            "tests_run": 0,
            "failed_tests": [],
            "error_tests": [
                {
                    "name": f"{context.task_name}/validation_exception",
                    "status": "error",
                    "phase": "memory" if memory_exceeded else "validation",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
            "timed_out": False,
            "memory_exceeded": memory_exceeded,
            "timeout_seconds": timeout_seconds,
            "memory_limit_mb": memory_limit_mb,
            "result_dir": str(result_output_dir),
        }
        write_validation_artifacts(result_output_dir, record)
        return ValidationResult(
            passed=False,
            message=validation_message(record),
            pass_rate=None,
            result_dir=str(result_output_dir),
            raw_record=record,
            timed_out=False,
            memory_exceeded=memory_exceeded,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def run_validation_test_cases(
    context: ProgramContext,
    temp_source_path: Path,
    temp_program_dir: Path,
    timeout_seconds: float,
    memory_limit_mb: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    test_cases = iter_task_test_cases(context.task_name)
    if not test_cases:
        return build_validation_record(
            context=context,
            pass_count=0,
            tests_run=0,
            failed_tests=[],
            error_tests=[
                build_error_case(
                    name=f"{context.task_name}/test_discovery",
                    phase="test_discovery",
                    message=f"No ConDefects test cases were found for task {context.task_name}.",
                )
            ],
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )

    if context.language.lower() == "java":
        failed_tests, error_tests, pass_count = run_java_validation_cases(
            context=context,
            temp_source_path=temp_source_path,
            temp_program_dir=temp_program_dir,
            test_cases=test_cases,
            deadline=deadline,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )
    else:
        failed_tests, error_tests, pass_count = run_python_validation_cases(
            context=context,
            temp_source_path=temp_source_path,
            temp_program_dir=temp_program_dir,
            test_cases=test_cases,
            deadline=deadline,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )

    return build_validation_record(
        context=context,
        pass_count=pass_count,
        tests_run=len(test_cases),
        failed_tests=failed_tests,
        error_tests=error_tests,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )


def run_python_validation_cases(
    context: ProgramContext,
    temp_source_path: Path,
    temp_program_dir: Path,
    test_cases: list[TestCase],
    deadline: float,
    timeout_seconds: float,
    memory_limit_mb: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    source_lines = temp_source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    executable = "python2" if any("print " in line for line in source_lines) else os.environ.get(
        "CONDEFECTS_PYTHON3",
        "python3",
    )
    failed_tests: list[dict[str, Any]] = []
    error_tests: list[dict[str, Any]] = []
    pass_count = 0

    for test_case in test_cases:
        remaining_timeout = remaining_validation_seconds(deadline)
        if remaining_timeout <= 0:
            error_tests.append(build_validation_timeout_case(context, timeout_seconds))
            break

        result = run_guarded_process(
            [executable, str(temp_source_path)],
            cwd=temp_program_dir,
            input_text=test_case.input_text,
            timeout_seconds=remaining_timeout,
            memory_limit_mb=memory_limit_mb,
            language=context.language,
        )
        case_name = f"{context.task_name}/{test_case.name}"
        if result.timed_out:
            error_tests.append(
                build_error_case(
                    name=case_name,
                    phase="timeout",
                    message=(
                        "Python execution timed out before the validation attempt "
                        f"{timeout_seconds:g}-second budget was exhausted."
                    ),
                    result=result,
                )
            )
            break
        if result.memory_exceeded or is_memory_failure(result.returncode, result.stderr):
            error_tests.append(
                build_error_case(
                    name=case_name,
                    phase="memory",
                    message=(
                        "Python execution exceeded the validation memory limit "
                        f"of {memory_limit_mb} MB."
                    ),
                    result=result,
                )
            )
            break
        if result.returncode != 0:
            error_tests.append(
                build_error_case(
                    name=case_name,
                    phase="runtime",
                    message="Python fix code exited with a non-zero status.",
                    result=result,
                )
            )
        elif normalize_output(result.stdout) == normalize_output(test_case.expected_output):
            pass_count += 1
        else:
            failed_tests.append(build_output_failure_case(context, test_case, result.stdout))

    return failed_tests, error_tests, pass_count


def run_java_validation_cases(
    context: ProgramContext,
    temp_source_path: Path,
    temp_program_dir: Path,
    test_cases: list[TestCase],
    deadline: float,
    timeout_seconds: float,
    memory_limit_mb: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    execution_dir = temp_program_dir / "validation_execution"
    if execution_dir.exists():
        shutil.rmtree(execution_dir)
    execution_dir.mkdir(parents=True, exist_ok=True)
    main_java_path = execution_dir / "Main.java"
    shutil.copy2(temp_source_path, main_java_path)

    try:
        remaining_timeout = remaining_validation_seconds(deadline)
        if remaining_timeout <= 0:
            return [], [build_validation_timeout_case(context, timeout_seconds)], 0

        compile_result = run_guarded_process(
            ["javac", "Main.java"],
            cwd=execution_dir,
            input_text="",
            timeout_seconds=remaining_timeout,
            memory_limit_mb=memory_limit_mb,
            language=context.language,
        )
        if compile_result.timed_out:
            return [], [
                build_error_case(
                    name=f"{context.task_name}/compile_timeout",
                    phase="timeout",
                    message=f"Java compilation timed out after {timeout_seconds:g} seconds.",
                    result=compile_result,
                )
            ], 0
        if compile_result.memory_exceeded or is_memory_failure(
            compile_result.returncode,
            compile_result.stderr,
        ):
            return [], [
                build_error_case(
                    name=f"{context.task_name}/compile_memory",
                    phase="memory",
                    message=(
                        "Java compilation exceeded the validation memory limit "
                        f"of {memory_limit_mb} MB."
                    ),
                    result=compile_result,
                )
            ], 0
        if compile_result.returncode != 0:
            return [], [
                build_error_case(
                    name=f"{context.task_name}/compile",
                    phase="compile",
                    message="Java fix code failed to compile.",
                    result=compile_result,
                )
            ], 0

        failed_tests: list[dict[str, Any]] = []
        error_tests: list[dict[str, Any]] = []
        pass_count = 0

        for test_case in test_cases:
            remaining_timeout = remaining_validation_seconds(deadline)
            if remaining_timeout <= 0:
                error_tests.append(build_validation_timeout_case(context, timeout_seconds))
                break

            result = run_guarded_process(
                ["java", "Main"],
                cwd=execution_dir,
                input_text=test_case.input_text,
                timeout_seconds=remaining_timeout,
                memory_limit_mb=memory_limit_mb,
                language=context.language,
            )
            case_name = f"{context.task_name}/{test_case.name}"
            if result.timed_out:
                error_tests.append(
                    build_error_case(
                        name=case_name,
                        phase="timeout",
                        message=(
                            "Java execution timed out before the validation attempt "
                            f"{timeout_seconds:g}-second budget was exhausted."
                        ),
                        result=result,
                    )
                )
                break
            if result.memory_exceeded or is_memory_failure(result.returncode, result.stderr):
                error_tests.append(
                    build_error_case(
                        name=case_name,
                        phase="memory",
                        message=(
                            "Java execution exceeded the validation memory limit "
                            f"of {memory_limit_mb} MB."
                        ),
                        result=result,
                    )
                )
                break
            if result.returncode != 0:
                error_tests.append(
                    build_error_case(
                        name=case_name,
                        phase="runtime",
                        message="Java fix code exited with a non-zero status.",
                        result=result,
                    )
                )
            elif normalize_output(result.stdout) == normalize_output(test_case.expected_output):
                pass_count += 1
            else:
                failed_tests.append(build_output_failure_case(context, test_case, result.stdout))

        return failed_tests, error_tests, pass_count
    finally:
        shutil.rmtree(execution_dir, ignore_errors=True)


def build_validation_record(
    context: ProgramContext,
    pass_count: int,
    tests_run: int,
    failed_tests: list[dict[str, Any]],
    error_tests: list[dict[str, Any]],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> dict[str, Any]:
    pass_rate = pass_count / tests_run if tests_run else None
    timed_out = any(test.get("phase") == "timeout" for test in error_tests)
    memory_exceeded = any(test.get("phase") == "memory" for test in error_tests)
    passed = pass_rate == 1.0 and not error_tests
    failed_test_count = len(failed_tests)
    error_test_count = len(error_tests)
    saved_failed_tests, saved_error_tests = limit_saved_validation_failures(
        failed_tests,
        error_tests,
    )
    if passed:
        status = "pass"
    elif timed_out:
        status = "timeout"
    elif memory_exceeded:
        status = "memory_exceeded"
    else:
        status = "fail"
    return {
        "task": context.task_name,
        "language": context.language,
        "program_id": context.program_id,
        "status": status,
        "pass": passed,
        "pass_rate": pass_rate,
        "tests_run": tests_run,
        "passed_tests": pass_count,
        "failed_tests": saved_failed_tests,
        "error_tests": saved_error_tests,
        "failed_test_count": failed_test_count,
        "error_test_count": error_test_count,
        "saved_failed_test_count": len(saved_failed_tests),
        "saved_error_test_count": len(saved_error_tests),
        "max_saved_failure_cases": MAX_SAVED_VALIDATION_FAILURE_CASES,
        "timed_out": timed_out,
        "memory_exceeded": memory_exceeded,
        "timeout_seconds": timeout_seconds,
        "memory_limit_mb": memory_limit_mb,
    }


def limit_saved_validation_failures(
    failed_tests: list[dict[str, Any]],
    error_tests: list[dict[str, Any]],
    max_cases: int = MAX_SAVED_VALIDATION_FAILURE_CASES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    saved_error_tests = error_tests[:max_cases]
    remaining = max(0, max_cases - len(saved_error_tests))
    saved_failed_tests = failed_tests[:remaining]
    return saved_failed_tests, saved_error_tests


def validation_message(record: dict[str, Any]) -> str:
    if record.get("pass") is True:
        return "the patch successfully resolved the issue"
    if record.get("timed_out") is True:
        return (
            "the patch could not solve the issue because validation timed out. "
            f"timeout_seconds={record.get('timeout_seconds')}"
        )
    if record.get("memory_exceeded") is True:
        return (
            "the patch could not solve the issue because validation exceeded the "
            f"memory limit. memory_limit_mb={record.get('memory_limit_mb')}"
        )
    return f"the patch could not solve the issue. pass_rate={record.get('pass_rate')}"


def write_validation_artifacts(result_dir: Path, record: dict[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json_file(result_dir / "validation_record.json", record)
    summary_lines = [
        f"task = {record.get('task')}",
        f"language = {record.get('language')}",
        f"program_id = {record.get('program_id')}",
        f"status = {record.get('status')}",
        f"pass = {record.get('pass')}",
        f"pass_rate = {record.get('pass_rate')}",
        f"tests_run = {record.get('tests_run')}",
        f"passed_tests = {record.get('passed_tests')}",
        f"failed_test_count = {record.get('failed_test_count')}",
        f"error_test_count = {record.get('error_test_count')}",
        f"saved_failed_test_count = {record.get('saved_failed_test_count')}",
        f"saved_error_test_count = {record.get('saved_error_test_count')}",
        f"max_saved_failure_cases = {record.get('max_saved_failure_cases')}",
        f"timed_out = {record.get('timed_out')}",
        f"memory_exceeded = {record.get('memory_exceeded')}",
        f"timeout_seconds = {record.get('timeout_seconds')}",
        f"memory_limit_mb = {record.get('memory_limit_mb')}",
    ]
    (result_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if record.get("failed_tests"):
        write_json_file(result_dir / "failed_tests.json", {"tests": record["failed_tests"]})
    if record.get("error_tests"):
        write_json_file(result_dir / "error_tests.json", {"tests": record["error_tests"]})


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_output_failure_case(
    context: ProgramContext,
    test_case: TestCase,
    actual_output: str,
) -> dict[str, Any]:
    return {
        "name": f"{context.task_name}/{test_case.name}",
        "status": "failed",
        "phase": "output",
        "test_case": test_case.name,
        "message": (
            "ConDefects test case output did not match the expected output."
            f"\n\nInput:\n{truncate_text(test_case.input_text)}"
            f"\n\nExpected output:\n{truncate_text(test_case.expected_output)}"
            f"\n\nActual output:\n{truncate_text(actual_output)}"
        ),
        "input": truncate_text(test_case.input_text),
        "expected_output": truncate_text(test_case.expected_output),
        "actual_output": truncate_text(actual_output),
    }


def build_validation_timeout_case(context: ProgramContext, timeout_seconds: float) -> dict[str, Any]:
    return build_error_case(
        name=f"{context.task_name}/validation_timeout",
        phase="timeout",
        message=(
            "Validation timed out after "
            f"{timeout_seconds:g} seconds before all test cases finished."
        ),
    )


def build_error_case(
    name: str,
    phase: str,
    message: str,
    result: GuardedProcessResult | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "status": "error",
        "phase": phase,
        "message": message,
    }
    if result is not None:
        payload.update(
            {
                "returncode": result.returncode,
                "stdout": truncate_text(result.stdout),
                "stderr": truncate_text(result.stderr),
                "timed_out": result.timed_out,
                "memory_exceeded": result.memory_exceeded,
                "peak_rss_mb": result.peak_rss_mb,
            }
        )
    return payload


def run_guarded_process(
    command: list[str],
    cwd: Path,
    input_text: str,
    timeout_seconds: float,
    memory_limit_mb: int,
    language: str,
) -> GuardedProcessResult:
    env = os.environ.copy()
    if language.lower() == "java" and memory_limit_mb > 0:
        configure_java_memory(env, memory_limit_mb)

    preexec_fn = None
    if language.lower() != "java" and memory_limit_mb > 0:
        preexec_fn = python_memory_limiter(memory_limit_mb)

    with tempfile.TemporaryFile("w+", encoding="utf-8") as stdin_file, tempfile.TemporaryFile(
        "w+",
        encoding="utf-8",
    ) as stdout_file, tempfile.TemporaryFile("w+", encoding="utf-8") as stderr_file:
        stdin_file.write(input_text)
        stdin_file.seek(0)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=stdin_file,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env=env,
            preexec_fn=preexec_fn,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        memory_limit_bytes = memory_limit_mb * 1024 * 1024 if memory_limit_mb > 0 else None
        timed_out = False
        memory_exceeded = False
        peak_rss_bytes = 0

        while True:
            returncode = process.poll()
            peak_rss_bytes = max(peak_rss_bytes, process_tree_rss(process.pid))
            if returncode is not None:
                break

            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                terminate_process_group(process)
                break
            if memory_limit_bytes is not None and peak_rss_bytes > memory_limit_bytes:
                memory_exceeded = True
                terminate_process_group(process)
                break

            time.sleep(min(MONITOR_INTERVAL_SECONDS, max(0.01, deadline - now)))

        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=3)
        if process.poll() is None:
            terminate_process_group(process)
        returncode = process.returncode if process.returncode is not None else -9
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read() or ""
        stderr = stderr_file.read() or ""
        memory_exceeded = memory_exceeded or (not timed_out and is_memory_failure(returncode, stderr))
        if timed_out:
            stderr = append_process_note(stderr, "Process timed out during validation.")
        if memory_exceeded:
            stderr = append_process_note(stderr, "Process exceeded validation memory limit.")
        return GuardedProcessResult(
            args=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            memory_exceeded=memory_exceeded,
            peak_rss_mb=round(peak_rss_bytes / 1024 / 1024, 3),
        )


def append_process_note(stderr: str, note: str) -> str:
    return f"{stderr.rstrip()}\n{note}\n" if stderr else f"{note}\n"


def remaining_validation_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def python_memory_limiter(memory_limit_mb: int):
    def limit_memory() -> None:
        try:
            import resource
        except ImportError:
            return

        memory_bytes = int(memory_limit_mb * 1024 * 1024)
        for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
            limit = getattr(resource, limit_name, None)
            if limit is None:
                continue
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(limit, (memory_bytes, memory_bytes))

    return limit_memory


def configure_java_memory(env: dict[str, str], memory_limit_mb: int) -> None:
    java_heap_mb = max(64, int(memory_limit_mb * 0.65))
    defaults = [
        f"-Xmx{java_heap_mb}m",
        "-XX:MaxMetaspaceSize=128m",
        "-XX:CompressedClassSpaceSize=64m",
        "-XX:ReservedCodeCacheSize=64m",
        "-Xss256k",
    ]
    current = env.get("JAVA_TOOL_OPTIONS", "")
    missing = [option for option in defaults if java_option_key(option) not in current]
    if missing:
        env["JAVA_TOOL_OPTIONS"] = f"{' '.join(missing)} {current}".strip()


def java_option_key(option: str) -> str:
    if option.startswith("-Xmx"):
        return "-Xmx"
    if option.startswith("-Xss"):
        return "-Xss"
    return option.split("=", 1)[0]


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=3)
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=3)


def process_tree_rss(pid: int) -> int:
    seen: set[int] = set()
    pending = [pid]
    total = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        total += proc_rss(current)
        pending.extend(child for child in proc_children(current) if child not in seen)
    return total


def proc_rss(pid: int) -> int:
    try:
        statm = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def proc_children(pid: int) -> list[int]:
    children: set[int] = set()
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tasks = list(task_dir.iterdir())
    except OSError:
        return []
    for task in tasks:
        try:
            raw = (task / "children").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        for value in raw.split():
            with contextlib.suppress(ValueError):
                children.add(int(value))
    return list(children)


def is_memory_failure(returncode: int | None, stderr: str) -> bool:
    if returncode in {-9, -11, 137}:
        return True
    lowered = (stderr or "").lower()
    return any(marker in lowered for marker in MEMORY_ERROR_MARKERS)


def is_memory_exception(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    return any(marker in str(exc).lower() for marker in MEMORY_ERROR_MARKERS)


def truncate_text(text: str, limit: int = MAX_FAILURE_TEXT_LENGTH) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half].strip()}\n...[truncated]...\n{text[-half:].strip()}"


def find_reproducing_test_case(
    context: ProgramContext,
    location: FaultyFunctionLocation,
    fix_code: str | None = None,
    test_root: str | Path = DEFAULT_TEST_ROOT,
    max_cases: int | None = None,
) -> tuple[TestCase, ReproResult] | None:
    for index, test_case in enumerate(iter_task_test_cases(context.task_name, test_root)):
        if max_cases is not None and index >= max_cases:
            break
        result = execute_test_case(context, location, test_case, fix_code=fix_code)
        if result.reproduced:
            return test_case, result
    return None


def execute_test_case(
    context: ProgramContext,
    location: FaultyFunctionLocation,
    test_case: TestCase,
    fix_code: str | None = None,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> ReproResult:
    temp_root = Path(tempfile.mkdtemp(prefix=f"specrover_repro_{context.program_id}_"))
    try:
        _, temp_program_dir = copy_program_to_temp_dataset(context, dataset_root, temp_root)
        temp_source_path = temp_program_dir / location.source_path.name
        if fix_code is not None:
            replace_source_range(
                temp_source_path,
                location.start_line_index,
                location.end_line_index,
                fix_code,
            )

        if context.language.lower() == "java":
            cp = run_java_case(temp_source_path, test_case.input_path, temp_program_dir)
        else:
            cp = run_python_case(temp_source_path, test_case.input_path, temp_program_dir)

        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
        output_matches = normalize_output(stdout) == normalize_output(test_case.expected_output)
        passed = cp.returncode == 0 and output_matches
        if not passed and not stderr:
            stderr = (
                "Output mismatch.\n\n"
                f"Expected:\n{test_case.expected_output}\n\n"
                f"Actual:\n{stdout}"
            )
        return ReproResult(
            stdout=stdout,
            stderr=stderr,
            returncode=cp.returncode,
            reproduced=not passed,
            passed=passed,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def iter_task_test_cases(
    task_name: str,
    test_root: str | Path = DEFAULT_TEST_ROOT,
) -> list[TestCase]:
    test_dir = test_dir_for_task(task_name, test_root)
    if test_dir is None:
        return []
    input_dir = test_dir / "in"
    output_dir = test_dir / "out"
    cases: list[TestCase] = []
    for input_path in sorted(input_dir.iterdir()):
        if not input_path.is_file():
            continue
        expected_output_path = output_dir / input_path.name
        if not expected_output_path.exists():
            continue
        cases.append(
            TestCase(
                name=input_path.name,
                input_path=input_path,
                expected_output_path=expected_output_path,
                input_text=input_path.read_text(encoding="utf-8", errors="ignore"),
                expected_output=expected_output_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ),
            )
        )
    return cases


def test_dir_for_task(task_name: str, test_root: str | Path) -> Path | None:
    if "_" not in task_name:
        candidate = Path(test_root) / task_name
        return candidate if candidate.is_dir() else None
    contest, problem = task_name.rsplit("_", 1)
    contest_path = resolve_case_insensitive_child(Path(test_root), contest)
    if not contest_path.is_dir():
        return None
    for candidate_name in (problem.upper(), problem, "Ex"):
        candidate = resolve_case_insensitive_child(contest_path, candidate_name)
        if (candidate / "in").is_dir() and (candidate / "out").is_dir():
            return candidate
    return None


def copy_program_to_temp_dataset(
    context: ProgramContext,
    dataset_root: str | Path,
    temp_root: Path,
) -> tuple[Path, Path]:
    temp_dataset_root = temp_root / "dataset"
    temp_program_dir = (
        temp_dataset_root / context.task_name / context.language / context.program_id
    )
    temp_program_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(context.source_path, temp_program_dir / context.source_path.name)
    for metadata_name in FAULT_LOCATION_METADATA_FILENAMES:
        metadata_path = context.program_dir / metadata_name
        if metadata_path.exists() and metadata_path.is_file():
            shutil.copy2(metadata_path, temp_program_dir / metadata_name)

    leaked_correct_versions = sorted(temp_program_dir.glob("correctVersion.*"))
    if leaked_correct_versions:
        raise RuntimeError(
            "Temporary ConDefects program directory unexpectedly contains "
            f"correctVersion files: {leaked_correct_versions}"
        )
    return temp_dataset_root, temp_program_dir


FAULT_LOCATION_METADATA_FILENAMES = (OCHIAI_FL_FILENAME,)


def replace_source_range(
    source_path: str | Path,
    start_line_index: int,
    end_line_index: int,
    replacement: str,
) -> None:
    source_path = Path(source_path)
    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )
    if not 0 <= start_line_index <= end_line_index < len(lines):
        raise ValueError("Faulty function range is outside source file")

    suffix = "".join(lines[end_line_index + 1 :])
    replacement_text = replacement
    if suffix and replacement_text and not replacement_text.endswith(("\n", "\r")):
        replacement_text += infer_line_ending(lines[end_line_index])

    patched = "".join(lines[:start_line_index]) + replacement_text + suffix
    source_path.write_text(patched, encoding="utf-8")


def run_python_case(
    source_path: Path,
    input_path: Path,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    input_text = input_path.read_text(encoding="utf-8", errors="ignore")
    result = run_guarded_process(
        [os.environ.get("CONDEFECTS_PYTHON3", "python3"), str(source_path)],
        cwd=cwd,
        input_text=input_text,
        timeout_seconds=30,
        memory_limit_mb=DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
        language="Python",
    )
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run_java_case(
    source_path: Path,
    input_path: Path,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    main_path = cwd / "Main.java"
    if source_path != main_path:
        shutil.copy2(source_path, main_path)
    compile_result = run_guarded_process(
        ["javac", str(main_path)],
        cwd=cwd,
        input_text="",
        timeout_seconds=30,
        memory_limit_mb=DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
        language="Java",
    )
    if compile_result.returncode != 0 or compile_result.timed_out or compile_result.memory_exceeded:
        return subprocess.CompletedProcess(
            args=compile_result.args,
            returncode=compile_result.returncode,
            stdout=compile_result.stdout,
            stderr=compile_result.stderr,
        )
    input_text = input_path.read_text(encoding="utf-8", errors="ignore")
    run_result = run_guarded_process(
        ["java", "Main"],
        cwd=cwd,
        input_text=input_text,
        timeout_seconds=30,
        memory_limit_mb=DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
        language="Java",
    )
    return subprocess.CompletedProcess(
        args=run_result.args,
        returncode=run_result.returncode,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
    )


def normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def infer_line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    return "\n"


def resolve_case_insensitive_child(parent: Path, child: str) -> Path:
    if not parent.is_dir():
        return parent / child
    for entry in parent.iterdir():
        if entry.name.lower() == child.lower():
            return entry
    return parent / child
