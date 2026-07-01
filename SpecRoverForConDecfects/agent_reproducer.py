from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

from .condefects_dataset import FaultyFunctionLocation, ProgramContext
from .data_structures import ReproResult
from .validation import TestCase, execute_test_case, find_reproducing_test_case


TestHandle: TypeAlias = str


class NoReproductionStep(RuntimeError):
    """Raised when no ConDefects test case reproduces the issue."""


class TestAgent:
    def __init__(
        self,
        context: ProgramContext,
        location: FaultyFunctionLocation,
        task_dir: str | Path,
        max_reproducer_tests: int | None = None,
    ) -> None:
        self.context = context
        self.location = location
        self.task_dir = Path(task_dir)
        self.max_reproducer_tests = max_reproducer_tests

        self._request_idx = -1
        self._tests: dict[TestHandle, TestCase] = {}
        self._responses: dict[TestHandle, str] = {}
        self._feedbacks: dict[TestHandle, list[str]] = defaultdict(list)
        self._history: list[TestHandle] = []

    def write_reproducing_test_without_feedback(
        self,
        retries: int = 1,
    ) -> tuple[TestHandle, str, ReproResult]:
        return self._write_reproducing_test(retries=retries)

    def write_reproducing_test_with_feedback(
        self,
        max_feedbacks: int = 1,
        retries: int = 1,
    ) -> tuple[TestHandle, str, ReproResult]:
        return self._write_reproducing_test(retries=retries)

    def _write_reproducing_test(
        self,
        retries: int,
    ) -> tuple[TestHandle, str, ReproResult]:
        for _ in range(retries):
            found = find_reproducing_test_case(
                self.context,
                self.location,
                max_cases=self.max_reproducer_tests,
            )
            if found is None:
                continue
            test_case, repro_result = found
            self._request_idx += 1
            handle = str(self._request_idx)
            content = test_case.to_prompt(actual_output=repro_result.stdout)
            self._tests[handle] = test_case
            self._responses[handle] = content
            self._history.append(handle)
            self.save_test(handle)
            self.save_execution_result(handle, repro_result)
            return handle, content, repro_result

        raise NoReproductionStep("No ConDefects test case reproduces the issue.")

    def add_feedback(self, handle: TestHandle, feedback: str) -> None:
        if handle not in self._tests:
            raise ValueError(f"test {handle} does not exist")
        self._feedbacks[handle].append(feedback)

    def execute(self, handle: TestHandle, fix_code: str | None = None) -> ReproResult:
        if handle not in self._tests:
            raise ValueError(f"test {handle} does not exist")
        return execute_test_case(
            self.context,
            self.location,
            self._tests[handle],
            fix_code=fix_code,
        )

    def save_test(self, handle: TestHandle) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.joinpath(f"reproducer_{handle}.txt").write_text(
            self._responses[handle],
            encoding="utf-8",
        )

    def save_execution_result(self, handle: TestHandle, result: ReproResult) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.joinpath(f"execution_EMPTY_{handle}.json").write_text(
            json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

