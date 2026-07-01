from __future__ import annotations

import json
from pathlib import Path
from pprint import pformat
from typing import Any


class MessageThread:
    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages: list[dict[str, Any]] = messages or []

    def add_system(self, message: str) -> None:
        self.messages.append({"role": "system", "content": message})

    def add_user(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def add_model(self, message: str) -> None:
        self.messages.append({"role": "assistant", "content": message})

    def to_msg(self) -> list[dict[str, Any]]:
        return self.messages

    def save_to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.messages, ensure_ascii=False, indent=2) + "\n")

    def __str__(self) -> str:
        return pformat(self.messages, width=160, sort_dicts=False)


class ReproResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        reproduced: bool | None = None,
        passed: bool | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.passed = returncode == 0 if passed is None else passed
        self.reproduced = (not self.passed) if reproduced is None else reproduced

    def to_json(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "passed": self.passed,
            "triggered": self.reproduced,
        }

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Reproduced: {self.reproduced}",
                f"Passed: {self.passed}",
                f"Return code: {self.returncode}",
                "",
                "Stdout:",
                self.stdout,
                "",
                "Stderr:",
                self.stderr,
            ]
        )

