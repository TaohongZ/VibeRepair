from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

from .agent_write_patch import InvalidLLMResponse
from .data_structures import MessageThread, ReproResult
from .model import ChatModel
from .prompts import REVIEWER_SYSTEM_PROMPT


class ReviewDecision(Enum):
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class Review:
    patch_decision: ReviewDecision
    patch_analysis: str
    patch_advice: str
    test_decision: ReviewDecision
    test_analysis: str
    test_advice: str

    def to_json(self) -> dict:
        return {
            "patch-correct": self.patch_decision.value,
            "patch-analysis": self.patch_analysis,
            "patch-advice": self.patch_advice,
            "test-correct": self.test_decision.value,
            "test-analysis": self.test_analysis,
            "test-advice": self.test_advice,
        }

    def __str__(self) -> str:
        return (
            f"Patch decision: {self.patch_decision.value}\n\n"
            f"Patch analysis: {self.patch_analysis}\n\n"
            f"Patch advice: {self.patch_advice}\n\n"
            f"Test decision: {self.test_decision.value}\n\n"
            f"Test analysis: {self.test_analysis}\n\n"
            f"Test advice: {self.test_advice}"
        )


def run(
    model: ChatModel,
    issue_statement: str,
    test_content: str,
    patch_content: str,
    orig_repro: ReproResult,
    patched_repro: ReproResult,
    retries: int = 5,
) -> tuple[Review, MessageThread]:
    thread = construct_review_thread(
        issue_statement,
        test_content,
        patch_content,
        orig_repro,
        patched_repro,
    )
    for _ in range(retries):
        chat_result = model.complete(thread.to_msg())
        response = chat_result.content
        review_thread = deepcopy(thread)
        review_thread.add_model(response)
        review = extract_review_result(response)
        if review is not None:
            return review, review_thread
    raise InvalidLLMResponse(f"failed to review in {retries} attempts")


def construct_review_thread(
    issue_statement: str,
    test: str,
    patch: str,
    orig_repro: ReproResult,
    patched_repro: ReproResult,
) -> MessageThread:
    thread = MessageThread()
    thread.add_system(REVIEWER_SYSTEM_PROMPT)
    thread.add_user(f"Here is the issue: <issue>{issue_statement}</issue>.\n")
    thread.add_user(f"Here is the test written by Engineer A: <test>{test}</test>.\n")
    thread.add_user(
        "Here is the result of executing the test on the original buggy program:\n"
        f"stdout:\n\n{orig_repro.stdout}\n\n"
        f"stderr:\n\n{orig_repro.stderr}\n"
    )
    thread.add_user(f"Here is the patch written by Engineer B: <patch>{patch}</patch>.\n")
    thread.add_user(
        "Here is the result of executing the test on the patched program:\n"
        f"stdout:\n\n{patched_repro.stdout}\n\n"
        f"stderr:\n\n{patched_repro.stderr}."
    )
    thread.add_user(
        "Think about (1) whether the test correctly reproduces the issue, and "
        "(2) whether the patch resolves the issue. "
        "Provide your answer in the following json schema:\n"
        "\n"
        "```json\n"
        "{\n"
        '    "patch-correct": "...",\n'
        '    "test-correct": "...",\n'
        '    "patch-analysis": "...",\n'
        '    "test-analysis": "...",\n'
        '    "patch-advice": "...",\n'
        '    "test-advice": "..."\n'
        "}\n"
        "```\n"
        "\n"
        'where "patch-correct"/"test-correct" is "yes" or "no"; '
        '"patch-analysis"/"test-analysis" should explain the reasoning behind your answer.\n'
        'Moreover, if your answer is "no", then give me advice about how to correct'
        ' the patch/test in the "patch-advice"/"test-advice" field.\n'
        'If your answer is "yes", you can leave "patch-advice"/"test-advice"'
        "empty.\n"
        "\n"
        "NOTE: not only the patch, but also the test case, can be wrong."
    )
    return thread


def extract_review_result(content: str) -> Review | None:
    data = extract_json_object(content)
    if not isinstance(data, dict):
        return None
    try:
        review = Review(
            patch_decision=ReviewDecision(str(data["patch-correct"]).lower()),
            patch_analysis=str(data["patch-analysis"]),
            patch_advice=str(data.get("patch-advice") or ""),
            test_decision=ReviewDecision(str(data["test-correct"]).lower()),
            test_analysis=str(data["test-analysis"]),
            test_advice=str(data.get("test-advice") or ""),
        )
        if (
            (review.patch_decision == ReviewDecision.NO and not review.patch_advice)
            and (review.test_decision == ReviewDecision.NO and not review.test_advice)
        ):
            return None
        return review
    except Exception:
        return None


def extract_json_object(content: str) -> object | None:
    candidates = [content.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    )

    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        while start != -1:
            try:
                data, _ = decoder.raw_decode(candidate[start:])
                return data
            except json.JSONDecodeError:
                start = candidate.find("{", start + 1)
    return None

