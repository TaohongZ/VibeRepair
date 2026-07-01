from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

from .condefects_dataset import FaultyFunctionLocation, ProgramContext
from .data_structures import MessageThread
from .model import ChatModel
from .patch_utils import PatchExtractionResult, extract_fixed_function_from_response
from .prompts import PATCH_SYSTEM_PROMPT, PATCH_USER_PROMPT_INIT


PatchHandle: TypeAlias = str


class InvalidLLMResponse(RuntimeError):
    """Raised when the model cannot produce an applicable tagged edit."""


class PatchAgent:
    EMPTY_PATCH_HANDLE = "EMPTY"

    def __init__(
        self,
        context: ProgramContext,
        location: FaultyFunctionLocation,
        task_information: str,
        model: ChatModel,
        task_dir: str | Path,
    ) -> None:
        self.context = context
        self.location = location
        self.task_information = task_information
        self.model = model
        self.task_dir = Path(task_dir)

        self._request_idx = -1
        self._responses: dict[PatchHandle, str] = {}
        self._fix_codes: dict[PatchHandle, str] = {}
        self._feedbacks: dict[PatchHandle, list[str]] = defaultdict(list)
        self._history: list[PatchHandle] = []
        self._extractions: dict[PatchHandle, PatchExtractionResult] = {}
        self.usage_records: list[dict] = []

    def write_applicable_patch_without_feedback(
        self,
        test_content: str | None = None,
        retries: int = 3,
    ) -> tuple[PatchHandle, str]:
        return self._write_applicable_patch(
            max_feedbacks=0,
            retries=retries,
            test_content=test_content,
        )

    def write_applicable_patch_with_feedback(
        self,
        test_content: str | None = None,
        max_feedbacks: int = 1,
        retries: int = 3,
    ) -> tuple[PatchHandle, str]:
        return self._write_applicable_patch(
            max_feedbacks=max_feedbacks,
            retries=retries,
            test_content=test_content,
        )

    def add_feedback(self, handle: PatchHandle, feedback: str) -> None:
        if handle not in self._fix_codes:
            raise ValueError(f"patch {handle} does not exist")
        self._feedbacks[handle].append(feedback)

    def _write_applicable_patch(
        self,
        max_feedbacks: int,
        retries: int,
        test_content: str | None = None,
    ) -> tuple[PatchHandle, str]:
        max_feedbacks = max_feedbacks if max_feedbacks >= 0 else len(self._history)
        history_handles = self._history[-min(max_feedbacks, len(self._history)) :]

        last_summary = ""
        for _ in range(retries):
            response, extraction, thread = self._write_patch(
                history_handles,
                test_content=test_content,
                previous_extract_summary=last_summary,
            )
            self._request_idx += 1
            handle = str(self._request_idx)
            self.task_dir.mkdir(parents=True, exist_ok=True)
            self.task_dir.joinpath(f"patch_raw_{handle}.md").write_text(
                response,
                encoding="utf-8",
            )
            thread.save_to_file(self.task_dir / f"conv_patch_{handle}.json")
            self.task_dir.joinpath(f"extract_status_{handle}.json").write_text(
                json.dumps(extraction_to_json(extraction), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )

            last_summary = extraction.summary
            if extraction.applicable:
                self._responses[handle] = response
                self._fix_codes[handle] = extraction.fixed_code
                self._extractions[handle] = extraction
                self._history.append(handle)
                self.task_dir.joinpath(f"fix_code_{handle}.txt").write_text(
                    extraction.fixed_code,
                    encoding="utf-8",
                )
                return handle, extraction.fixed_code

        raise InvalidLLMResponse(f"Failed to write an applicable patch in {retries} attempts")

    def _write_patch(
        self,
        history_handles: list[PatchHandle],
        test_content: str | None,
        previous_extract_summary: str,
    ) -> tuple[str, PatchExtractionResult, MessageThread]:
        thread = self._construct_init_thread()

        for handle in history_handles:
            feedbacks = self._feedbacks.get(handle, [])
            if not feedbacks:
                continue
            thread.add_model(self._responses[handle])
            for feedback in feedbacks:
                thread.add_user(feedback)

        if test_content is not None:
            thread.add_user(
                "A ConDefects test case reproduces the issue. You can use it as additional context:\n"
                + test_content
            )

        thread.add_user(PATCH_USER_PROMPT_INIT)
        if previous_extract_summary:
            thread.add_user(
                "Your previous edit could not be applied to the program. "
                + previous_extract_summary
                + " Please try again."
            )

        chat_result = self.model.complete(thread.to_msg())
        response = chat_result.content
        self.usage_records.append(chat_result.usage.to_dict())
        thread.add_model(response)

        extraction = extract_fixed_function_from_response(
            response=response,
            faulty_function=self.location.code,
            source_file_name=self.context.file_name,
            language=self.context.language,
        )
        return response, extraction, thread

    def _construct_init_thread(self) -> MessageThread:
        thread = MessageThread()
        thread.add_system(PATCH_SYSTEM_PROMPT)
        thread.add_user(f"Here is the issue:\n{self.task_information}")
        thread.add_user(self._construct_code_context_prompt())
        return thread

    def _construct_code_context_prompt(self) -> str:
        prompt = "Here are the possible buggy locations collected by someone else. "
        prompt += (
            "Each location contains the actual code snippet and the intended behavior of "
            "the code for resolving the issue.\n"
        )
        prompt += (
            "Location #1:\n"
            f"<file>{self.context.file_name}</file>\n"
            "<class></class>\n"
            "<method></method>\n"
            f"<code>\n{self.location.code}\n</code>"
            "<intended_behavior>This code should implement the task described by the Problem Statement and Constraints while preserving the existing input/output contract.</intended_behavior>\n\n"
        )
        prompt += (
            "Note that you DO NOT NEED to modify every location; you should think what changes "
            "are necessary for resolving the issue, and only propose those modifications."
        )
        return prompt


def extraction_to_json(extraction: PatchExtractionResult) -> dict:
    return {
        "status": extraction.status.value,
        "summary": extraction.summary,
        "fixed_code": extraction.fixed_code,
        "applied_edit_count": extraction.applied_edit_count,
        "edits": [
            {
                "filename": edit.filename,
                "before": edit.before,
                "after": edit.after,
            }
            for edit in extraction.edits
        ],
    }

