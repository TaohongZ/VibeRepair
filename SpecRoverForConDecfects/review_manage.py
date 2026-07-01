from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

from . import agent_reviewer
from .agent_reproducer import NoReproductionStep, TestAgent, TestHandle
from .agent_reviewer import Review, ReviewDecision
from .agent_write_patch import InvalidLLMResponse, PatchAgent, PatchHandle
from .condefects_dataset import FaultyFunctionLocation, ProgramContext
from .data_structures import ReproResult
from .model import ChatModel


class ReviewManager:
    def __init__(
        self,
        context: ProgramContext,
        location: FaultyFunctionLocation,
        issue_statement: str,
        patch_agent: PatchAgent,
        test_agent: TestAgent,
        model: ChatModel,
        output_dir: str | Path,
        repro_result_map: dict[tuple[PatchHandle, TestHandle], ReproResult] | None = None,
    ) -> None:
        self.context = context
        self.location = location
        self.issue_statement = issue_statement
        self.patch_agent = patch_agent
        self.test_agent = test_agent
        self.model = model
        self.output_dir = Path(output_dir)
        self.repro_result_map = dict(repro_result_map or {})

    def patch_only_generator(
        self,
        retries: int = 3,
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        try:
            while True:
                patch_handle, patch_content = (
                    self.patch_agent.write_applicable_patch_without_feedback(
                        retries=retries
                    )
                )
                self.save_patch(patch_handle, patch_content)
                evaluation_msg = yield patch_handle, patch_content
                if evaluation_msg:
                    self.patch_agent.add_feedback(patch_handle, evaluation_msg)
        except InvalidLLMResponse:
            return

    def generator(
        self,
        rounds: int = 5,
        patch_retries: int = 3,
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        try:
            yield from self._generator(rounds=rounds, patch_retries=patch_retries)
        except (InvalidLLMResponse, NoReproductionStep):
            return

    def _generator(
        self,
        rounds: int,
        patch_retries: int,
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        if not self.test_agent._history:
            test_handle, test_content, orig_repro_result = (
                self.test_agent.write_reproducing_test_without_feedback()
            )
        else:
            test_handle = self.test_agent._history[-1]
            test_content = self.test_agent._responses[test_handle]
            orig_repro_result = self.repro_result_map[
                (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
            ]

        coords = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
        self.repro_result_map[coords] = orig_repro_result
        self.save_execution_result(orig_repro_result, *coords)

        patch_handle, patch_content = (
            self.patch_agent.write_applicable_patch_without_feedback(
                test_content=test_content,
                retries=patch_retries,
            )
        )
        self.save_patch(patch_handle, patch_content)

        for _ in range(rounds):
            patched_repro_result = self.test_agent.execute(test_handle, patch_content)
            coords = (patch_handle, test_handle)
            self.repro_result_map[coords] = patched_repro_result
            self.save_execution_result(patched_repro_result, *coords)

            review, review_thread = agent_reviewer.run(
                self.model,
                self.issue_statement,
                test_content,
                patch_content,
                orig_repro_result,
                patched_repro_result,
            )
            self.save_review(patch_handle, test_handle, review)
            review_thread.save_to_file(
                self.output_dir / f"conv_review_{patch_handle}_{test_handle}.json"
            )

            if review.patch_decision == ReviewDecision.YES:
                evaluation_msg = yield patch_handle, patch_content
                if evaluation_msg:
                    self.patch_agent.add_feedback(patch_handle, evaluation_msg)
                    patch_handle, patch_content = (
                        self.patch_agent.write_applicable_patch_with_feedback(
                            test_content=test_content,
                            retries=patch_retries,
                        )
                    )
                    self.save_patch(patch_handle, patch_content)
                    continue

            if review.patch_decision == ReviewDecision.NO:
                feedback = self.compose_feedback_for_patch_generation(review, test_content)
                self.patch_agent.add_feedback(patch_handle, feedback)
                patch_handle, patch_content = (
                    self.patch_agent.write_applicable_patch_with_feedback(
                        test_content=test_content,
                        retries=patch_retries,
                    )
                )
                self.save_patch(patch_handle, patch_content)

            if review.test_decision == ReviewDecision.NO:
                self.test_agent.add_feedback(
                    test_handle,
                    self.compose_feedback_for_test_generation(review, patch_content),
                )

    @classmethod
    def compose_feedback_for_patch_generation(cls, review: Review, test: str) -> str:
        return (
            f"The previous patch failed a test written by another developer.\n"
            f"Rethink about the code context, reflect, and write another patch.\n"
            f"You can also write the new patch at other locations.\n"
            f"Here is the test file:\n"
            "```\n"
            f"{test}"
            "```\n"
            f"By executing the test file with and without the patch,"
            " the following analysis can be made:\n"
            "\n"
            f"{review.patch_analysis}\n"
            "\n"
            "Therefore, the patch does not correctly resovle the issue.\n"
            "\n"
            "To correct the patch, here is the advice given by another engineer:\n"
            "\n"
            f"{review.patch_advice}"
        )

    @classmethod
    def compose_feedback_for_test_generation(cls, review: Review, patch: str) -> str:
        return (
            f"Here is a patch to the program:\n"
            "```\n"
            f"{patch}"
            "```\n"
            f"By executing your test with and without the patch,"
            " the following analysis can be made:\n"
            "\n"
            f"{review.test_analysis}"
            "\n"
            "Therefore, the test does not correctly reproduce the issue.\n"
            "\n"
            "To correct the test, here is my advice:\n"
            "\n"
            f"{review.test_advice}"
        )

    def save_patch(self, handle: PatchHandle, content: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath(f"extracted_patch_{handle}.txt").write_text(
            content,
            encoding="utf-8",
        )

    def save_review(
        self,
        patch_handle: PatchHandle,
        test_handle: TestHandle,
        review: Review,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath(f"review_p{patch_handle}_t{test_handle}.json").write_text(
            json.dumps(review.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def save_execution_result(
        self,
        result: ReproResult,
        patch_handle: PatchHandle,
        test_handle: TestHandle,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath(f"execution_{patch_handle}_{test_handle}.json").write_text(
            json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

