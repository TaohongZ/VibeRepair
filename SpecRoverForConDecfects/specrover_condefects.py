from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ConDefects.util.get_task_sections import get_task_sections

from .agent_reproducer import NoReproductionStep, TestAgent
from .agent_write_patch import PatchAgent
from .condefects_dataset import FaultyFunctionLocation, ProgramContext, locate_ochiai_faulty_function
from .model import ChatModel
from .prompts import PATCH_SYSTEM_PROMPT, PATCH_USER_PROMPT_INIT
from .review_manage import ReviewManager
from .validation import (
    DEFAULT_VALIDATION_MEMORY_LIMIT_MB,
    DEFAULT_VALIDATION_TIMEOUT_SECONDS,
    ValidationResult,
    evaluate_patch,
)


@dataclass(frozen=True)
class RepairRunOptions:
    model_name: str
    temperature: float
    max_patch_retries: int
    dry_run: bool = False
    reproduce_and_review: bool = True
    enable_validation: bool = False
    review_rounds: int = 5
    max_reproducer_tests: int | None = None
    validation_timeout_seconds: float = DEFAULT_VALIDATION_TIMEOUT_SECONDS
    validation_memory_limit_mb: int = DEFAULT_VALIDATION_MEMORY_LIMIT_MB


def run_one_repair(
    context: ProgramContext,
    options: RepairRunOptions,
    work_dir: str | Path,
    task_information_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    location = locate_ochiai_faulty_function(context)
    faulty_function = location.code
    task_information = get_cached_task_information(context, task_information_cache)

    (work_dir / "task_information.txt").write_text(task_information, encoding="utf-8")
    (work_dir / "faulty_function.txt").write_text(faulty_function, encoding="utf-8")
    write_json(
        work_dir / "conv_patch_initial.json",
        build_patch_messages(task_information=task_information, location=location),
    )

    if options.dry_run:
        return build_base_result(
            status="dry_run",
            context=context,
            location=location,
            options=options,
            task_information=task_information,
            faulty_function=faulty_function,
            extra={"message": "Dry run completed before model invocation."},
        )

    model = ChatModel(model_name=options.model_name, temperature=options.temperature)
    patch_agent = PatchAgent(
        context=context,
        location=location,
        task_information=task_information,
        model=model,
        task_dir=work_dir,
    )
    test_agent = TestAgent(
        context=context,
        location=location,
        task_dir=work_dir,
        max_reproducer_tests=options.max_reproducer_tests,
    )

    reproduced = False
    repro_result_map = {}
    if options.reproduce_and_review:
        try:
            test_handle, _test_content, orig_repro = (
                test_agent.write_reproducing_test_without_feedback()
            )
            repro_result_map[(PatchAgent.EMPTY_PATCH_HANDLE, test_handle)] = orig_repro
            reproduced = orig_repro.reproduced
        except NoReproductionStep as exc:
            write_json(
                work_dir / "reproducer_status.json",
                {"status": "no_reproducer", "message": str(exc)},
            )

    review_manager = ReviewManager(
        context=context,
        location=location,
        issue_statement=task_information,
        patch_agent=patch_agent,
        test_agent=test_agent,
        model=model,
        output_dir=work_dir,
        repro_result_map=repro_result_map,
    )

    if options.reproduce_and_review and reproduced:
        patch_generator = review_manager.generator(
            rounds=options.review_rounds,
            patch_retries=options.max_patch_retries,
        )
        workflow = "reproduce_and_review"
    else:
        patch_generator = review_manager.patch_only_generator(
            retries=options.max_patch_retries,
        )
        workflow = "patch_only"

    evaluation_msg = None
    last_patch_handle = None
    last_fix_code = None
    last_validation: ValidationResult | None = None
    max_outer_rounds = max(1, options.review_rounds)

    for _ in range(max_outer_rounds):
        try:
            patch_handle, fix_code = patch_generator.send(evaluation_msg)
        except StopIteration:
            break

        last_patch_handle = patch_handle
        last_fix_code = fix_code

        if options.enable_validation:
            validation_result = evaluate_patch(
                context=context,
                location=location,
                fix_code=fix_code,
                output_dir=work_dir,
                timeout_seconds=options.validation_timeout_seconds,
                memory_limit_mb=options.validation_memory_limit_mb,
            )
            last_validation = validation_result
            write_json(
                work_dir / f"validation_{patch_handle}.json",
                validation_result.to_json(),
            )
            if validation_result.passed:
                patch_generator.close()
                return build_success_result(
                    context=context,
                    location=location,
                    options=options,
                    task_information=task_information,
                    faulty_function=faulty_function,
                    fix_code=fix_code,
                    raw_patch_response=patch_agent._responses.get(patch_handle, ""),
                    patch_handle=patch_handle,
                    workflow=workflow,
                    patch_agent=patch_agent,
                    validation_result=validation_result,
                    work_dir=work_dir,
                )
            evaluation_msg = validation_result.message
            continue

        patch_generator.close()
        return build_success_result(
            context=context,
            location=location,
            options=options,
            task_information=task_information,
            faulty_function=faulty_function,
            fix_code=fix_code,
            raw_patch_response=patch_agent._responses.get(patch_handle, ""),
            patch_handle=patch_handle,
            workflow=workflow,
            patch_agent=patch_agent,
            validation_result=None,
            work_dir=work_dir,
        )

    return build_error_result(
        context=context,
        location=location,
        options=options,
        task_information=task_information,
        faulty_function=faulty_function,
        raw_patch_response=(
            patch_agent._responses.get(last_patch_handle, "")
            if last_patch_handle is not None
            else ""
        ),
        fix_code=last_fix_code,
        patch_handle=last_patch_handle,
        workflow=workflow,
        patch_agent=patch_agent,
        validation_result=last_validation,
        work_dir=work_dir,
    )


def get_cached_task_information(
    context: ProgramContext,
    task_information_cache: dict[str, str] | None,
) -> str:
    if task_information_cache is not None and context.task_name in task_information_cache:
        return task_information_cache[context.task_name]
    task_information = get_task_sections(context.task_name)
    if task_information_cache is not None:
        task_information_cache[context.task_name] = task_information
    return task_information


def build_patch_messages(
    task_information: str,
    location: FaultyFunctionLocation,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PATCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the issue:\n{task_information}"},
        {"role": "user", "content": construct_code_context_prompt(location)},
        {"role": "user", "content": PATCH_USER_PROMPT_INIT},
    ]


def construct_code_context_prompt(location: FaultyFunctionLocation) -> str:
    prompt = "Here are the possible buggy locations collected by someone else. "
    prompt += (
        "Each location contains the actual code snippet and the intended behavior of "
        "the code for resolving the issue.\n"
    )
    prompt += bug_location_to_model_string(location)
    prompt += (
        "Note that you DO NOT NEED to modify every location; you should think what changes "
        "are necessary for resolving the issue, and only propose those modifications."
    )
    return prompt


def bug_location_to_model_string(location: FaultyFunctionLocation) -> str:
    intended_behavior = (
        "This code should implement the task described by the Problem Statement and "
        "Constraints while preserving the existing input/output contract."
    )
    return (
        "Location #1:\n"
        f"<file>{location.context.file_name}</file>\n"
        "<class></class>\n"
        "<method></method>\n"
        f"<code>\n{location.code}\n</code>"
        f"<intended_behavior>{intended_behavior}</intended_behavior>\n\n"
    )


def build_success_result(
    context: ProgramContext,
    location: FaultyFunctionLocation,
    options: RepairRunOptions,
    task_information: str,
    faulty_function: str,
    fix_code: str,
    raw_patch_response: str,
    patch_handle: str,
    workflow: str,
    patch_agent: PatchAgent,
    validation_result: ValidationResult | None,
    work_dir: Path,
) -> dict[str, Any]:
    extra = {
        "fix_code": fix_code,
        "raw_patch_response": raw_patch_response,
        "patch_handle": patch_handle,
        "workflow": workflow,
        "usage": {"patch_generation": patch_agent.usage_records},
        "estimated_cost": {"total": sum_known_costs(patch_agent.usage_records)},
    }
    if validation_result is not None:
        extra["validation"] = validation_result.to_json()
    return build_base_result(
        status="success",
        context=context,
        location=location,
        options=options,
        task_information=task_information,
        faulty_function=faulty_function,
        extra=extra,
    )


def build_error_result(
    context: ProgramContext,
    location: FaultyFunctionLocation,
    options: RepairRunOptions,
    task_information: str,
    faulty_function: str,
    raw_patch_response: str,
    fix_code: str | None,
    patch_handle: str | None,
    workflow: str,
    patch_agent: PatchAgent,
    validation_result: ValidationResult | None,
    work_dir: Path,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "raw_patch_response": raw_patch_response,
        "fix_code": fix_code,
        "patch_handle": patch_handle,
        "workflow": workflow,
        "usage": {"patch_generation": patch_agent.usage_records},
        "estimated_cost": {"total": sum_known_costs(patch_agent.usage_records)},
    }
    if validation_result is not None:
        extra["validation"] = validation_result.to_json()
    return build_base_result(
        status="error",
        context=context,
        location=location,
        options=options,
        task_information=task_information,
        faulty_function=faulty_function,
        extra=extra,
    )


def build_base_result(
    status: str,
    context: ProgramContext,
    location: FaultyFunctionLocation,
    options: RepairRunOptions,
    task_information: str,
    faulty_function: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "programId": context.program_id,
        "repair_index": None,
        "task_name": context.task_name,
        "language": context.language,
        "reproduce_and_review": options.reproduce_and_review,
        "enable_validation": options.enable_validation,
        "review_rounds": options.review_rounds,
        "validation_timeout_seconds": options.validation_timeout_seconds,
        "validation_memory_limit_mb": options.validation_memory_limit_mb,
        "generated_at": datetime.now().isoformat(),
        "task_information": task_information,
        "faulty_function": faulty_function,
    }
    payload.update(extra)
    return payload


def sum_known_costs(records: list[dict[str, Any]]) -> float | None:
    total = 0.0
    seen = False
    for record in records:
        cost = record.get("estimated_cost_usd")
        if isinstance(cost, (int, float)):
            total += float(cost)
            seen = True
    return total if seen else None


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
