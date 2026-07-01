from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ConDefects.util.get_faulty_function_with_bug_comment_by_program_id import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    ProgramFunctionLookupError,
    get_faulty_function_with_bug_comment_by_program_id,
)
from ConDefects.util.get_task_sections import (  # noqa: E402
    TaskSectionFetchError,
    get_task_sections,
)
from viberepair.langchain_support import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    build_chat_model,
)


@dataclass(slots=True, frozen=True)
class ProgramContext:
    program_id: str
    program_dir: Path
    task_name: str
    language: str


@dataclass(slots=True, frozen=True)
class DemoPipelineOutput:
    context: ProgramContext
    faulty_function: str
    task_information: str
    flawed_specification_result: Any
    fix_specification_result: Any
    fix_code_result: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small VibeRepair demo for a ConDefects program id: "
            "faulty_function -> flawed_specification -> fixed specification -> fix code."
        )
    )
    parser.add_argument("program_id", help="ConDefects program id, for example 51970859")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the ConDefects dataset_2024 directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Chat model name used by all VibeRepair stages. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature used by all VibeRepair stages. Default: {DEFAULT_TEMPERATURE}",
    )
    return parser.parse_args()


def find_program_dir(program_id: str, dataset_root: str | Path) -> Path:
    normalized_program_id = str(program_id).strip()
    if not normalized_program_id:
        raise ValueError("program_id is required")

    dataset_root = Path(dataset_root)
    matches = []
    for candidate in dataset_root.rglob(normalized_program_id):
        if not candidate.is_dir() or candidate.name != normalized_program_id:
            continue
        if (candidate / "faultyVersion.py").exists() or (candidate / "faultyVersion.java").exists():
            matches.append(candidate)

    if not matches:
        raise ProgramFunctionLookupError(
            f"No folder named {normalized_program_id} was found under {dataset_root}"
        )
    if len(matches) > 1:
        raise ProgramFunctionLookupError(
            f"Multiple program folders named {normalized_program_id} were found: {matches}"
        )
    return matches[0]


def resolve_program_context(program_id: str, dataset_root: str | Path) -> ProgramContext:
    program_dir = find_program_dir(program_id, dataset_root)

    try:
        language = program_dir.parent.name
        task_name = program_dir.parents[1].name
    except IndexError as exc:
        raise ProgramFunctionLookupError(
            f"Unexpected dataset layout for program {program_id}: {program_dir}"
        ) from exc

    return ProgramContext(
        program_id=str(program_id).strip(),
        program_dir=program_dir,
        task_name=task_name,
        language=language,
    )


def build_fix_specification_input(task_information: str, flawed_specification: str) -> str:
    return (
        "Task Information\n"
        f"{task_information.strip()}\n\n"
        "Flawed Specification\n"
        f"{flawed_specification.strip()}"
    )


def import_viberepair_stage_functions() -> tuple[Any, Any, Any]:
    try:
        from viberepair.modules.buggy_code_to_flawed_specification import (
            generate_flawed_specification_result,
        )
        from viberepair.modules.fix_specification import generate_fix_specification_result
        from viberepair.modules.fix_specification_and_buggy_code_to_fix_code import (
            generate_fix_code_result,
        )
    except ImportError as exc:
        raise ImportError(
            "VibeRepair pipeline requires `langchain>=1.0.0`, `langchain-core`, and "
            "`langchain-openai`. Install them with "
            "`pip install -U langchain langchain-core langchain-openai`."
        ) from exc

    return (
        generate_flawed_specification_result,
        generate_fix_specification_result,
        generate_fix_code_result,
    )


def run_demo_pipeline(
    program_id: str,
    dataset_root: str | Path,
    model_name: str,
    temperature: float,
) -> DemoPipelineOutput:
    (
        generate_flawed_specification_result,
        generate_fix_specification_result,
        generate_fix_code_result,
    ) = import_viberepair_stage_functions()

    faulty_function = get_faulty_function_with_bug_comment_by_program_id(program_id, dataset_root=dataset_root)
    context = resolve_program_context(program_id, dataset_root=dataset_root)
    task_information = get_task_sections(context.task_name)

    model = build_chat_model(model_name=model_name, temperature=temperature)

    flawed_specification_result = generate_flawed_specification_result(
        buggy_code=faulty_function,
        model_name=model_name,
        temperature=temperature,
        model=model,
    )

    fix_specification_input = build_fix_specification_input(
        task_information=task_information,
        flawed_specification=flawed_specification_result.flawed_specification,
    )
    fix_specification_result = generate_fix_specification_result(
        information=fix_specification_input,
        model_name=model_name,
        temperature=temperature,
        model=model,
    )

    fix_code_result = generate_fix_code_result(
        fix_specification=fix_specification_result.fix_specification,
        buggy_code=faulty_function,
        model_name=model_name,
        temperature=temperature,
        model=model,
    )

    return DemoPipelineOutput(
        context=context,
        faulty_function=faulty_function,
        task_information=task_information,
        flawed_specification_result=flawed_specification_result,
        fix_specification_result=fix_specification_result,
        fix_code_result=fix_code_result,
    )


def print_block(title: str, content: str) -> None:
    print(f"\n=== {title} ===")
    print(content.rstrip())


def print_fix_specification_reasoning(result: Any) -> None:
    print("\n=== Fix Specification Reasoning ===")
    print("Intended Behavior:")
    print((result.intended_behavior or "").rstrip())
    print("\nDiagnosis and Repair Directions:")
    print((result.diagnosis_and_repair_directions or "").rstrip())
    print("\nFixed Specification:")
    print((result.fix_specification or "").rstrip())


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "unknown"
    return f"${cost:.6f}"


def print_summary(output: DemoPipelineOutput) -> None:
    print(f"Program ID: {output.context.program_id}")
    print(f"Task Name: {output.context.task_name}")
    print(f"Language: {output.context.language}")
    print(f"Program Dir: {output.context.program_dir}")

    print_block("Faulty Function", output.faulty_function)
    print_block("Task Information", output.task_information)
    print_block(
        "Flawed Specification",
        output.flawed_specification_result.flawed_specification,
    )
    print_fix_specification_reasoning(output.fix_specification_result)
    print_block("Fix Code", output.fix_code_result.replacement_code)

    print("\n=== Stage Metadata ===")
    print(
        "buggy_code_to_flawed_specification: "
        f"tokens_in={output.flawed_specification_result.usage.input_tokens}, "
        f"tokens_out={output.flawed_specification_result.usage.output_tokens}, "
        f"estimated_cost={format_cost(output.flawed_specification_result.estimated_cost_usd)}"
    )
    print(
        "fix_specification: "
        f"tokens_in={output.fix_specification_result.usage.input_tokens}, "
        f"tokens_out={output.fix_specification_result.usage.output_tokens}, "
        f"estimated_cost={format_cost(output.fix_specification_result.estimated_cost_usd)}"
    )
    print(
        "fix_specification_and_buggy_code_to_fix_code: "
        f"tokens_in={output.fix_code_result.usage.input_tokens}, "
        f"tokens_out={output.fix_code_result.usage.output_tokens}, "
        f"estimated_cost={format_cost(output.fix_code_result.estimated_cost_usd)}"
    )


def main() -> int:
    args = parse_args()

    try:
        output = run_demo_pipeline(
            program_id=args.program_id,
            dataset_root=args.dataset_root,
            model_name=args.model,
            temperature=args.temperature,
        )
    except (
        EnvironmentError,
        ImportError,
        ProgramFunctionLookupError,
        TaskSectionFetchError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
