from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:  # pragma: no cover - depends on installed langchain version
    AgentExecutor = None
    create_react_agent = None
try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:  # pragma: no cover - depends on installed langchain version
    try:
        from langchain.callbacks import get_openai_callback
    except ImportError:  # pragma: no cover
        get_openai_callback = None
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI, OpenAIError

from .config import (
    DEFAULT_AGENT_MODEL_NAME,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PATCH_MODEL_NAME,
    DEFAULT_TEST_ROOT,
    OPENAI_API_KEY,
)
from .condefects_dataset import (
    ProgramContext,
    ProgramFunctionLookupError,
    build_trigger_context,
    get_ochiai_faulty_function_with_reference_comment,
    iter_program_contexts,
)
from .tools import (
    analyze_method_control_flow_tool,
    analyze_method_details_tool,
    close_proj_tool,
    example_patch_search_tool,
    find_class_loc_tool,
    find_method_in_file_tool,
    get_imports_tool,
    identify_class_tool,
    open_proj_tool,
    trace_method_usage_tool,
)


if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def extract_and_save_final_answer(text, output_file, name):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    root_cause_match = re.search(r"(?i)root\s*cause:?\s*(.+?)(?=\n+ *suggestion|\Z)", text, re.DOTALL | re.IGNORECASE)
    suggestions_matches = re.finditer(r"(?i)suggestion\s*(\d+):?\s*(.*?)(?=\n\s*suggestion|\Z)", text,
                                      re.DOTALL | re.IGNORECASE)

    if not root_cause_match:
        raise ValueError("Root Cause not found in the text")

    root_cause = root_cause_match.group(1).strip()
    suggestions = []

    for match in suggestions_matches:
        suggestion_number = match.group(1)
        suggestion_text = match.group(2).strip()

        lines = suggestion_text.split('\n', 1)
        title = lines[0].strip()
        details = lines[1].strip() if len(lines) > 1 else ""

        suggestions.append("Suggestion " + suggestion_number + ". " + title + " " + details + "\n")
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_analyses = json.load(f)
    else:
        all_analyses = {}

    if name in all_analyses:
        all_analyses[name][root_cause] = suggestions
    else:
        all_analyses[name] = {
            root_cause: suggestions
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, ensure_ascii=False, indent=2)

    return all_analyses


def parse_patches(result):
    pattern = r'\[START PATCH \d+\]\n```\w*\n(.*?)\n```\n\[END PATCH \d+\]'
    patches = re.findall(pattern, result, re.DOTALL)

    return [patch.strip() for patch in patches]


def api_gpt_response(prompt, n, model_name=DEFAULT_PATCH_MODEL_NAME):
    client = OpenAI()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        n=n,
        temperature=0.8)
    return response


def append_patches_to_json(patches, file_path, project_name):
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(file_path):
        data = {}
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
    if project_name not in data:
        data[project_name] = {"patches": []}
    data[project_name]["patches"].extend(patches)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


tools = [open_proj_tool, trace_method_usage_tool, find_method_in_file_tool, analyze_method_details_tool,
         analyze_method_control_flow_tool,
         find_class_loc_tool, identify_class_tool, get_imports_tool, close_proj_tool, example_patch_search_tool]

prompt_solution_template = PromptTemplate.from_template("""
You are an expert Java programmer and code analyzer. Your task is to analyze a given Java code snippet, identify the root cause of a bug, and provide repair suggestions.

## Available Tools

You have access to the following Joern-based tools and one Example Patch Search tool:
{tools}
{tool_names}

for each tool call:
  "tool": "Tool to be called from tools, if applicable. Set to 'None' if no tool is needed.",
  "parameters": "The input parameter(s) used by the tools. Only the parameter value(s) (e.g. getIndexOf), (e.g. getLegendItems, AbstractCategoryItemRenderer.java). Set to 'None' if there are no parameters.". DO NOT try to modify any parameter format(e.g., open("Chart-1_buggy") rather than open("Chart_1_buggy\"\n") etc.). 

- IMPORTANT: Do not modify tool parameters in any way. Use them exactly as provided in the input context. DO NOT Add any quota marks to the parameter because the tool call will process it.

## Analysis Process

1. Open the CPG for the buggy code (always required as first stage)
2. Analyze the buggy code, focusing on the area marked with /* bug is here */
3. Use tools to gather necessary information about( DO NOT Add any quota marks to the input parameter because the tool call will process it.):
   - Variable definitions and usages
   - Function calls and definitions
   - Control flow
   - Similar patterns in the codebase
   - Import statements and library usage
4. Synthesize gathered information
5. Close the project (always required after gather enough information and before output the Summary Answer)
6. Output the Summary Answer.
7. Concatenate the complete Buggy function code and root cause, use them as the input parameter of calling example_patch_search_tool. It will return a fix pattern contain buggy code, fix code and root cause.
8. Observe the fix pattern of similar bug from example_patch_search_tool output as reference, learn and update the root cause and suggestions as the Final Answer.

## Guidelines
- Open and close the CPG project properly
- Focus on the buggy function and lines(before or at) marked with /* bug is here */
- Avoid unnecessary tool calls if information can be inferred from context
- Ensure your analysis covers context, potential causes, and fix suggestions
- Gather repair contexts as much as you need to draw the root cause.
- Call example_patch_search_tool if you think the joern analyze information is not enough to provide the Final Answer.


## Input Context

- Buggy ID(proj_name): {buggy_id}
- Buggy File: {buggy_file_path}
- Buggy Function: 
```java
{buggy_code}
```
- Trigger Test: {trigger_src}
- Error Message: {err_msg}

To help you better use example_patch_search_tool call, make sure the parameter is the full buggy code concatenate with the root cause, here is an Example:
Action Input:"input_string":  <Buggy code>, Root Cause: <content>"


## Output Format

Use the following format for your analysis:

Thought: [Your reasoning about the next step]
Action: [Tool name]
Action Input: "value1", "value2"
Observation: [Tool output]
... (Repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to provide the Summary Answer
Action: [Call example_patch_search_tool]
Action Input: "buggy_code, root_cause"
Observation: [Tool output]
Thought: I now have learnt the fix pattern, and I can update and output the Final Answer

Final Answer(IMPORTANT: Strictly follow the output format below and DO NOT add '#', '*' or other signs before each Root Cause & Suggestion title): 
Root Cause: [Detailed explanation of the bug's root cause]

Suggestion 1: [Suggestion title]
[Detailed description of the first repair suggestion, give step by step and sufficient suggestion to help generate correct patches. Only provide how to generate correct patch suggestions, not involve test process. Note that each suggestion should be independent]
Suggestion 2: [Suggestion title]
[Detailed description of the second repair suggestion, give step by step and sufficient suggestion to help generate correct patches. Only provide how to generate correct patch suggestions, not involve test process. Note that each suggestion should be independent]

[Add more suggestions as needed, give as much as it can help to generate correct patches. Please make sure your Final Answer follows the format above exactly, without adding any extra marks or symbols]

## Begin Your Analysis

Start by opening the CPG and analyzing the buggy code. Focus on the area marked with /* bug is here */ and use the available tools to gather necessary information.
Exit the chain when you already output the Final Answer.
Thought: {agent_scratchpad}
""")

tool_descriptions = "\n".join(
    f"{i + 1}. {tool.description}\n"
    for i, tool in enumerate(tools)
)
tool_names = "\n".join(
    f"{i + 1}. {tool.name}\n"
    for i, tool in enumerate(tools)
)


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'a', encoding='utf-8')
        self._closed = False

    def _log(self, level, message):
        if message.strip():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f'[{timestamp}] [{level.upper()}] {message}\n'
            self.terminal.write(message)
            self.log.write(message)

    def info(self, message):
        self._log('info', message)

    def error(self, message):
        self._log('error', message)

    def warning(self, message):
        self._log('warning', message)

    def write(self, message):
        if message.strip():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f'[{timestamp}] {message}'
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.flush()
        self.log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-function ReInFix on ConDefects datasetForExperiment programs."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--program-id", action="append", dest="program_ids")
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--solution-runs", type=int, default=1)
    parser.add_argument("--patch-samples", type=int, default=5)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL_NAME, choices=[DEFAULT_AGENT_MODEL_NAME])
    parser.add_argument("--patch-model", default=DEFAULT_PATCH_MODEL_NAME, choices=[DEFAULT_PATCH_MODEL_NAME])
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-patches", action="store_true")
    parser.add_argument("--no-reinfix-marker", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def create_chat_model(model_name: str):
    try:
        return ChatOpenAI(model_name=model_name)
    except TypeError:
        return ChatOpenAI(model=model_name)


MAX_REACT_STEPS = 15


def invoke_solution_agent(agent: Any, prompt: PromptTemplate, llm: Any, logger: Logger) -> dict[str, str]:
    if AgentExecutor is not None and agent is not None:
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            stream_runnable=False,
            handle_parsing_errors=(
                "Check your output and make sure it conforms, use the Action/Action Input "
                "syntax. Output the Final Answer always should always required as last step. "
                "Exit after output the Final Answer."
            ),
        )
        return agent_executor.invoke({})

    logger.warning("LangChain AgentExecutor is unavailable; using local ReAct runner.")
    return run_local_react_agent(prompt=prompt, llm=llm)


def run_local_react_agent(prompt: PromptTemplate, llm: Any) -> dict[str, str]:
    scratchpad = ""
    prompt_with_tools = prompt.partial(tools=tool_descriptions, tool_names=tool_names)

    for _ in range(MAX_REACT_STEPS):
        rendered_prompt = prompt_with_tools.format(agent_scratchpad=scratchpad)
        response = llm.invoke(rendered_prompt)
        generated_text = _message_content_to_text(getattr(response, "content", response)).strip()
        if not generated_text:
            raise ValueError("Could not parse LLM output: ``")

        if re.search(r"(?im)^\s*Final Answer", generated_text):
            return {"output": _extract_final_answer_text(generated_text)}

        parsed_action = _parse_react_action(generated_text)
        if parsed_action is None:
            raise ValueError(f"Could not parse LLM output: `{generated_text}`")

        action_name, action_input = parsed_action
        tool = _resolve_tool(action_name)
        if tool is None:
            observation = f"Unknown tool: {action_name}. Available tools: {', '.join(tool.name for tool in tools)}"
        else:
            observation = _call_tool(tool, action_input)

        scratchpad += f"{generated_text}\nObservation: {observation}\nThought: "

    raise ValueError(f"Agent stopped after {MAX_REACT_STEPS} steps without Final Answer.")


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_final_answer_text(text: str) -> str:
    match = re.search(r"(?is)Final Answer.*?:\s*(.*)$", text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def _parse_react_action(text: str) -> tuple[str, str] | None:
    action_matches = list(re.finditer(r"(?im)^\s*Action\s*:\s*(.*?)\s*$", text))
    if not action_matches:
        return None

    action_match = action_matches[-1]
    action_name = _clean_action_text(action_match.group(1))
    if not action_name or action_name.lower() == "none":
        return None

    remaining = text[action_match.end():]
    input_match = re.search(
        r"(?ims)^\s*Action Input\s*:\s*(.*?)(?=^\s*(?:Observation|Thought|Action|Final Answer)\s*:|\Z)",
        remaining,
    )
    if not input_match:
        return None
    return action_name, _clean_action_input(input_match.group(1))


def _clean_action_text(text: str) -> str:
    cleaned = text.strip().strip("` ").strip()
    cleaned = cleaned.strip("[]")
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    if "." in cleaned and cleaned.split(".", 1)[0].strip().isdigit():
        cleaned = cleaned.split(".", 1)[1].strip()
    return cleaned


def _clean_action_input(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    return cleaned


def _resolve_tool(action_name: str) -> Any | None:
    normalized_action = _normalize_tool_name(action_name)
    for index, tool in enumerate(tools, start=1):
        if action_name.strip() == str(index):
            return tool
        if normalized_action == _normalize_tool_name(tool.name):
            return tool
    for tool in tools:
        if normalized_action in _normalize_tool_name(tool.name):
            return tool
    return None


def _normalize_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _call_tool(tool: Any, action_input: str) -> str:
    try:
        if getattr(tool, "func", None) is not None:
            return str(tool.func(action_input))
        if hasattr(tool, "invoke"):
            return str(tool.invoke(action_input))
        if hasattr(tool, "run"):
            return str(tool.run(action_input))
        return str(tool(action_input))
    except Exception as exc:
        return f"Tool {getattr(tool, 'name', tool)} raised {type(exc).__name__}: {exc}"


def run_solution_generation(
    context: ProgramContext,
    faulty_function: str,
    fault_line_number: int,
    output_root: Path,
    args: argparse.Namespace,
    logger: Logger,
) -> Path:
    solution_path = output_root / "sf_solutions" / f"{safe_filename(context.program_id)}.json"
    if args.overwrite:
        write_json(solution_path, {})

    trigger_src_list = build_trigger_context(context, test_root=args.test_root)
    err_msg_list: list[str] = []
    prompt = prompt_solution_template.partial(
        buggy_id=context.program_id + "_buggy",
        buggy_file_path=str(context.source_path),
        buggy_code=faulty_function,
        trigger_src=str(trigger_src_list),
        err_msg=str(err_msg_list)
    )
    llm = create_chat_model(DEFAULT_AGENT_MODEL_NAME)
    agent = create_react_agent(llm, tools, prompt) if create_react_agent is not None else None

    for run_index in range(1, args.solution_runs + 1):
        logger.info(f"Generating Solution for: {context.program_id}, Round: {run_index}")
        try:
            if get_openai_callback is None:
                result = invoke_solution_agent(agent=agent, prompt=prompt, llm=llm, logger=logger)
            else:
                with get_openai_callback() as cb:
                    result = invoke_solution_agent(agent=agent, prompt=prompt, llm=llm, logger=logger)
                    logger.info(f"Agent Total tokens: {cb.total_tokens}")
                    logger.info(f"Agent Prompt tokens: {cb.prompt_tokens}")
                    logger.info(f"Agent Completion tokens: {cb.completion_tokens}")
                    logger.info(f"Agent Cost (USD): ${cb.total_cost}")
            logger.info(result['output'])
            extract_and_save_final_answer(result['output'], str(solution_path), context.program_id)
        except ValueError as exc:
            response = str(exc)
            if response.startswith("Could not parse LLM output: `"):
                response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            logger.error(response)
            raw_path = output_root / "raw_solution_errors" / f"{safe_filename(context.program_id)}_{run_index:02d}.txt"
            write_text(raw_path, response)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    metadata_path = output_root / "sf_solution_metadata" / f"{safe_filename(context.program_id)}.json"
    write_json(
        metadata_path,
        {
            "programId": context.program_id,
            "task_name": context.task_name,
            "language": context.language,
            "source_path": str(context.source_path),
            "solution_path": str(solution_path),
        },
    )
    return solution_path


def run_patch_generation(
    context: ProgramContext,
    faulty_function: str,
    fault_line_number: int,
    solution_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    logger: Logger,
) -> Path:
    output_patches_file = output_root / "sf_patches" / f"{safe_filename(context.program_id)}.json"
    output_patches_file.parent.mkdir(parents=True, exist_ok=True)
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file does not exist: {solution_path}")

    solutions = read_json(solution_path)
    if context.program_id not in solutions:
        raise ValueError(f"No parsed solutions for {context.program_id} in {solution_path}")

    trigger_src_list = build_trigger_context(context, test_root=args.test_root)
    err_msg_list: list[str] = []
    context_info = ["buggy_id: " + context.program_id + "_buggy", "\nbuggy_code: " + faulty_function,
                    "\nbuggy_file_path: " + str(context.source_path), "\ntrigger_src: " + str(trigger_src_list),
                    "\nerr_msg: " + str(err_msg_list)]

    for root_cause in solutions[context.program_id].keys():
        for suggestion in solutions[context.program_id][root_cause]:
            prompt_patch = f"""You expert Java programmer in generating patches for a given buggy snippet.\n Bug code information(buggy_id, buggy_code, buggy_file_path, trigger_src, err_msg):"+{str(context_info)}+"\nYour task is to generate ONE unique version of fix for the buggy function code for the pair of root cause and suggestion: Root Cause: {str(root_cause)}\n{str(suggestion)}\n You should look deeply through all the bug information and combine the root cause and repair suggestions to generate correct patches,
    Requirement:
    1. For each root cause and suggestion pair, generate ONE patch.
    2. Each patch should be a complete version of the buggy function code with the fix applied.
    3. Ensure each patch is unique and addresses the identified issue.
    4. When looking at the buggy function code, the location of the bug is indicated by the /* bug is here */ comment.
    5. The fix may involve changes to surrounding lines, not just the commented line itself.

    Return the whole fixed buggy function code, NOT just the line(s) of fixed area. Only output the following content, nothing else. Please format your response as follows:
    [START PATCH 1]
    ```java
    <entire buggy function fixed code here>
    ```
    [END PATCH 1]

                """

            try:
                response = api_gpt_response(prompt_patch, args.patch_samples, model_name=DEFAULT_PATCH_MODEL_NAME)
                usage = response.usage
                if usage:
                    input_cost = usage.prompt_tokens * 0.000005
                    output_cost = usage.completion_tokens * 0.00002
                    patch_total_cost = input_cost + output_cost
                    logger.info(f"Patch prompt tokens: {usage.prompt_tokens}")
                    logger.info(f"Patch completion tokens: {usage.completion_tokens}")
                    logger.info(f"Patch total tokens: {usage.total_tokens}")
                    logger.info(f"Patch total cost $: {patch_total_cost}")
                else:
                    logger.info("No tokens recorded")

                print("***********Gen Patch***********")
                print(response)

                for choice in response.choices:
                    generated_text = choice.message.content.strip()
                    parsed_patches = parse_patches(generated_text)
                    logger.info(f"suggest_patch: {parsed_patches}")
                    append_patches_to_json(parsed_patches, str(output_patches_file), context.program_id)
                    for patch in parsed_patches:
                        write_fix_code_result(
                            output_root=output_root,
                            context=context,
                            fault_line_number=fault_line_number,
                            root_cause=root_cause,
                            suggestion=suggestion,
                            fix_code=patch,
                        )
            except OpenAIError as exc:
                logger.error(f'OpenAI API error: {exc}')
                break
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    return output_patches_file


def write_fix_code_result(
    output_root: Path,
    context: ProgramContext,
    fault_line_number: int,
    root_cause: str,
    suggestion: str,
    fix_code: str,
) -> None:
    result_dir = output_root / "fix_code_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(result_dir.glob(f"{safe_filename(context.program_id)}_repair_*.json"))
    repair_index = len(existing) + 1
    output_path = result_dir / f"{safe_filename(context.program_id)}_repair_{repair_index:03d}.json"
    write_json(
        output_path,
        {
            "status": "success",
            "programId": context.program_id,
            "repair_index": repair_index,
            "task_name": context.task_name,
            "language": context.language,
            "root_cause": root_cause,
            "suggestion": suggestion,
            "fix_code": fix_code,
        },
    )


def write_error_record(
    output_root: Path,
    context: ProgramContext | None,
    error: Exception,
    program_id: str | None = None,
) -> None:
    record_id = program_id or (context.program_id if context else "unknown")
    output_path = output_root / "errors" / f"{safe_filename(record_id)}_{int(time.time())}.json"
    payload = {
        "status": "error",
        "programId": record_id,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if context is not None:
        payload.update(
            {
                "task_name": context.task_name,
                "language": context.language,
            }
        )
    write_json(output_path, payload)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def ensure_environment() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)

    if args.solution_runs < 1:
        print("Error: --solution-runs must be at least 1", file=sys.stderr)
        return 1
    if args.patch_samples < 1:
        print("Error: --patch-samples must be at least 1", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1 when provided", file=sys.stderr)
        return 1

    try:
        ensure_environment()
        contexts = iter_program_contexts(
            dataset_root=args.dataset_root,
            program_ids=args.program_ids,
            require_ochiai=True,
            languages=args.languages,
        )
    except (EnvironmentError, ProgramFunctionLookupError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        contexts = contexts[: args.limit]

    print(
        f"Processing {len(contexts)} ConDefects programs with agent_model={DEFAULT_AGENT_MODEL_NAME}, "
        f"patch_model={DEFAULT_PATCH_MODEL_NAME}"
    )
    print("Reading fault lines from Ochiai_FL.txt")
    print(f"Writing outputs to {output_root}")

    completed = 0
    failed = 0
    skipped = 0
    original_stdout = sys.stdout

    for context in contexts:
        logger = Logger(str(output_root / "log" / f"{safe_filename(context.program_id)}.log"), sys.stdout)
        sys.stdout = logger
        try:
            solution_path = output_root / "sf_solutions" / f"{safe_filename(context.program_id)}.json"
            patches_path = output_root / "sf_patches" / f"{safe_filename(context.program_id)}.json"
            if (
                solution_path.exists()
                and (args.skip_patches or patches_path.exists())
                and not args.overwrite
            ):
                skipped += 1
                logger.info(f"[skip] {context.program_id}")
                continue

            start_time = datetime.now()
            faulty_function, fault_line_number = get_ochiai_faulty_function_with_reference_comment(
                context,
                include_reinfix_marker=not args.no_reinfix_marker,
            )
            generated_solution_path = run_solution_generation(
                context=context,
                faulty_function=faulty_function,
                fault_line_number=fault_line_number,
                output_root=output_root,
                args=args,
                logger=logger,
            )
            if not args.skip_patches:
                run_patch_generation(
                    context=context,
                    faulty_function=faulty_function,
                    fault_line_number=fault_line_number,
                    solution_path=generated_solution_path,
                    output_root=output_root,
                    args=args,
                    logger=logger,
                )
            completed += 1
            logger.info(f"Time cost for {context.program_id}: {datetime.now() - start_time}")
        except Exception as exc:
            failed += 1
            logger.error(f"[error] {context.program_id}: {exc}")
            write_error_record(output_root, context, exc)
            if args.stop_on_error:
                return 1
        finally:
            sys.stdout = original_stdout
            logger.close()
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"Finished: completed={completed}, skipped={skipped}, failed={failed}, total={len(contexts)}")
    return 1 if failed and args.stop_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

