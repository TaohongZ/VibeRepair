import os
import re
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI


from tools import *  # expects Joern-based tools + example_patch_search_tool


def extract_and_save_final_answer(text: str, output_file: str, name: str) -> Dict[str, Any]:
    # ---- 1. Extract Correct Intended Behavior ----
    cib_match = re.search(
        r"Correct Intended Behavior:\s*(.+?)(?=\n\s*Root Cause:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if not cib_match:
        raise ValueError("Correct Intended Behavior not found in output")

    cib_block = cib_match.group(1).strip()

    
    cib_items = re.findall(
        r"\d+\.\s*(.+?)(?=\n\d+\.|\Z)",
        cib_block,
        re.DOTALL
    )
    cib_items = [item.strip() for item in cib_items]

    # ---- 2. Extract Root Cause ----
    root_cause_match = re.search(
        r"Root Cause:\s*(.+?)(?=\n\s*Suggestion|\Z)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if not root_cause_match:
        raise ValueError("Root Cause not found in the text")

    root_cause = root_cause_match.group(1).strip()

    # ---- 3. Extract Suggestions ----
    suggestions_matches = re.finditer(
        r"Suggestion\s*(\d+):?\s*(.*?)(?=\n\s*Suggestion|\Z)",
        text,
        re.DOTALL | re.IGNORECASE
    )

    suggestions = []
    for m in suggestions_matches:
        num = m.group(1)
        content = m.group(2).strip()

        lines = content.split("\n", 1)
        title = lines[0].strip()
        details = lines[1].strip() if len(lines) > 1 else ""

        suggestions.append(f"Suggestion {num}. {title} {details}")

    # ---- 4. Save to JSON ----
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[name] = {
        "Correct Intended Behavior": cib_items,
        "Root Cause": root_cause,
        "Suggestions": suggestions
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results




class Logger(object):
    def __init__(self, filename: str = 'default.log', stream=sys.stdout):
        self.terminal = stream
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'a', encoding='utf-8')
        self._closed = False

    def _log(self, level: str, message: str):
        if message is None:
            return
        if str(message).strip():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            msg = f'[{timestamp}] [{level.upper()}] {message}\n'
            self.terminal.write(msg)
            self.log.write(msg)

    def info(self, message: str):
        self._log('info', message)

    def error(self, message: str):
        self._log('error', message)

    def warning(self, message: str):
        self._log('warning', message)

    def write(self, message: str):
        
        if message is None:
            return
        if str(message).strip():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            msg = f'[{timestamp}] {message}'
        else:
            msg = message
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if not self._closed:
            try:
                self.log.close()
            finally:
                self._closed = True


# Tools used by the ReAct agent (same as your original script)
TOOLS = [open_proj_tool, trace_method_usage_tool, find_method_in_file_tool, analyze_method_details_tool, analyze_method_control_flow_tool,
         find_class_loc_tool, identify_class_tool, get_imports_tool, close_proj_tool, example_patch_search_tool]

PROMPT_SOLUTION_TEMPLATE = PromptTemplate.from_template(
    """
You are an expert Java programmer and specification/code analyzer. 
Your task is to analyze a given set of Java flawed specifications which lead to a bug (multi-function bug), infer the correct intended behavior, 
identify the root cause of the flaw at the specification level, and then provide concrete suggestions on how to rewrite 
all flawed specifications into correct specifications.

## Inputs

You are given the following information:

- buggy_id: {buggy_id}
- Buggy File Path List: {buggy_file_path_list}
- Flawed Specification List: {flawed_specification_list}
- Buggy Function List: {buggy_code_list}
- Trigger Test: {trigger_src}
- Error Message: {err_msg}

### Flawed specification

The flawed_specification describes the intended behavior of a single method (or a small set of related behaviors).
It contains a "Behavior" section. In that section, behaviors that are WRONG are wrapped in brackets, e.g.:

(The behavior may contain potential errors.)

These bracketed behaviors are incorrect and must be revised.

Your first goal is NOT to fix the code, but to:
1. Carefully read all flawed_specification in the Flawed Specification List, especially their Behavior section.
2. Infer the **correct intended behavior** (Correct Intended Behavior) that methods SHOULD follow.
3. Explain the **root cause** of why flawed specifications are wrong (e.g., missing edge cases, wrong return condition, wrong invariants, misunderstanding of domain rules, etc.).

Only AFTER you have inferred the Correct Intended Behavior and Root Cause, you will then:
4. Use all buggy code from the Buggy Function List and the Root Cause as input to the example patch search tool to retrieve similar fix patterns.
5. Based on the retrieved examples, the Correct Intended Behavior, and the Root Cause, produce the final answer.
6. In the final answer, your Suggestions must describe **how to modify all flawed specifications** to become correct specifications, 
   NOT how to modify the code.

## Available Tools

You have access to the following Joern-based tools and one Example Patch Search tool::
{tools}
{tool_names}

Each tool call must follow the ReAct pattern described below. 
Do not invent new tools. Only use tools from the list above.

- IMPORTANT: Do not modify tool parameters in any way. Use them exactly as provided in the prompt variables 
  (e.g., open(Chart-1_buggy) rather than open("Chart_1_buggy")). 
- IMPORTANT: Do not add extra quotes, escape characters, or additional formatting around tool parameters.
  The parameter must be passed exactly as plain text.

## ReAct Tool Usage Format

During **all intermediate steps (before the final answer)**, you MUST strictly follow this format:

Thought: [your reasoning about why to use a tool and which one]
Action: [the exact tool name from the tool list, e.g., open_proj_tool]
Action Input: [the input for the tool, exactly as required by that tool]

After the tool is executed, you will be given an:

Observation: [the tool's output]

Then you continue reasoning with:

Thought: [your updated reasoning based on the observation]

You can repeat Thought → Action → Action Input → Observation as many times as needed.

- During intermediate steps, your response **must start with `Thought:`**.
- You may include at most **one** Action / Action Input block per step.
- You must NOT output the final answer until you have finished all necessary tool calls and closed the project.

### Mandatory tools

- You must call **open_proj_tool** at the beginning of your analysis to open the CPG for the buggy project.
- You must call **close_proj_tool** exactly once before you output the final answer, to close the CPG.
- At some point AFTER you have inferred the Correct Intended Behavior and Root Cause, 
  you MUST call **example_patch_search_tool** exactly once, with the following Action Input:

  - A single plain-text string that is:  
    "Buggy code" concatenated with "Root Cause".  
    For example:  
    `public boolean equals(Object obj) {{ ... }}, Root Cause: The equals method ...`

  This string should contain:
  - The full buggy method code (buggy_code), and
  - The Root Cause you have identified.

After you have called close_proj_tool and received its Observation, you must NOT call any more tools.  
Only then are you allowed to output the final answer.

## Analysis Process

Follow this high-level process:

1. **Understand the Flawed Specification**
   - Read the flawed_specification carefully.
   - Focus on the Behavior section.
   - Identify all behaviors wrapped in brackets ( ... ) as flawed behaviors.
   - Think about what the behavior SHOULD be, based on:
     - The rest of the specification,
     - The trigger test code (trigger_src),
     - The error messages (err_msg),
     - And, when needed, insights from the actual buggy code.

2. **Plan How to Infer the Correct Intended Behavior**
   - Think about which tools can help you understand the method’s real semantics, constraints, and usage:
     - use trace_method_usage_tool to see how the method is used,
     - use find_method_in_file_tool to locate the exact method definition,
     - use analyze_method_details_tool and analyze_method_control_flow_tool to understand parameters, return values, and control flow,
     - use find_class_loc_tool, identify_class_tool, get_imports_tool to understand the surrounding context.
   - Before calling any tool, always write a clear "Thought" explaining why you need that tool.

3. **Use Tools to Gather Evidence**
   - Iteratively call tools using the ReAct format (Thought → Action → Action Input → Observation).
   - From the code, tests, and error messages, determine what behavior is actually expected at runtime.
   - Your target is to build a clear and precise **Correct Intended Behavior** description that can replace the flawed behaviors in the specification.

4. **Write the Correct Intended Behavior**
   - After you have enough evidence, stop using tools for a moment and summarize a precise **Correct Intended Behavior** section.
   - This section should:
     - Cover all relevant cases (normal cases, edge cases, error cases, etc.).
     - Clarify the method’s inputs, outputs, and important invariants.
   - You will later include this as a dedicated section in your Final Answer.

5. **Derive the Root Cause (Specification-Level)**
   - Compare the flawed behaviors (the bracketed ones) with your Correct Intended Behavior.
   - Explain why the flawed specification is wrong:
     - e.g., it misstates the return value in a certain condition,
     - omits required validation,
     - uses the wrong domain rule,
     - or contradicts the real expected semantics as implied by tests/error messages.
   - Summarize this as a single, clear **Root Cause** description.

6. **Derive concrete suggestions**
    - Derive concrete suggestions on how to rewrite the flawed specification into a correct one, based on the Correct Intended Behavior and the Root Cause.

7. **Call the Example Patch Search Tool**
   - Once you have the Root Cause and you know all the buggy_code, you MUST call the example_patch_search_tool using a **single plain-text input string**.

   - Format the Action Input EXACTLY as follows:
       t:
         [FUNCTION 1]
         buggy_code_list[0]
    
         [FUNCTION 2]
         buggy_code_list[1]
    
         ...
    
         [ROOT_CAUSE]
         root_cause
   - Rules:
     - Replace `buggy_code_list[i]` with the exact corresponding code, without modification.
     - Do NOT merge, rewrite, or summarize the code.
     - Do NOT change the order of functions.
     - Do NOT add quotes or JSON formatting.
     
   - Then perform the tool call:
   
     Thought: [e.g., I should search for similar fix patterns to validate and refine my understanding.]
     Action: example_patch_search_tool
     Action Input: [the formatted plain-text string above]

   - Wait for the Observation from the example_patch_search_tool.
   - The Observation will include example fix patterns (buggy code, fixed code, root cause, etc.) from a knowledge base.
   
8. **Refine Your Understanding and Suggestions**
   - Use the Observation from example_patch_search_tool to:
     - Validate your Correct Intended Behavior,
     - Refine or slightly adjust your Root Cause if needed,
     - Derive concrete suggestions on how to rewrite the flawed specification into a correct one.
     
9. **Close the Project**
   - Before you output your final answer, call:

     Thought: [I have finished my analysis, now I should close the CPG project.]
     Action: close_proj_tool
     Action Input: [appropriate project name / identifier]

   - After the Observation from close_proj_tool, you should NOT call any more tools.
   - Your **next and final response** after closing the project must be the Final Answer, and nothing else.

## Final Answer Format (VERY IMPORTANT)

At the end of your analysis, you must output exactly **one** final answer in the following format:

- The final answer MUST start with the exact string:  
  **`Final Answer:`**

- In the final answer, you must NOT include any Thought, Action, Action Input, or Observation lines.

The required structure is:

Final Answer:
Correct Intended Behavior:
1. [First behavior rule, clearly and precisely stated]
2. [Second behavior rule, clearly and precisely stated]
[Add more numbered items as needed to fully describe the correct behavior]

Root Cause: [A clear and concise explanation of why flawed specifications are wrong, focusing on the mismatch between the flawed behavior and the correct intended behavior.]

Suggestion 1: [Short title describing the change to the flawed specification 1]
[Detailed, step-by-step guidance on how to modify the flawed specification to obtain the correct specification. 
Explain exactly which behavior statements should be rewritten, added, or removed. 
Do not describe code changes here; describe SPECIFICATION changes only.]

Suggestion 2: [Short title describing the change to the flawed specification 2]
[Detailed, step-by-step guidance on how to modify the flawed specification to obtain the correct specification. 
Explain exactly which behavior statements should be rewritten, added, or removed. 
Do not describe code changes here; describe SPECIFICATION changes only.]

[Add more suggestions as needed. Each flawed specification should have a corresponding suggestion. Always focused on improving the specification, 
and always following the same format as above.]

- Do NOT output any additional commentary after the Final Answer block.
- Do NOT include your internal chain-of-thought or tool call details in the Final Answer.
- Do NOT include any Thought, Action, Action Input, or Observation lines in the Final Answer.
- Make sure your Final Answer strictly follows the structure and labels above.

## Begin Your Analysis

Start by analyzing the flawed specification and planning which tools you need to infer the correct intended behavior. 
Then use the available tools to gather information from the code, tests, and project context. 
Remember to:
- open the CPG with open_proj_tool,
- infer the Correct Intended Behavior,
- derive the Root Cause,
- call example_patch_search_tool with buggy code + Root Cause,
- close the CPG with close_proj_tool, and
- finally output your Final Answer in the required format.

Thought: {agent_scratchpad}

"""
)

def run_agent_for_bug(
    *,
    bug_id: str,
    function_list,
    flawed_specification_list,
    path_list,
    trigger_src_list,
    err_msg_list,
    llm: Optional[ChatOpenAI] = None,
    tools_list: Optional[list] = None,
) -> Dict[str, Any]:
    """
    One call = one LLM analysis run:
      - Runs ReAct agent to produce Root Cause + Suggestions (Final Answer).
      - Calls extract_and_save_final_answer(...) to parse & append into JSON.
      - Returns the whole JSON dict after write.

    This function does NOT generate patches (removed by request).
    """
    log_path = f"./output/log/{bug_id}.log"
    json_output_path = f"./output/mf_solutions_no_valid/{bug_id}.json"

    original_stdout = sys.stdout
    logger = Logger(log_path, sys.stdout)
    sys.stdout = logger  # redirect stdout to file + console

    try:
        start_time = datetime.now()
        total_cost = 0.0

        if llm is None:
            llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=1)

        if tools_list is None:
            tools_list = TOOLS

        prompt_solution = PROMPT_SOLUTION_TEMPLATE.partial(
            buggy_id=bug_id + "_buggy",
            buggy_file_path_list=path_list,
            flawed_specification_list=flawed_specification_list,
            buggy_code_list=function_list,
            trigger_src=trigger_src_list,
            err_msg=err_msg_list,
        )

        agent = create_react_agent(llm, tools_list, prompt_solution)
        agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)

        try:
            with get_openai_callback() as cb:
                result = agent_executor.invoke({})
                logger.info(f"Agent Total tokens: {cb.total_tokens}")
                logger.info(f"Agent Prompt tokens: {cb.prompt_tokens}")
                logger.info(f"Agent Completion tokens: {cb.completion_tokens}")
                logger.info(f"Agent Successful Requests: {cb.successful_requests}")
                logger.info(f"Agent Total Cost (USD): {cb.total_cost}")
                total_cost += cb.total_cost

            logger.info(result["output"])
            parsed = extract_and_save_final_answer(result["output"], json_output_path, bug_id)

        except ValueError as e:
            err = str(e)
            prefix = "Could not parse LLM output: `"
            suffix = "`"

            if err.startswith(prefix) and err.endswith(suffix):
                
                final_text = err[len(prefix):-len(suffix)]
                logger.warning("Parsing error occurred, using wrapped LLM output as Final Answer fallback.")
                logger.info(final_text)

                try:
                    parsed = extract_and_save_final_answer(final_text, json_output_path, bug_id)
                except Exception as parse_e:
                    logger.error(f"Failed to parse fallback Final Answer text: {parse_e}")
                    raise
            else:
                raise

        end_time = datetime.now()
        logger.info(f"Time cost: {end_time - start_time}")
        logger.info(f"Money cost for {bug_id}: {total_cost}")

        return parsed

    finally:
        sys.stdout = original_stdout
        logger.close()
        time.sleep(10)

