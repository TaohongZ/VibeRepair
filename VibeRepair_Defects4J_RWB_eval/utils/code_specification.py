from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def code_gen_spec(code: str) -> str:
    
    
    PRICE_INPUT = 5.0 / 1_000_000    
    PRICE_OUTPUT = 15.0 / 1_000_000

    
    model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=1)

    # system 提示词
    system_message = """
You are a code analyst. You will be provided with a piece of code to analyze.
Your task is to determine the intent of the code and fill in the specification template according to the provided template and its corresponding guidelines.
The completed specification should enable developers to clearly understand the code’s purpose and allow them to reimplement the code with the same intent based on the specification.

The Specification Template:
[
  Function: #Function name
  Purpose: #Purpose of this function
  Signature: #The unique identifier of a function, defined by its name and parameter list (and sometimes its return type, depending on the language)
  Input: #Input parameters of the function, including their types, and explain the meaning of "input".
  Output: #Output parameters of the function, including their types, and explain the meaning of "output".
  Behavior: #Explain step by step the behaviour of a function during its execution. “If a comment in the code indicates that a certain part has bug, after the corresponding behavior, add a parenthesis containing additional information indicating that this behavior has a bug.
]

Output:
#The completed specification
    """

    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{code}"),
        ]
    )

   
    messages = prompt_template.format_messages(code=code)

    
    response = model.invoke(messages)

    spec_text = response.content

    
    usage = response.usage_metadata
    prompt_tokens = usage["input_tokens"]
    completion_tokens = usage["output_tokens"]

    
    cost = (
        prompt_tokens * PRICE_INPUT
        + completion_tokens * PRICE_OUTPUT
    )

    return spec_text, cost
