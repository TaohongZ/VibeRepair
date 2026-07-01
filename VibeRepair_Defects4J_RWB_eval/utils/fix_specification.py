from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import re


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import math

from langchain_openai import ChatOpenAI

class SpecAligner_sf:
    

    PRICE_INPUT = 5.0 / 1_000_000
    PRICE_OUTPUT = 15.0 / 1_000_000

    def __init__(self, model_name: str = "gpt-4o-2024-05-13", temperature: float = 1.0):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)

        self.system_message_text = (
            '''
        You are a code analysis expert. 
        The user will provide you with a flawed specifications. The flawed specification represent a function and is meant to guide engineers during development.
        The code developed based on the flawed specification has failed some test cases. The content of the failed test cases and their corresponding error messages will also be provided to you.
      
        Your output should be a step-by-step analysis:
      
          1. Based on the failing test cases and the error messages, infer the correct intended behavior for the function.
          2. According to the correct intention and the parts of the flawed specification that indicate bugs, analyze the root cause and propose how to fix it.
          3. Based on the above analysis, produce the corrected specification for flawed specification.
            '''
        )

        
        self.messages = []

    def _calc_cost(self, response) -> float:
        
        usage = response.usage_metadata
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        return (
            input_tokens * self.PRICE_INPUT
            + output_tokens * self.PRICE_OUTPUT
        )

    def first_align(self, flawed_specification, trigger_tests, issue_des):
        
        system_msg = SystemMessage(content=self.system_message_text)

        specification_parts = f"The flawed specification: {flawed_specification}\n"
        issue_description = f"Issue Description: {issue_des}\n"

        tests_parts = [
            f"The failing test case and error message {i + 1} : \n"
            f"  Test case:\n    {test_case[0]}\n"
            f"    Error Message:\n    {test_case[1]} \n\n"
            for i, test_case in enumerate(trigger_tests)
        ]

        human_content = specification_parts + issue_description + "".join(tests_parts)
        human_msg = HumanMessage(content=human_content)

       
        self.messages = [system_msg, human_msg]

        response = self.model.invoke(self.messages)

        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)

        cost = self._calc_cost(response)

        return response.content, cost

    def additional_info(self, vali_res):
        
        human_content = f'''The revised specification still contain errors, and the code developed based on them still fails the tests. 
        Error message: {vali_res}

        Redo the step-by-step analysis and produce new corrected specification.
        '''
        
        human_msg = HumanMessage(content=human_content)
        self.messages.append(human_msg)

        response = self.model.invoke(self.messages)

        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)

        cost = self._calc_cost(response)

        return response.content, cost

    def get_history(self):
        
        return self.messages




class SpecAligner:
    

    PRICE_INPUT = 5.0 / 1_000_000
    PRICE_OUTPUT = 15.0 / 1_000_000

    def __init__(self, model_name: str = "gpt-4o-2024-05-13", temperature: float = 1.0):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)

        self.system_message_text = (
            '''
  You are a code analysis expert. 
  The user will provide you with several flawed specifications. These flawed specifications represent different functions within the same project and are meant to guide engineers during development.
  The code developed based on these flawed specifications has failed some test cases. The content of the failed test cases and their corresponding error messages will also be provided to you.
  
  Your output should be a step-by-step analysis:

    1. Based on the failing test cases and the error messages, infer the correct intended behavior for each function.
    2. According to the correct intention and the parts of the flawed specifications that indicate bugs, analyze the root cause and propose how to fix it.
    3. Based on the above analysis, produce the corrected specification for each flawed specification in order.
      '''
        )

        
        self.messages = []

    def _calc_cost(self, response) -> float:
        
        usage = response.usage_metadata
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        return (
            input_tokens * self.PRICE_INPUT
            + output_tokens * self.PRICE_OUTPUT
        )

    def first_align(self, flawed_specification, trigger_tests):
        
        system_msg = SystemMessage(content=self.system_message_text)

        specification_parts = [
            f"The flawed specification {i + 1}: {spec}\n"
            for i, spec in enumerate(flawed_specification)
        ]

        tests_parts = [
            f"The failing test case and error message {i + 1} : \n"
            f"  Test case:\n    {test_case[0]}\n"
            f"    Error Message:\n    {test_case[1]} \n\n"
            for i, test_case in enumerate(trigger_tests)
        ]

        human_content = "".join([
            "".join(specification_parts),
            "\n",
            "".join(tests_parts)
        ])

        human_msg = HumanMessage(content=human_content)

        
        self.messages = [system_msg, human_msg]

        response = self.model.invoke(self.messages)

        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)

        cost = self._calc_cost(response)

        return response.content, cost

    def additional_info(self, vali_res):
        
        human_content = f'''The revised specifications still contain errors, and the code developed based on them still fails the tests.
        Error message: {vali_res}

        Redo the step-by-step analysis and produce new corrected specifications.
        '''
        human_msg = HumanMessage(content=human_content)
        self.messages.append(human_msg)

        response = self.model.invoke(self.messages)

        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)

        cost = self._calc_cost(response)

        return response.content, cost

    def get_history(self):
        
        return self.messages
