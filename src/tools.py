import tiktoken
from cpgqls_client import CPGQLSClient
import requests
from langchain_classic.tools import StructuredTool
from datetime import datetime

#from src.config import JOERN_ADDR

server_endpoint = "127.0.0.1:8081"
client = CPGQLSClient(server_endpoint)
proj_name_gl = ""


def open_proj(proj_name: str) -> str:
    """
    Open the current analysing project by proj_name.

    @param proj_name: The name of project to open. "
    @return: The result of the request execution.

    """

    if not proj_name.endswith("\n"):
        print(proj_name)
    proj_name = proj_name.strip()
    query = f'open("{proj_name}")'
    print(query)
    result = client.execute(query)
    global proj_name_gl
    proj_name_gl = proj_name
    return result['stdout']


def identify_variable(input_string: str) -> str:
    """
    Identifies variable definitions and usages for a given variable name.

    @param input_string: A string containing both the variable name and file name,
                         separated by a comma. Format: variable_name, file_name
    @return: The result of the query execution.
    """
    variable_name, file_name = input_string.split(',')
    variable_name = variable_name.strip()
    file_name = file_name.strip()
    query = f'cpg.identifier.name("{variable_name}").filter(_.location.filename.matches(".*{file_name}.*")).map(n=>List(n.code, n.typeFullName, n.start.dump)).take(3).l'
    print(query)
    result = client.execute(query)
    return result['stdout']


def trace_method_usage(method_name: str) -> str:
    """
    Traces the usage of a specific method in the code.

    @param method_name: The name of the method to trace.
    @return: The result of the query execution.
    """
    query = f'cpg.call.name("{method_name}").map(n=>List(n.code, n.methodFullName)).l'
    print(query)
    result = client.execute(query)
    return slim_joern_token(result['stdout'])


def find_method_in_file(input_string: str) -> str:
    """
    Finds a specific method within a given file and retrieves its location and the whole method code. DO NOT Add any quota marks(") to the input parameter because the tool call will process it.

    @param input_string: A string containing both the method name and file name, separated by a comma. method_name: The name of the method to find. file_name: The name of the file to search within (only the file name, not the file path). Format: method_name, file_name.
    @return: The result of the query execution.

    input_string Example: visit, CheckSideEffects.java
    """
    method_name, file_name = input_string.split(',')
    method_name = method_name.strip()
    file_name = file_name.strip()
    query = (
        f'cpg.method.name("{method_name}")'
        f'.filter(_.location.filename.matches(".*{file_name}"))'
        f'.map(m => (m.location.filename, m.start.dump))'
        f'.l'
    )
    print(query)
    result = client.execute(query)
    return result['stdout']


def analyze_method_details(input_string: str) -> str:
    """
    Identifies various aspects of a given method in a file. DO NOT Add any quota marks(") to the input parameter because the tool call will process it.

    @param input_string: A string containing both the method name and file name, separated by a comma. method_name: The name of the method to find. file_name: The name of the file to search within (only the file name, not the file path). Format: method_name, file_name. Example: visit, CheckSideEffects.java

    @return: The result of the query execution, including:
        - Method name and parameter types
        - Filename where the method is located
        - All throw statements within the method
        - All if statement conditions within the method
        - All method calls within the method
        - All variable assignments within the method
        - The return statement of the method
    """
    input_string = input_string.replace('"', '')
    method_name, file_name = input_string.split(',')
    method_name = method_name.strip()
    file_name = file_name.strip()
    query = f"""
    cpg.method("{method_name}").filter(_.location.filename.matches(".*{file_name}")).map{{ method =>
      (
        method.name,
        method.parameter.map(_.typeFullName).mkString(", "),
        method.location.filename,
        method.ast.isCall.name("throw.*").code.l,
        method.ast.isControlStructure.isIf.condition.code.l,
        method.ast.isCall.name.l,
        method.ast.isCall.name("<operator>.assignment").code.l,
        method.ast.isReturn.code.l
      )
    }}.l
    """
    print(query)
    result = client.execute(query)
    return result['stdout']

def find_class_loc(input_string: str) -> str:
    """
    Finds the location of a class with a given name.

    @param input_string: The name of the class to locate
    @return: A string containing the location/full type name of the class
    """
    class_name = input_string.strip().replace('"', '')

    query = (
        f'cpg.identifier.name("{class_name}").typeFullName.take(1).l'
    )
    print(query)
    result = client.execute(query)
    return result['stdout']

def identify_class(input_string: str) -> str:
    """
    Identifies the class code matches the given search string for a class name.

    @param input_string: The name of the class
    @return: A string containing the class codes
    """

    class_name = input_string.strip().replace('"', '')

    query = (
        f'cpg.typeDecl.fullName(".*{class_name}.*").astChildren.code.dedup.l'
    )
    print(query)
    result = client.execute(query)
    return result['stdout']



def analyze_method_control_flow(method_name: str) -> str:
    """
    Analyzes the control flow (if, while, for, etc.) structures within a specific method.

    @param method_name: The name of the method to analyze.
    @return: The code of the control flow structures within the method.
    """
    query = f'cpg.method.name("{method_name}").ast.isControlStructure.code.l'
    print(query)
    result = client.execute(query)
    return result['stdout']


def get_imports(file_name: str) -> str:
    """
    Retrieves all import statements from a specific file.

    @param file_name: The name of the file to search within (only the file name, not the file path).
    @return: The code of all import statements within the file.
    """
    query = (
        f'cpg.imports'
        f'.filter(_.location.filename.matches(".*{file_name}"))'
        f'.map(m => (m.code))'
        f'.l'
    )
    print(query)
    result = client.execute(query)
    return result['stdout']

def close_proj(proj_name: str) -> str:
    """
    Close the current analysing project by proj_name.

    @param proj_name: The name of project to close. Example format: "Chart-1_buggy"
    @return: The result of the request execution.

    Example: close_proj("Chart-1_buggy")
    """
    if proj_name.endswith("\n"):
        proj_name.replace("\n", "")
    if not proj_name.endswith("\n"):
        print(proj_name)

    query = f'close("{proj_name}")'
    print(query)
    result = client.execute(query)
    return result['stdout']


def example_patch_search(input_string: str) -> str:
    """
    Search for the most similar (by similarity rate) fix pattern example in a Knowledge Base and return it.

    @param input_string: Buggy code concatenate with Root cause. input example:  public boolean equals(Object obj) \n\n    if (obj == this) \n   return true, Root Cause: The `equals` method in the `ShapeList` class is calling `super.equals(obj)` which likely only checks the reference equality or fields defined in the superclass, rather than checking the properties specific to `ShapeList`. This would lead to instances that may contain the same shapes being considered unequal if the parent class does not override `equals` appropriately for deep equality checks.
    @return: The most similar fix pattern, including BuggyCode, FixedCode, RootCause

    """
    url = "http://localhost:5000/search"
    data = {
        "query": input_string,
        "n": 1,
        "threshold": 0.6
    }
    start_time = datetime.now()
    response = requests.post(url, json=data)
    # print(response.json())
    end_time = datetime.now()
    time_difference = end_time - start_time
    print(f"Time cost for retrieving: {time_difference}")
    close_proj(proj_name=proj_name_gl)
    if "'similarity'" in str(response.json()):
        print("example found")
        with open("./output/searched_example.txt", "a", encoding="utf-8") as f:
            f.write(proj_name_gl + "\n")

    return response.json()

def slim_joern_token(joern_response):
    token_upper_limit = 10000
    response_list = joern_response.split('\n')
    slim_res_list = []
    for line in response_list:
        curr_line_token_cnt = num_tokens_from_string(line)
        token_upper_limit -= curr_line_token_cnt
        if token_upper_limit <= 0:
            break
        slim_res_list.append(line)
    return '\n'.join(slim_res_list)


def num_tokens_from_string(string: str) -> int:
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Define the tools
open_proj_tool = StructuredTool.from_function(open_proj)
identify_variable_tool = StructuredTool.from_function(identify_variable)
trace_method_usage_tool = StructuredTool.from_function(trace_method_usage)
find_method_in_file_tool = StructuredTool.from_function(find_method_in_file)
analyze_method_details_tool = StructuredTool.from_function(analyze_method_details)
analyze_method_control_flow_tool = StructuredTool.from_function(analyze_method_control_flow)
find_class_loc_tool = StructuredTool.from_function(find_class_loc)
identify_class_tool = StructuredTool.from_function(identify_class)
get_imports_tool = StructuredTool.from_function(get_imports)
close_proj_tool = StructuredTool.from_function(close_proj)
example_patch_search_tool = StructuredTool.from_function(example_patch_search)


