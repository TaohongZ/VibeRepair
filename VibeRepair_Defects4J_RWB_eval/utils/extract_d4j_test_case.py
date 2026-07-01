import os
from pathlib import Path
import re


def extract_test_case(test_identifier: str, project_root: str) -> str:
   

   
    if "::" not in test_identifier:
        raise ValueError("test_identifier must include '::' ")

    class_fqn, method_name = test_identifier.split("::")

    
    class_path = class_fqn.replace(".", "/") + ".java"

    
    java_file = None
    for root, dirs, files in os.walk(project_root):
        if files and class_path.split("/")[-1] in files:
            candidate = Path(root) / class_path.split("/")[-1]
           
            if str(candidate).replace("\\", "/").endswith(class_path):
                java_file = candidate
                break

    if java_file is None:
        raise FileNotFoundError(f"未找到对应的测试类文件: {class_path}")

    
    src = java_file.read_text(encoding="utf-8")

    
    pattern = re.compile(
        r"(?:public|protected|private)\s+void\s+"
        + re.escape(method_name)
        + r"\s*\([^)]*\)\s*\{",
        re.MULTILINE
    )

    m = pattern.search(src)
    if not m:
        raise ValueError(f"can't find {method_name}")

    start = m.start()

    
    brace_pos = src.find("{", m.start())
    if brace_pos == -1:
        raise ValueError("error")

    brace_count = 0
    end = brace_pos
    for i in range(brace_pos, len(src)):
        if src[i] == "{":
            brace_count += 1
        elif src[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    method_src = src[start:end]
    return method_src

def get_failing_test_trace(project_root: str, n_lines: int = 10) -> str:
    

    project_root = Path(project_root)

    
    failing_tests_path = project_root / "failing_tests"

    if not failing_tests_path.exists():
        raise FileNotFoundError(f" can't find failing_tests : {failing_tests_path}")

    lines = failing_tests_path.read_text(encoding="utf-8").splitlines()

    head = "\n".join(lines[:n_lines])
    return head
