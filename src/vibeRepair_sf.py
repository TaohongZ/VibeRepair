import os
from utils.parse_d4j import clean_parse_d4j
import argparse
from utils.build_d4j import build_d4j1_2
import json
import time
from utils.validate_d4j import validate_patch_sf
from utils.extract_d4j_test_case import extract_test_case, get_failing_test_trace
from utils.code_specification import code_gen_spec
from utils.fix_specification import SpecAligner_sf
from utils.fix_code_gen import spec_gen_code_sf,extract_fix_code

os.environ["OPENAI_API_KEY"] = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4JV1.2",
                        help="Dataset to use, current support: D4JV1.2, D4JV2.0")
    args = parser.parse_args()

    with open("./Datasets/D4J/defects4j-sf.json", "r") as f:
        d4j_sf_dataset = json.load(f)

    d4j1_2 = build_d4j1_2()

    if args.dataset == "D4JV1.2":
        inference_dataset = {
            key: value
            for key, value in d4j_sf_dataset.items()
            if key.split('.')[0] in d4j1_2
        }
    elif args.dataset == "D4JV2.0":
        inference_dataset = {
            key: value
            for key, value in d4j_sf_dataset.items()
            if key.split('.')[0] not in d4j1_2
        }
    else:
        inference_dataset = d4j_sf_dataset

    # 1. 创建 / 读取 JSON 结果文件
    results_path = "./d4jv2.0_sf_results.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as rf:
                results_data = json.load(rf)
            if not isinstance(results_data, list):
                results_data = []
        except Exception:
            results_data = []
    else:
        results_data = []

    fix_count = 0
    count = 0
    for file, bug in inference_dataset.items():
        

        valid = False
        n = 0

        last_error_message = None  

        
        patch = ""
        flawed_specification = ""
        fix_specification = ""

        file_start_time = time.perf_counter()

        while not valid:
            
            if n > 4:
                
                result_record = {
                    "file": file,
                    "patch": patch,   
                    "valid": False,
                    "flaw_speci": flawed_specification,
                    "fix_speci": fix_specification
                }
                
                if last_error_message is not None:
                    result_record["error"] = last_error_message

                results_data.append(result_record)
                with open(results_path, "w", encoding="utf-8") as wf:
                    json.dump(results_data, wf, ensure_ascii=False, indent=2)
                print(f"[{file}] Tried 5 outer attempts, still invalid, recorded to {results_path}")
                break

            try:
                
                last_error_message = None  
                cost = 0

                print("Testing {} ... ".format(file.split(".")[0]))
                print("===============Generating Flawed Specification=============")

                buggy_code = bug["buggy_fl"]
                flawed_specification, code_spec_cost = code_gen_spec(buggy_code)

                cost += code_spec_cost


                trigger_tests = []
                for key, test_case in bug["trigger_test"].items():
                    test_detail = [test_case["src"], test_case["clean_error_msg"]]
                    trigger_tests.append(test_detail)

                specAligner = SpecAligner_sf()

                print("===============Fixing Bad Specification (first_align)=============")
                if bug["issue_description"] == "":
                    issue_des = "None"
                else:
                    issue_des = bug["issue_description"]
                fix_specification,  fix_speci_cost= specAligner.first_align(
                    flawed_specification,
                    trigger_tests,
                    issue_des
                )
                
                print(fix_specification)

                cost += fix_speci_cost

                
                def generate_and_validate(spec):
                    """根据给定 spec 生成代码、提取 patch、验证并返回 (valid, message, patchs)"""
                    print("===============Generating code based on specification=============")
                    fix_code_local, code_gen_cost = spec_gen_code_sf(spec, buggy_code)

                    patch = extract_fix_code(fix_code_local)

                    v, msg = validate_patch_sf(file, patch)
                    print(v, msg)
                    return v, msg, patch, code_gen_cost

                
                valid, message, patch, code_gen_cost = generate_and_validate(fix_specification)
                
                print(message)

                cost += code_gen_cost

                
                additional_try = 0
                while not valid and additional_try < 3:
                    print(f"=== additional_info attempt {additional_try + 1} for {file} ===")
                    fix_specification, fix_speci_cost = specAligner.additional_info(message)
                    cost += fix_speci_cost
                    valid, message, patch, code_gen_cost = generate_and_validate(fix_specification)
                    cost += code_gen_cost
                    additional_try += 1

                
                if valid:
                    
                    time_cost_sec = time.perf_counter() - file_start_time
                    results_data.append({
                        "file": file,
                        "patch": patch,
                        "valid": True,
                        "time_cost_sec": time_cost_sec,
                        "cost": cost,
                        "flaw_speci": flawed_specification,
                        "fix_speci": fix_specification
                    })
                    with open(results_path, "w", encoding="utf-8") as wf:
                        json.dump(results_data, wf, ensure_ascii=False, indent=2)
                    print(f"[{file}] Patch valid, recorded to {results_path}")
                    break  

                

            except Exception as e:
                
                last_error_message = str(e)  
                print(f"[{file}] Error at outer attempt {n + 1}: {e}")

            
            n += 1


if __name__=="__main__":
    main()

