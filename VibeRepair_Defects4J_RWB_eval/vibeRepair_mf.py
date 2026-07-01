import os
from utils.parse_d4j import clean_parse_d4j
import argparse
from utils.build_d4j import build_d4j1_2
import json
import time
from utils.validate_d4j import validate_patch_mf
from utils.extract_d4j_test_case import extract_test_case, get_failing_test_trace
from utils.code_specification import code_gen_spec
from utils.fix_specification import SpecAligner
from utils.fix_code_gen import spec_gen_code,extract_fix_code_regex

os.environ["OPENAI_API_KEY"] = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4JV1.2",
                        help="Dataset to use, current support: D4JV1.2, D4JV2.0")
    args = parser.parse_args()

    with open("./Datasets/D4J/defects4j-mf.json", "r") as f:
        d4j_mf_dataset = json.load(f)

    d4j1_2 = build_d4j1_2()

    if args.dataset == "D4JV1.2":
        inference_dataset = {
            key: value
            for key, value in d4j_mf_dataset.items()
            if key.split('.')[0] in d4j1_2
        }
    elif args.dataset == "D4JV2.0":
        inference_dataset = {
            key: value
            for key, value in d4j_mf_dataset.items()
            if key.split('.')[0] not in d4j1_2
        }
    else:
        inference_dataset = d4j_mf_dataset

    
    results_path = "./d4jv2.0_mf_results.json"
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
        patchs = []
        last_error_message = None  

       
        flawed_specification = []
        fix_specification = ""

        file_start_time = time.perf_counter()

        while not valid:
            
            if n > 4:
                
                result_record = {
                    "file": file,
                    "patch": patchs,  
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
                flawed_specification = []
                buggy_codes = []
                for function_detail in bug["functions"]:
                    buggy_code = function_detail["buggy_fl"]
                    buggy_codes.append(buggy_code)
                    specification, code_spec_cost = code_gen_spec(buggy_code)
                    flawed_specification.append(specification)
                    cost += code_spec_cost

                trigger_tests = []
                for key, test_case in bug["trigger_test"].items():
                    test_detail = [test_case["src"], test_case["clean_error_msg"]]
                    trigger_tests.append(test_detail)

                specAligner = SpecAligner()

                print("===============Fixing Bad Specification (first_align)=============")
                fix_specification, fix_speci_cost = specAligner.first_align(
                    flawed_specification,
                    trigger_tests
                )

                cost += fix_speci_cost

               
                def generate_and_validate(spec):
                    
                    print("===============Generating code based on specification=============")
                    fix_code_local, code_gen_cost = spec_gen_code(spec, buggy_codes)

                    local_patches = []
                    for i in range(len(buggy_codes)):
                        patch = extract_fix_code_regex(fix_code_local, i + 1)
                        print(patch)
                        local_patches.append(patch)

                    v, msg = validate_patch_mf(file, local_patches)
                    print(v, msg)
                    return v, msg, local_patches, code_gen_cost

                
                valid, message, patchs, code_gen_cost = generate_and_validate(fix_specification)

                cost += code_gen_cost

                
                additional_try = 0
                while not valid and additional_try < 3:
                    print(f"=== additional_info attempt {additional_try + 1} for {file} ===")
                    fix_specification, fix_speci_cost = specAligner.additional_info(message)
                    cost += fix_speci_cost
                    valid, message, patchs, code_gen_cost = generate_and_validate(fix_specification)
                    cost += code_gen_cost
                    additional_try += 1

               
                if valid:
                   
                    time_cost_sec = time.perf_counter() - file_start_time
                    results_data.append({
                        "file": file,
                        "patch": patchs,
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

