import glob
import json
import time
from pickle import FALSE

import javalang
import subprocess
import re
import os
import signal
from utils.extract_d4j_test_case import extract_test_case


def cut_before_segment(path: str, segment: str) -> str:
    
    parts = path.replace("\\", "/").split("/")

    
    try:
        idx = parts.index(segment)
    except ValueError:
        raise ValueError(f"Segment '{segment}' not found in path '{path}'")

    
    return "/".join(parts[idx:])


def run_d4j_test(source, testmethods, bug_id):
    bugg = False
    compile_fail = False
    timed_out = False
    entire_bugg = False
    error_string = ""  
    log = None         

    try:
        tokens = javalang.tokenizer.tokenize(source)
        parser = javalang.parser.Parser(tokens)
        parser.parse()
    except:
        print("Syntax Error")
        
        return compile_fail, timed_out, bugg, entire_bugg, True, None, error_string

    for t in testmethods:
        print(t.strip())
        cmd = 'defects4j test -w %s/ -t %s' % (('/tmp/' + bug_id), t.strip())
        Returncode = ""
        error_file = open("stderr.txt", "wb")
        child = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=error_file,
            bufsize=-1,
            start_new_session=True
        )
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()
                print(b"".join(Returncode).decode('utf-8'))
                error_file.close()
                break
            elif Flag != 0 and Flag is not None:
                compile_fail = True
                error_file.close()
                
                try:
                    with open("stderr.txt", "rb") as f:
                        r = f.readlines()
                    for line in r:
                        decoded = line.decode('utf-8', errors='ignore')
                        if re.search(r':\serror:\s', decoded):
                            error_string = decoded.strip()
                            break
                except Exception:
                    pass
                print(error_string)
                break
            elif time.time() - while_begin > 15:
                error_file.close()
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                timed_out = True
                break
            else:
                time.sleep(0.01)

        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            continue
        else:
            bugg = True
            break

    
    if not bugg:
        print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
        cmd = 'defects4j test -w %s/' % ('/tmp/' + bug_id)
        Returncode = ""
        child = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   
            bufsize=-1,
            start_new_session=True
        )
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()
                break
            elif Flag != 0 and Flag is not None:
                bugg = True
                compile_fail = True
                
                try:
                    stderr_lines = child.stderr.readlines()
                    for line in stderr_lines:
                        decoded = line.decode('utf-8', errors='ignore')
                        if re.search(r':\serror:\s', decoded):
                            error_string = decoded.strip()
                            break
                except Exception:
                    pass
                break
            elif time.time() - while_begin > 180:
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                bugg = True
                timed_out = True
                break
            else:
                time.sleep(0.01)

        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            print('success')
        else:
            entire_bugg = True

    
    return compile_fail, timed_out, bugg, entire_bugg, False, log, error_string


def validate_patch_sf(file=None, patch=None):
    with open("./Datasets/D4J/defects4j-sf.json", "r") as f:
        bug_dict = json.load(f)

    bug_id = file.split('.')[0]
    project = bug_id.split("-")[0]
    bug = bug_id.split("-")[1]

    tmp_bug_id = "test_" + bug_id
    subprocess.run('rm -rf /tmp/' + tmp_bug_id, shell=True)
    subprocess.run(f"defects4j checkout -p {project} -v {bug}b -w /tmp/{tmp_bug_id}", shell=True)

    testmethods = os.popen(f"defects4j export -w /tmp/{tmp_bug_id} -p tests.trigger").readlines()
    source_dir  = os.popen(f"defects4j export -p dir.src.classes -w /tmp/{tmp_bug_id}").read().strip()

    patch_lines = patch.splitlines(keepends=True)

    start = bug_dict[bug_id]["start"]
    end = bug_dict[bug_id]["end"]
    print(start, end)

    loc = bug_dict[bug_id]["loc"]
#    if bug_id.startswith("Closure"):
#        loc = cut_before_segment(loc, "com")
#    else:
#        loc = cut_before_segment(loc, "org")
    if bug_id.startswith("Cli") or bug_id.startswith("Chart") or bug_id.startswith("Codec") or bug_id.startswith("Collections") or bug_id.startswith("Compress") or bug_id.startswith("Csv") or bug_id.startswith("Jsoup") or bug_id.startswith("JxPath") or bug_id.startswith("Lang") or bug_id.startswith("Math") or bug_id.startswith("Mockito") or bug_id.startswith("Time"):
      loc = cut_before_segment(loc, "org")
    
    if bug_id.startswith("Closure") or bug_id.startswith("Gson") or bug_id.startswith("JacksonCore") or bug_id.startswith("JacksonDatabind") or bug_id.startswith("JacksonXml"):
      loc = cut_before_segment(loc, "com")
    
    

    print(loc)

    try:
        with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "r") as f:
            source = f.readlines()
    except:
        with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "r", encoding="ISO-8859-1") as f:
            source = f.readlines()

    new_source = "".join(source[:start - 1] + patch_lines + source[end:])
    try:
        with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "w") as f:
            f.write(new_source)
    except:
        with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "w", encoding="ISO-8859-1") as f:
            f.write(new_source)

    
    compile_fail, timed_out, bugg, entire_bugg, syntax_error, log, error_string = run_d4j_test(
        new_source, testmethods, tmp_bug_id
    )

    subprocess.run('rm -rf /tmp/' + tmp_bug_id, shell=True)

    if not compile_fail and not timed_out and not bugg and not entire_bugg and not syntax_error:
        return True, None
    else:
        if compile_fail:
            
            if error_string:
                message = "Compile Fail: " + error_string
            else:
                message = "Compile Fail"
        elif timed_out:
            message = "Time Out"
        elif syntax_error:
            message = "Syntax Error"
        else:
            
            message = "Failing test: " + log[-1].decode('utf-8')[4:-1]
        return False, message

def validate_patch_mf(file=None, patchs=None):
    with open("./Datasets/D4J/defects4j-mf.json", "r") as f:
        bug_dict = json.load(f)

    bug_id = file.split('.')[0]
    project = bug_id.split("-")[0]
    bug = bug_id.split("-")[1]

    tmp_bug_id = "test_" + bug_id
    subprocess.run('rm -rf /tmp/' + tmp_bug_id, shell=True)
    subprocess.run(f"defects4j checkout -p {project} -v {bug}b -w /tmp/{tmp_bug_id}", shell=True)

    testmethods = os.popen(f"defects4j export -w /tmp/{tmp_bug_id} -p tests.trigger").readlines()
    source_dir  = os.popen(f"defects4j export -p dir.src.classes -w /tmp/{tmp_bug_id}").read().strip()

    

    patchs_detail = []


    for i in range(len(patchs)):
        loc = bug_dict[bug_id]["functions"][i]["path"]
        if bug_id.startswith("Cli") or bug_id.startswith("Chart") or bug_id.startswith("Codec") or bug_id.startswith("Collections") or bug_id.startswith("Compress") or bug_id.startswith("Csv") or bug_id.startswith("Jsoup") or bug_id.startswith("JxPath") or bug_id.startswith("Lang") or bug_id.startswith("Math") or bug_id.startswith("Mockito") or bug_id.startswith("Time"):
            loc = cut_before_segment(loc, "org")

        if bug_id.startswith("Closure") or bug_id.startswith("Gson") or bug_id.startswith("JacksonCore") or bug_id.startswith("JacksonDatabind") or bug_id.startswith("JacksonXml"):
            loc = cut_before_segment(loc, "com")
        # loc = "/".join(loc.split("/")[loc_cut:])
        patch_detail = {
            "patch": patchs[i],
            "start": bug_dict[bug_id]["functions"][i]['start_loc'],
            "end" : bug_dict[bug_id]["functions"][i]['end_loc'],
            "loc" : loc
        }

        patchs_detail.append(patch_detail)

    patchs_detail.sort(key=lambda x: x["start"], reverse=True)


    for i in range(len(patchs)):
        patch = patchs_detail[i]["patch"]
        patch_lines = patch.splitlines(keepends=True)

        start = patchs_detail[i]["start"]
        end = patchs_detail[i]["end"]
        print(start, end)

        loc = patchs_detail[i]["loc"]

        print(loc)

        try:
            with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "r") as f:
                source = f.readlines()
        except:
            with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "r", encoding="ISO-8859-1") as f:
                source = f.readlines()

        new_source = "".join(source[:start - 1] + patch_lines + source[end:])
        try:
            with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "w") as f:
                f.write(new_source)
        except:
            with open(f"/tmp/{tmp_bug_id}/{source_dir}/{loc}", "w", encoding="ISO-8859-1") as f:
                f.write(new_source)


    compile_fail, timed_out, bugg, entire_bugg, syntax_error, log, error_string = run_d4j_test(
        new_source, testmethods, tmp_bug_id
    )

    subprocess.run('rm -rf /tmp/' + tmp_bug_id, shell=True)



    if not compile_fail and not timed_out and not bugg and not entire_bugg and not syntax_error:
        return True, None
    else:
        if compile_fail:
            
            if error_string:
                message = "Compile Fail: " + error_string
            else:
                message = "Compile Fail"
        elif timed_out:
            message = "Time Out"
        elif syntax_error:
            message = "Syntax Error"
        else:
            
            message = "Failing test: " + log[-1].decode('utf-8')[4:-1]
        return False, message
