"""
generate defects4j projects
"""

import os
from get_bugList import bugList


def get_repos(root_dir, proj_list, id_list):
    repos_dir = root_dir + 'defects4j/'
    for i in range(len(proj_list)):
        project = proj_list[i]
        for j in id_list[i]:
            unique_id = project + '_' + str(j)
            try:
                print("in processing: " + project + '_' + str(j))
                cmd = 'defects4j checkout -p ' + project + ' -v ' + str(j) + 'b -w ' + repos_dir + unique_id + '_buggy'
                os.system(cmd)
                print(cmd)
            except (RuntimeError, TypeError, NameError, FileNotFoundError) as e:
                print(e)

def get_repos_by_id(root_dir, proj_id):
    repos_dir = root_dir + 'defects4j/'
    proj = proj_id.split('-')[0]
    print(proj)
    num = proj_id.split('-')[1]
    try:
        print("in processing: " + proj + '-' + str(num))
        cmd = 'defects4j checkout -p ' + proj + ' -v ' + str(num) + 'b -w ' + repos_dir +  proj_id +'_buggy'
        os.system(cmd)
    except (RuntimeError, TypeError, NameError, FileNotFoundError) as e:
        print(e)

root_dir = ''
d4j_dir = './defects4j/'

# list = ['Closure-101']

for proj in bugList:
    get_repos_by_id(root_dir, proj)