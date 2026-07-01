"""
generate jeorn project on a joern server
"""
import os

from cpgqls_client import CPGQLSClient

server_endpoint = "127.0.0.1:8081"   # server url
client = CPGQLSClient(server_endpoint)

def get_repos(root_dir, proj_list, id_list):
    repos_dir = root_dir + 'defects4j/'
    for i in range(len(proj_list)):
        project = proj_list[i]
        for j in id_list[i]:
            unique_id = project + '-' + str(j)
            try:
                print("in processing: " + project + '_' + str(j))
                id = './' + unique_id + '_buggy'
                name = unique_id + '_buggy'
                query = f'importCode(inputPath="{id}", projectName="{name}")'
                print(query)
                result = client.execute(query)
                print(result)
            except (RuntimeError, TypeError, NameError, FileNotFoundError) as e:
                print(e)


def get_repos_by_id(proj_id):
    try:
        name = proj_id.split('-')[0]
        id = proj_id.split('-')[1]
        proj_id_buggy = proj_id + '_buggy'
        proj_path = './data/' + name+ '/' + id + 'b'
        print("in processing: " + proj_id)
        query = f'importCode(inputPath="{proj_path}", projectName="{proj_id_buggy}")'
        print(query)
        result = client.execute(query)
        print(result)
    except (RuntimeError, TypeError, NameError, FileNotFoundError) as e:
        print(e)



# root_dir = ''
# d4j_dir = './'


#list = ['']
#
#
#for proj in list:
#    get_repos_by_id(proj)
def get_bug_repo(proj):
    try:
        proj_path = f"./defects4j/{proj}_buggy/"
        project_name = f"{proj}_buggy"
        query = f'importCode(inputPath="{proj_path}", projectName="{project_name}")'
        print("Executing:", query)
        result = client.execute(query)
        print(result)
    except (RuntimeError, TypeError, NameError, FileNotFoundError) as e:
        print(e)




# proj_path = "./defects4j/Closure-101_buggy/"      # defects4j checkout ???
# project_name = "Closure-101_buggy"     # Joern ?? project ??
#
# query = f'importCode(inputPath="{proj_path}", projectName="{project_name}")'
# print("Executing:", query)
# result = client.execute(query)
# print(result)