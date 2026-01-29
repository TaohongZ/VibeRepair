import json


with open('./d4jv2.0_sf_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)



bugList = []

tested_bug = []

for value in data:
  if value["valid"] == False:
      bugList.append(value["file"])



#bugList = [b for b in bug if b not in tested_bug]
#bugList = ["Closure-101"]
#print(len(bug))
#print(len(tested_bug))
#
#print(len(bugList))


# with open('.sf_results_with_review.jsonl', 'r', encoding='utf-8') as f:
#     sf_review_results = json.load(f)
#
# for item in sf_review_results:
#     if item["is_correct"] == False:
