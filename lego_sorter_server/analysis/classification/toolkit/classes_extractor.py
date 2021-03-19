import json

classes = ["3710", "2780", "663626", "6558", "379523", "2654", "3031", "32523", "3832", "32526", "3701", "32001",
           "30145", "370826", "19220", "2569",
           "32073", "3673", "3737", "370626", "6562", "32062", "32002", "6111", "61252", "15712", "3895", "3709",
           "87079", "30374", "30383",
           "2540", "3710", "6111", "4032", "30208", "329826", "4092", "30303", "30459", "30497", "30645", "32019",
           "32059", "32218", "89648"]

print(len(classes))

with open("parts.json", "r") as src:
    parts = json.load(src)

results = []
for part in parts["parts"]:
    if part["base_file_name"] in classes:
        results.append(part)
        classes.remove(part["base_file_name"])

print(len(results))
print(classes)

with open("out.json", "w") as target:
    target.write(json.dumps({"parts": results}))
