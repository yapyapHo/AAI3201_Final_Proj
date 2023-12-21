import json
f = open('handwriting_data_info_clean.json', 'r', encoding='UTF-8')

json_data = json.load(f)

annotation_list = json_data['annotations']

new_dict = dict()

for annotation in annotation_list:
    if annotation['attributes']['type'] == '글자(음절)':
        new_dict[annotation['image_id']] = annotation['text']

print(json.dumps(new_dict, indent=4))

with open('handwriting.json', 'w', encoding='UTF-8', ) as f :
    json.dump(new_dict, f, indent=4, ensure_ascii=False)