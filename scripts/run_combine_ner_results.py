import os
import spacy
import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Combine all NER results for each deed document.")
    parser.add_argument('--file_path', type=str, default="./output/all_geo_parcel_w_txt_n_image_ids.jsonl",
                        help='load jsonl from this path')

    parser.add_argument('--output_path', type=str, default="./output/combine_ner_results.jsonl",
                        help='Save the combined model results to this path as jsonl file')

    return parser.parse_args()

def merge_dictionaries(dictionaries):
    """
    combine dicts with same image_ids
    
    param:
    dictionaries -- a list of dictionaries, each dictionary has the same structure
    
    return:
    combined dictionaries with same image_ids
    """
    # to store grouped dictionaries
    grouped_data = {}
    
    # group dictionaties by image_ids
    for item in dictionaries:
        # convert to tuple to use image_ids as keys for dictionaries
        image_key = tuple(item['image_ids'])
        
        # if image_ids exist in the group, merge
        if image_key in grouped_data:
            existing_item = grouped_data[image_key]
            
            # merge all attributes except text
            for key in existing_item:
                if key != 'text' and key != 'image_ids' and key in item and item[key]:
                    existing_item[key] = existing_item[key] + item[key]
        # if image_ids does not exist, create new group
        else:
            # copy the current dictionary as the basis of find other items that should be belonging to its group
            grouped_data[image_key] = item.copy()
    
    # convert to list and return the grouped data
    return list(grouped_data.values())

def load_test_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    
    args = parse_args()

    data = load_test_data(args.file_path)

    merged_result = merge_dictionaries(data)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    file = open(args.output_path, 'w')

    for item in merged_result:
        # if len(item['image_ids']) > 1:
        # print(f"image_ids: {item['image_ids']}")
        temp = {"image_ids": item['image_ids']}
        for key in item:
            if key != 'text' and key != 'image_ids':
                # print(f"  {key}: {item[key]}")
                temp[key] = item[key]
        # print("------------------------")

        # json_line = json.dumps({"image_ids": item['image_ids'], "NERpredicted_STATE": kv['STATE'], "NERpredicted_CNTY": kv['COUNTY'], "NERpredicted_CTY": kv['CITY']}, ensure_ascii=False)
        json_line = json.dumps(temp, ensure_ascii=False)
        file.write(json_line + '\n')

    file.close()
    print(f"Combined all predicted entities from NER and saved to {args.output_path}")

if __name__ == "__main__":

    main()
