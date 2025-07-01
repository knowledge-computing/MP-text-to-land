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
    合并具有相同image_ids的字典数据
    
    参数:
    dictionaries -- 包含多个字典的列表，每个字典具有相同的结构
    
    返回:
    合并后的字典列表
    """
    # 用于存储分组后的数据
    grouped_data = {}
    
    # 按image_ids对字典进行分组
    for item in dictionaries:
        # 将image_ids转换为元组以便作为字典键
        image_key = tuple(item['image_ids'])
        
        # 如果image_ids已存在于分组中，合并数据
        if image_key in grouped_data:
            existing_item = grouped_data[image_key]
            
            # 合并除text外的所有属性
            for key in existing_item:
                if key != 'text' and key != 'image_ids' and key in item and item[key]:
                    existing_item[key] = existing_item[key] + item[key]
        # 如果image_ids不存在，创建新的分组
        else:
            # 复制当前字典作为新分组的基础
            grouped_data[image_key] = item.copy()
    
    # 将分组后的数据转换为列表并返回
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

    # result = defaultdict(list)
    # for item in data:
    #     key = tuple(item["image_ids"])
    #     result[key].append(item["text"])

    # result_list = [
    #     {"image_ids": sorted(list(img_ids)), "sentences": sentences}
    #     for img_ids, sentences in result.items()
    # ]
    # print(len(result_list))
    # print(type(result_list))
    # print(result_list[:1])



    # for item in result_list[10:20]:
    # for item in result_list:
        # print("=====================================")
        # print(f"Image IDs: {item['image_ids']}")
        # print("Number of sentences: ", len(item['sentences']), )
        # for text in item['sentences']:
            # doc = nlp(text)
            # kv = extract_state_cnty_cty_kv(doc)
            # print(text)
            # print(kv)

    
    # file.close()
    # print(f"Predicted state/county/city with its image_ids and original sentence saved to {args.output_path}")


if __name__ == "__main__":

    main()
