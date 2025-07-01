import os
import spacy
import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Run the state/county/city NER model to predicted the subdivision names from the sentences.")
    parser.add_argument('--file_path', type=str, default="./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl",
                        help='load jsonl from this path')
    
    parser.add_argument('--model_aut_path', type=str, default="./src/models/state_county_city_ner_model/model-best",
                        help='To identify state/county/city and parcel information (lot/block/unit/township/range/section/quarter). Load model from this path')
    
    parser.add_argument('--model_subd_path', type=str, default="./src/models/subdivision_ner_model/model-best",
                        help='load trained NER model for subdivision from this path')

    parser.add_argument('--output_path', type=str, default="./output/all_geo_parcel_w_txt_n_image_ids.jsonl",
                        help='Save model recognition results to this path as jsonl file')

    return parser.parse_args()

def extract_state_cnty_cty_kv(doc):
    result = {"STATE":[], "COUNTY": [], "CITY": [], "LOT": [], "BLOCK": [], "UNIT": [], "TOWNSHIP": [], "RANGE": [], "SECTION": [], "QUARTER": []}
    for ent in doc.ents:
        # print(ent.label_, ent.text)
        # print(ent.start_char, ent.end_char)
        label = ent.label_
        text = ent.text.strip()
        if label in result:
            if isinstance(result[label], list):
                result[label].append((text, ent.start_char, ent.end_char))
            else:
                result[label] = (text, ent.start_char, ent.end_char)
    for key in result:
        temp = []
        for i in range(len(result[key])):
            if len(temp) != 0 and result[key][i][1] == temp[-1][2]+1:
                temp[-1] = (temp[-1][0] + " " + result[key][i][0], temp[-1][1], result[key][i][2])
            elif len(temp) != 0 and result[key][i][1] == temp[-1][2]:
                temp[-1] = (temp[-1][0] + result[key][i][0], temp[-1][1], result[key][i][2])
            else:
                temp.append((result[key][i][0], result[key][i][1], result[key][i][2]))
        result[key] = [entity[0] for entity in temp]
    return result

def extract_subd_kv(doc):
    result = {"SUBDIVISION": []}
    for ent in doc.ents:
        # print(ent.label_, ent.text)
        # print(ent.start_char, ent.end_char)
        label = ent.label_
        text = ent.text.strip()
        if label in result:
            if isinstance(result[label], list):
                result[label].append(text)
            else:
                result[label] = text
    return result

def load_test_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    
    args = parse_args()

    nlp1 = spacy.load(args.model_aut_path)
    nlp2 = spacy.load(args.model_subd_path)

    data = load_test_data(args.file_path)

    result = defaultdict(list)
    for item in data:
        key = tuple(item["image_ids"])
        result[key].append(item["text"])

    result_list = [
        {"image_ids": sorted(list(img_ids)), "sentences": sentences}
        for img_ids, sentences in result.items()
    ]
    # print(len(result_list))
    # print(type(result_list))
    # print(result_list[:1])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    file = open(args.output_path, 'w')

    # for item in result_list[10:20]:
    for item in result_list:
        # print("=====================================")
        # print(f"Image IDs: {item['image_ids']}")
        # print("Number of sentences: ", len(item['sentences']), )
        for text in item['sentences']:
            doc1 = nlp1(text)
            doc2 = nlp2(text)
            kv = extract_state_cnty_cty_kv(doc1)
            kv_subd = extract_subd_kv(doc2)
            # print(text)
            # print(kv)
            json_line = json.dumps({"text": text, "image_ids": item['image_ids'], "NERpredicted_STATE": kv['STATE'], "NERpredicted_CNTY": kv['COUNTY'], "NERpredicted_CTY": kv['CITY'], "NERpredicted_SUBD": kv_subd['SUBDIVISION'], "NERpredicted_LOT": kv['LOT'], "NERpredicted_BLOCK": kv['BLOCK'], "NERpredicted_UNIT": kv['UNIT'], "NERpredicted_TOWNSHIP": kv['TOWNSHIP'], "NERpredicted_RANGE": kv['RANGE'], "NERpredicted_SECTION": kv['SECTION'], "NERpredicted_QUARTER": kv['QUARTER']}, ensure_ascii=False)
            file.write(json_line + '\n')
    
    file.close()
    print(f"Predicted all with its image_ids and original sentence saved to {args.output_path}")


if __name__ == "__main__":

    main()
