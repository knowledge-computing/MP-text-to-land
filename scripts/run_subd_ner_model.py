import os
import spacy
import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Run the subdivision NER model to predicted the subdivision names from the sentences.")
    parser.add_argument('--file_path', type=str, default="./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl",
                        help='load jsonl from this path')
    
    parser.add_argument('--model_path', type=str, default="./src/models/subdivision_ner_model/model-best",
                        help='load trained NER model for subdivision from this path')

    parser.add_argument('--output_path', type=str, default="./output/subd_w_txt_n_image_ids.jsonl",
                        help='Save model recognition results to this path as jsonl file')

    return parser.parse_args()

def extract_kv(doc):
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

    nlp = spacy.load(args.model_path)

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
            doc = nlp(text)
            kv = extract_kv(doc)
            # print(text)
            # print(kv)
            json_line = json.dumps({"text": text, "image_ids": item['image_ids'], "NERpredicted_SUBD": kv['SUBDIVISION']}, ensure_ascii=False)
            file.write(json_line + '\n')
    
    file.close()
    print(f"Predicted subdivisions with its image_ids and original sentence saved to {args.output_path}")


if __name__ == "__main__":

    main()
