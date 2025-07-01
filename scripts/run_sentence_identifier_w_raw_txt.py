"""
This script is the geosentence_identifier.
It saves the filtered sentences from keyword fuzzy match
"""

import os
import re
import spacy
import argparse
from pathlib import Path
from src.data_preprocessing.sentence_identifier import filter_relevant_sentences
from src.utils.config_loader import load_config

geojson_path, folder_names, ocrtxt_path, _ = load_config()

def parse_args():
    parser = argparse.ArgumentParser(description="Test geo sentence identifier")
    parser.add_argument('--root_path', type=str, default="/.../covenants-deed-images/ocr/txt/",
                        help='load text files of ocr results from this path')
    
    parser.add_argument('--keyword_threshold', type=int, default=90,
                        help='Threshold for keyword fuzzy match')

    parser.add_argument('--item_threshold', type=int, default=0,
                        help='Number of matched items needed to filter a sentence')

    parser.add_argument('--spacy_model_name', type=str, default="en_core_web_sm",
                        help='spaCy language model to use')

    parser.add_argument('--output_path', type=str, default="./output/filtered_sentences_w_raw_txt.txt",
                        help='save filtered sentences to this path as txt file')

    return parser.parse_args()


def main():
    args = parse_args()

    nlp = spacy.load(args.spacy_model_name)
    # gdf = load_geojson_to_gdf(args.counties, args.columns_to_keep)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    file = open(args.output_path, 'w')


    root_dir = Path(args.root_path)
    for txt_file in root_dir.rglob("*.txt"):
        txt_path = Path(txt_file)

        if not txt_path.exists():
            print(f"[Warning] File not found: {txt_path}")
            continue
        
        text = ""
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = re.sub(r"\s+", " ", content.strip())
            text += content + "\n"

        relevant_sents = filter_relevant_sentences(
            text,
            entity_dict={},
            keyword_threshold=args.keyword_threshold,
            nlp=nlp
        )

        this_covenant_dense = [] # To store sentences with more than threshold number of matched keywords (I contains multipe keywords so I call it dense) 
        this_covenant_sent = []
        this_covenant_keywords = []

        for res in relevant_sents:
            temp_list = res["keyword_matches"] 
            temp_list = [x[0] for x in temp_list] # list of matched keywords in this sentence
            temp_tuple = (len(temp_list), res["sentence"], temp_list) # （number_of_matched_keywords, an_individual_sentence, list_of_matched_keywords）
            this_covenant_keywords += temp_list # list of matched keywords in this deed
            this_covenant_sent.append(res["sentence"]) # list of sentences with any number of matched keywords in this deed
            # this_covenant_list.append(temp_tuple)
            if len(temp_list) > args.item_threshold:
                this_covenant_dense.append(res["sentence"]) # list of sentences with more than threshold number of matched keywords in this deed
                # print(temp_tuple) # print tuple to see the keywords/entities matched in this sentence
                file.write(res["sentence"] + '\n')

    # print(this_covenant_dense)
    file.close()
    print(f"Filtered sentences saved to {args.output_path}")

if __name__ == "__main__":
    main()
