"""
This script is the geosentence_identifier.
It prints a list of dict results from entity or keyword fuzzy match:
    {
        "sentence": sentence,  # string
        "geo_class": "GEO" if is_geo else "NON-GEO",
        "entity_matches": entity_matches,  # list of tuple
        "keyword_matches": keyword_matches  # list of tuple
    }
"""

import os
import re
import spacy
import argparse
import geopandas as gpd
from rapidfuzz import fuzz, process
from tqdm import tqdm
from pathlib import Path
from src.data_preprocessing.geojson_loader import load_geojson_to_gdf
from src.data_preprocessing.sentence_identifier import filter_relevant_sentences
from src.utils.config_loader import load_config

geojson_path, folder_names, ocrtxt_path, _ = load_config()


def parse_args():
    parser = argparse.ArgumentParser(description="Test geo sentence identifier")

    parser.add_argument('--counties', nargs='+', default=["mn-anoka-county", "mn-dakota-county", "mn-olmsted-county", "mn-sherburne-county", "mn-washington-county", "wi-milwaukee-county"],
                        help='List of counties to load geojson data from')

    parser.add_argument('--columns_to_keep', nargs='+', default=["image_ids", "saved_path", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"],
                        help='List of columns to keep when loading geodataframe')

    parser.add_argument('--entity_columns', nargs='*', default=[],
                        help='Columns from geojson attributes used for entity matching. Example: street_add, add_cov')

    parser.add_argument('--entity_threshold', type=int, default=90,
                        help='Threshold for entity fuzzy match')

    parser.add_argument('--keyword_threshold', type=int, default=90,
                        help='Threshold for keyword fuzzy match')

    parser.add_argument('--item_threshold', type=int, default=0,
                        help='Number of matched items needed to filter a sentence')

    parser.add_argument('--spacy_model_name', type=str, default="en_core_web_sm",
                        help='spaCy language model to use')

    parser.add_argument('--output_path', type=str, default="./output/filtered_sentences.txt",
                        help='save filtered sentences to this path as txt file')

    return parser.parse_args()


def main():
    args = parse_args()

    nlp = spacy.load(args.spacy_model_name)
    gdf = load_geojson_to_gdf(args.counties, args.columns_to_keep)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    file = open(args.output_path, 'w')


    # Traverse the gdf and find all pages of deeds in *.txt files correspond to this gdf row
    for _, row in tqdm(gdf[args.columns_to_keep].iterrows(), total=len(gdf)):

        # This is the list of *.txt file paths for the current geojson item. 
        # If this geojson item has multiple pages of deeds, the saved_path will be a list of paths.
        path_list = row['saved_path'] if isinstance(row['saved_path'], list) else [row['saved_path']]
        print(f"\n[Processing] {path_list}") 
        
        # If a covenant location is consist of multiple pages of deeds, concat the text string into one.
        # If a covenant location is just consiste of one page of deed, directly save it to the text string. 
        # Read txt files and concat
        text = ""
        for p in path_list:
            txt_path = Path(p)
            if not txt_path.exists():
                print(f"[Warning] File not found: {txt_path}")
                continue
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content = re.sub(r"\s+", " ", content.strip())
                text += content + "\n"

        # Create a dictionary of entity attributes loaded from the geojson file for the current row 
        # This is used for geojson value entity matching with the text
        entity_dict = {col: row[col] for col in args.entity_columns}
        # print(entity_dict)

        relevant_sents = filter_relevant_sentences(
            text,
            entity_dict=entity_dict,
            entity_threshold=args.entity_threshold,
            keyword_threshold=args.keyword_threshold,
            nlp=nlp
        )

        this_covenant_dense = [] # To store sentences with more than threshold number of matched keywords (I contains multipe keywords so I call it dense) 
        this_covenant_sent = []
        this_covenant_keywords = []

        for res in relevant_sents:
            # print(res["sentence"])
            # print(" - geo_class:", res["geo_class"])
            # print(" - entity matches:", res["entity_matches"])
            # print(" - keyword matches:", res["keyword_matches"])
            temp_list = res["entity_matches"] + res["keyword_matches"] 
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
