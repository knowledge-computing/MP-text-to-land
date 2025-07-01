'''
This script tests the geosentence_identifier
It prints a list of dict results from entity or keyword fuzzy match
    {
        "sentence": sentence, # string
        "geo_class": "GEO" if is_geo else "NON-GEO",
        "entity_matches": entity_matches, # list of tuple
        "keyword_matches": keyword_matches # list of tuple
    }
'''

import os
import spacy
import re
import geopandas as gpd
from rapidfuzz import fuzz, process
from tqdm import tqdm
from pathlib import Path
from src.data_preprocessing.geojson_loader import load_geojson_to_gdf
from src.data_preprocessing.sentence_identifier import filter_relevant_sentences

columns_to_keep = ["image_ids", "saved_path", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"]
entity_columns = []
# entity_columns = ["street_add", "cnty_name", "city", "state", "add_cov"]
# Keep the entity_columns an empty list so that it is purely keyword match, using the keywords identified in /data/keywords/*.txt files
# If you want to use purely entity matching and non keyword matching, please use empty txt file.
# The "entity" refers to the attributes appeared in the geojson locations

def main():

    nlp = spacy.load("en_core_web_sm")
    
    # gdf = load_geojson_to_gdf(["mn-anoka-county", "mn-dakota-county", "wi-milwaukee-county"], ["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"])
    gdf = load_geojson_to_gdf(["mn-anoka-county"], columns_to_keep)

    # Traverse the gdf and find all pages of deeds in *.txt files correspond to this gdf row
    for _, row in tqdm(gdf[columns_to_keep].iterrows(), total=len(gdf)):

        # Create a dictionary of entity attributes loaded from the geojson file for the current row 
        # This is used for geojson value entity matching with the text
        entity_dict = {col: row[col] for col in entity_columns}
        
        # This is the list of *.txt file paths for the current geojson item. 
        # If this geojson item has multiple pages of deeds, the saved_path will be a list of paths.
        # ensure all saved_path are list type to simplify the code
        path_list = row['saved_path'] if isinstance(row['saved_path'], list) else [row['saved_path']]
        print(path_list)
        
        # If a covenant location is consist of multiple pages of deeds, concat the text string.
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

        relevant_sents = filter_relevant_sentences(text, entity_dict={}, entity_threshold=90, keyword_threshold=90, nlp=nlp)
        # for res in relevant_sents:
            # print(res["sentence"])
            # print(" - geo_class:", res["geo_class"])
            # print(" - entity matches:", res["entity_matches"])
            # print(" - keyword matches:", res["keyword_matches"])
            # print(res)
            # print(type(res))

        this_covenant_list = []
        this_covenant_sent = []
        this_covenant_dense = []
        this_covenant_keywords = []
        for res in relevant_sents:
            # print(res["sentence"])
            # print(" - geo_class:", res["geo_class"])
            # print(" - entity matches:", res["entity_matches"])
            # print(" - keyword matches:", res["keyword_matches"])
            temp_list = res["entity_matches"] + res["keyword_matches"]
            temp_list = [x[0] for x in temp_list]
            temp_tuple = (len(temp_list), res["sentence"], temp_list)
            this_covenant_keywords += temp_list
            this_covenant_sent.append(res["sentence"])
            this_covenant_list.append(temp_tuple)
            if len(temp_list) > 5:
                this_covenant_dense.append(res["sentence"])
                # this_covenant_dense.append(temp_tuple)
                # print(temp_tuple)
            
            # print(temp_list)
            # print(temp_tuple)
            # print(res)
            # print(type(res["keyword_matches"]))
        this_covenant_list = sorted(this_covenant_list, key=lambda x: x[0], reverse=True)
        # print(this_covenant_list)
        # print(this_covenant_sent)
        print(this_covenant_dense)

if __name__ == "__main__":
    main()