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
from src.data_preprocessing.geosentence_identifier import filter_relevant_sentences

columns_to_keep = ["image_ids", "saved_path", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"]
entity_columns = []
# entity_columns = ["street_add", "cnty_name", "city", "state", "add_cov"]



def main():

    nlp = spacy.load("en_core_web_lg")
    
    # gdf = load_geojson_to_gdf(["mn-anoka-county", "mn-dakota-county", "wi-milwaukee-county"], ["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"])
    gdf = load_geojson_to_gdf(["mn-anoka-county"], columns_to_keep)


    for _, row in tqdm(gdf[columns_to_keep].iterrows(), total=len(gdf)):

        entity_dict = {col: row[col] for col in entity_columns}

        # ensure all saved_path are list type
        path_list = row['saved_path'] if isinstance(row['saved_path'], list) else [row['saved_path']]
        print(path_list)

        text = ""
        
        # If a covenant location is consist of multiple pages of deeds, concat the text string.
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
        for res in relevant_sents:
            # print(res["sentence"])
            # print(" - geo_class:", res["geo_class"])
            # print(" - entity matches:", res["entity_matches"])
            # print(" - keyword matches:", res["keyword_matches"])
            print(res)
            print(type(res))

if __name__ == "__main__":
    main()