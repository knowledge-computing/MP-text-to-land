import os
import json
import geopandas as gpd
import pandas as pd

from src.data_preprocessing.path_generator import generate_saved_path
from src.utils.config_loader import load_config

geojson_path, folder_names, ocrtxt_path, _ = load_config()

def filter_and_modify(ids, prefix='Abstract_Images_Books_MR N-Z', target_ids={" 27-71 by Book and Page/MISC/doc_NONE_book_46_page_549", " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_550", " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_551"}):
    """ 
        This function has not been added to __init__.py 
        since this is currently used by only load_geojson_to_gdf() 
        to modify the specific 3 files of dakota county
    """
    if not isinstance(ids, list):
        return ids
    if any(id_ in target_ids for id_ in ids):
        if prefix in ids:
            # remove prefix from list，then concat prefix with suffix
            ids = [id_ for id_ in ids if id_ != prefix]
            ids = [f'{prefix}, {id_.strip()}' for id_ in ids]
    return ids

def load_geojson_to_gdf(list_of_county_folder=folder_names, columns_to_keep=["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov", 'add_mod', 'block_mod', 'lot_mod', 'ph_dsc_mod', 'geometry', 'zip_code', 'cnty_pin', 'cnty_fips', 'workflow', 'doc_num', 'zn_subj_id', 'zn_dt_ret', 'cov_type', 'med_score', 'manual_cx', 'match_type', 'plat', 'dt_updated']):
    """
        Read, process, and filter GeoJSON file.
        concat gdf of each county
        Return a full_gdf
    
    """
    # variable "list_of_county_folder" supports identify a list of foler names of the county you want to get

    full_gdf = gpd.GeoDataFrame()
    # Traverse through each folder and read GeoJSON files
    for folder in list_of_county_folder:
        folder_path = os.path.join(geojson_path, folder)
        
        # Check if the folder exists
        if os.path.exists(folder_path):
            print("Reading files from: {}".format(folder_path))
            
            # Traverse through the files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".geojson"):
                    geojson_file = os.path.join(folder_path, file_name)

                    try:
                        gdf = gpd.read_file(geojson_file)
                        
                        # Filter out invalid 'image_ids'
                        gdf = gdf[gdf['image_ids'].apply(lambda x: x != '' and x is not None)]

                        # Convert the 'image_ids' column from string to actual lists
                        gdf['image_ids'] = gdf['image_ids'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)


                        # Apply manual corrections ( 3 specific case for mn-dakota-county)
                        if folder == "mn-dakota-county":
                            gdf['image_ids'] = gdf['image_ids'].apply(lambda ids: filter_and_modify(ids))

                        # Create 'saved_path' for each row
                        gdf['saved_path'] = gdf.apply(lambda row: generate_saved_path(row, folder), axis=1)

                        # Remove rows with invalid 'image_ids'
                        gdf = gdf[gdf['image_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(i, str) and i.strip() for i in x))]
                        
                        print("Successfully loaded: {}".format(geojson_file))
                        # Concatenate the temporary GDF with the full GDF
                        full_gdf = pd.concat([full_gdf, gdf], ignore_index=True)
                        del gdf

                    except Exception as e:
                        print(f"Error processing {geojson_file}: {e}")
                        return None
                    
                    
    # Select relevant columns
    # print("Selecting relevant columns...", type(columns_to_keep))
    full_gdf = full_gdf[columns_to_keep]

    return full_gdf