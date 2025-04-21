# import os
# import json
# import geopandas as gpd
# from dotenv import load_dotenv
# import pandas as pd

# def load_config():
#     # Load the environment variables from the .env file
#     load_dotenv()

#     # Get the values from the .env file
#     geojson_path = os.getenv('GEOJSON_PATH')
#     folder_names = os.getenv('FOLDER_NAMES').split(',')

#     return geojson_path, folder_names

# Traverse through each folder and read GeoJSON files
# for folder in folder_names:
#     folder_path = os.path.join(path, folder)
    
#     # Check if the folder exists
#     if os.path.exists(folder_path):
#         print("Reading files from: {}".format(folder_path))
        
#         # Traverse through the files in the folder
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith(".geojson"):
#                 geojson_file = os.path.join(folder_path, file_name)
                
#                 # Read the GeoJSON file using GeoPandas
#                 try:
#                     gdf = gpd.read_file(geojson_file)

#                     gdf = gdf[gdf['image_ids'].apply(lambda x: x != '' and x is not None)]

#                     # Convert the 'image_ids' column from string to actual lists
#                     gdf['image_ids'] = gdf['image_ids'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)

#                     # Manually correct these three file paths
#                     if folder == "mn-dakota-county":
#                         target_ids = {
#                             " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_549",
#                             " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_550",
#                             " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_551"
#                         }
#                         prefix = 'Abstract_Images_Books_MR N-Z'

#                         def filter_and_modify(ids):
#                             if not isinstance(ids, list):
#                                 return ids
#                             if any(id_ in target_ids for id_ in ids):
#                                 if prefix in ids:
#                                     # 去掉 prefix，然后拼接 prefix
#                                     ids = [id_ for id_ in ids if id_ != prefix]
#                                     ids = [f'{prefix}, {id_.strip()}' for id_ in ids]
#                             return ids

#                         gdf['image_ids'] = gdf['image_ids'].apply(filter_and_modify)
#                         # gdf_filtered = gdf[gdf['image_ids'].apply(lambda ids: any(id_.startswith(prefix) for id_ in ids))]
#                         # print(gdf_filtered.to_dict(orient='list'))
                    
#                     def create_saved_paths(row):
#                         path2 = [f"/home/yaoyi/jiao0052/MapPrejudice/data/covenants-deed-images/ocr/txt/{folder}/{image_id}.txt" for image_id in row['image_ids']]
#                         return path2
                    
#                     # Apply the function to each row to create new paths
#                     gdf['saved_path'] = gdf.apply(create_saved_paths, axis=1)

#                     # Delete rows with empty image_id
#                     def is_valid_image_ids(x):
#                         if x is None:
#                             return False
#                         if isinstance(x, list):
#                             return len(x) > 0 and all(isinstance(i, str) and i.strip() for i in x)
#                         if isinstance(x, str):
#                             return x.strip() != ''
#                         return False

#                     # 保留有效的 image_ids 行
#                     gdf = gdf[gdf['image_ids'].apply(is_valid_image_ids)].copy()

#                     columns_to_keep = ["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"]
#                     gdf = gdf[columns_to_keep]
#                     # entity_columns = ["deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"]
#                     entity_columns = ["street_add", "cnty_name", "city", "state", "add_cov"]

#                     print("Successfully loaded: {}".format(geojson_file))
#                     # print("Number of rows in the GeoDataFrame:", gdf.shape[0])


import os
import json
import geopandas as gpd
import pandas as pd
# from src.utils.error_correction import correct_geojson_errors
from src.data_preprocessing.path_generator import generate_saved_path
from src.utils.config_loader import load_config

# import sys
# import os
# import json
# import geopandas as gpd
# from src.utils.config_loader import load_config
# from src.data_preprocessing.path_generator import *

geojson_path, folder_names, ocrtxt_path, _ = load_config()

def filter_and_modify(ids, prefix='Abstract_Images_Books_MR N-Z', target_ids={" 27-71 by Book and Page/MISC/doc_NONE_book_46_page_549", " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_550", " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_551"}):
    """ This function has not been added to __init__.py 
        since this is currently used by only load_geojson_to_gdf() 
        to modify the spesific 3 files of dakota county
    """
    if not isinstance(ids, list):
        return ids
    if any(id_ in target_ids for id_ in ids):
        if prefix in ids:
            # 去掉 prefix，然后拼接 prefix
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
                        
                        # Correct errors and modify 'image_ids'
                        # gdf['image_ids'] = gdf['image_ids'].apply(correct_geojson_errors)
                        # Convert the 'image_ids' column from string to actual lists
                        gdf['image_ids'] = gdf['image_ids'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)


                        # Apply manual corrections ( 3 specific case for mn-dakota-county)
                        if folder == "mn-dakota-county":
                            # target_ids = {
                            #     " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_549",
                            #     " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_550",
                            #     " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_551"
                            # }
                            # prefix = 'Abstract_Images_Books_MR N-Z'

                            # def filter_and_modify(ids, prefix='Abstract_Images_Books_MR N-Z', target_ids={" 27-71 by Book and Page/MISC/doc_NONE_book_46_page_549", " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_550", " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_551"}):
                            #     if not isinstance(ids, list):
                            #         return ids
                            #     if any(id_ in target_ids for id_ in ids):
                            #         if prefix in ids:
                            #             # 去掉 prefix，然后拼接 prefix
                            #             ids = [id_ for id_ in ids if id_ != prefix]
                            #             ids = [f'{prefix}, {id_.strip()}' for id_ in ids]
                            #     return ids

                            # def filter_and_modify(ids):
                            #     if isinstance(ids, list) and any(id_ in target_ids for id_ in ids):
                            #         ids = [f'{prefix}, {id_.strip()}' for id_ in ids if id_ != prefix]
                            #     return ids
                            # gdf['image_ids'] = gdf['image_ids'].apply(filter_and_modify)
                            gdf['image_ids'] = gdf['image_ids'].apply(lambda ids: filter_and_modify(ids))
                            # gdf_filtered = gdf[gdf['image_ids'].apply(lambda ids: any(id_.startswith(prefix) for id_ in ids))]

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
    # columns_to_keep = ["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"]
    full_gdf = full_gdf[columns_to_keep]

    return full_gdf