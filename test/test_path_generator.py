import sys
import os
import json
import geopandas as gpd
from src.utils.config_loader import load_config
from src.data_preprocessing.path_generator import *

# Add the 'src' directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def main():

    geojson_path, folder_names, ocrtxt_path, s3_path = load_config()

    print(geojson_path)
    print(folder_names)

    # Traverse through each folder and read GeoJSON files
    for folder in folder_names:
        folder_path = os.path.join(geojson_path, folder)
        
        # Check if the folder exists
        if os.path.exists(folder_path):
            print("Reading files from: {}".format(folder_path))
            
            # Traverse through the files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".geojson"):
                    geojson_file = os.path.join(folder_path, file_name)
                    
                    # Read the GeoJSON file using GeoPandas
                    try:
                        gdf = gpd.read_file(geojson_file)
                        gdf = gdf[gdf['image_ids'].apply(lambda x: x != '' and x is not None)]

                        # Convert the 'image_ids' column from string to actual lists
                        gdf['image_ids'] = gdf['image_ids'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)

                        # Manually correct these three file paths
                        if folder == "mn-dakota-county":
                            target_ids = {
                                " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_549",
                                " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_550",
                                " 27-71 by Book and Page/MISC/doc_NONE_book_46_page_551"
                            }
                            prefix = 'Abstract_Images_Books_MR N-Z'

                            def filter_and_modify(ids):
                                if not isinstance(ids, list):
                                    return ids
                                if any(id_ in target_ids for id_ in ids):
                                    if prefix in ids:
                                        # 去掉 prefix，然后拼接 prefix
                                        ids = [id_ for id_ in ids if id_ != prefix]
                                        ids = [f'{prefix}, {id_.strip()}' for id_ in ids]
                                return ids

                            gdf['image_ids'] = gdf['image_ids'].apply(filter_and_modify)

                        # Apply the function to each row to create new paths
                        gdf['s3_path'] = gdf.apply(lambda row: create_s3_paths(row, folder), axis=1)
                        gdf['download_path'] = gdf.apply(lambda row: create_download_paths(row, folder), axis=1)
                        gdf['saved_path'] = gdf.apply(lambda row: generate_saved_path(row, folder), axis=1)
                        gdf['command'] = gdf.apply(lambda row: create_command_paths(row, folder), axis=1)

                        # Delete rows with empty image_id
                        def is_valid_image_ids(x):
                            if x is None:
                                return False
                            if isinstance(x, list):
                                return len(x) > 0 and all(isinstance(i, str) and i.strip() for i in x)
                            if isinstance(x, str):
                                return x.strip() != ''
                            return False

                        # filter out rows with valid image_ids
                        gdf = gdf[gdf['image_ids'].apply(is_valid_image_ids)].copy()

                        columns_to_keep = ["image_ids", "s3_path", "download_path", "saved_path", "command"]
                        gdf = gdf[columns_to_keep]
                        print(gdf.head(3).to_dict(orient='list'))

                        save_commands_as_txt(gdf.head(2), f"/home/yaoyi/jiao0052/MapPrejudice/output/test_{folder}_commands.txt")

                        print("Successfully loaded: {}".format(geojson_file))

                    except Exception as e:
                        print("Error reading {}: {}".format(geojson_file, e))

if __name__ == "__main__":
    main()