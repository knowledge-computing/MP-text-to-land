from src.data_preprocessing.geojson_loader import load_geojson_to_gdf
# import geopandas as gpd
from src.utils.keyword_loader import load_keywords_from_text_file  # or load_keywords_from_json
import os

def main():

def main():
    
    # gdf = load_geojson_to_gdf(["mn-anoka-county", "wi-milwaukee-county"], ["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"])
    gdf = load_geojson_to_gdf(["mn-dakota-county"], ["image_ids", "saved_path", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"])
    # gdf = load_geojson_to_gdf()
    # print(type(gdf))
    print(gdf.head(3).to_dict(orient='list'))

    rows, columns = gdf.shape
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {columns}")
    # Print the column names
    print(f"Column names: {gdf.columns.tolist()}")

    # Define the path to the keywords file
    keywords_file_path = os.path.join("data", "geo_general.txt")  # or "keywords.json" for JSON

    # Load the keywords as a list of strings
    keywords = load_keywords_from_text_file(keywords_file_path)  # or load_keywords_from_json

    # Print the loaded keywords to verify
    print("Loaded keywords:", keywords)


if __name__ == "__main__":
    main()