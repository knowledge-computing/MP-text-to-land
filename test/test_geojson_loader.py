from src.data_preprocessing.geojson_loader import load_geojson_to_gdf
# import geopandas as gpd

def main():
    
    # gdf = load_geojson_to_gdf(["wi-milwaukee-county"], ["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"])
    # gdf = load_geojson_to_gdf(["mn-anoka-county", "mn-dakota-county"], ["image_ids", "saved_path", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"])
    gdf = load_geojson_to_gdf()
    # print(type(gdf))
    print(gdf.head(3).to_dict(orient='list'))

    rows, columns = gdf.shape
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {columns}")
    # Print the column names
    print(f"Column names: {gdf.columns.tolist()}")


if __name__ == "__main__":
    main()