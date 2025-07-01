import os
import geopandas as gpd

def extract_shp_attribute_to_txt(shp_path="data/subdivision_sample/", output_path="output/subdivision_names_"):

    target_columns = ["PLAT", "LEGAL_NAME", "Name", "PLATNAME"]
    
    for dirpath, dirnames, filenames in os.walk(shp_path):
        for file in filenames:
            if file.endswith(".shp"):
                full_path = os.path.join(dirpath, file)
                file_name = os.path.splitext(file)[0]
                output_file = f"{output_path}{file_name}.txt"
                all_values = []

                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created root dir: {output_dir}")

                try:
                    gdf = gpd.read_file(full_path)
                    for target_column in target_columns:
                        if target_column in gdf.columns:
                            values = gdf[target_column].dropna().astype(str).tolist()
                            all_values.extend(values)
                        else:
                            continue

                    with open(output_file, "w", encoding="utf-8") as f:
                        for val in all_values:
                            f.write(val + "\n")

                    print(f"Finished extracting {len(all_values)} values, written in: {output_file}")

                except Exception as e:
                    print(f"Failure reading: {full_path}, error: {e}")

    return None
