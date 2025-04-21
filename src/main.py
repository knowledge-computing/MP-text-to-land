import os
import spacy
import re
import geopandas as gpd
from rapidfuzz import fuzz, process
from tqdm import tqdm
from src.data_preprocessing.geojson_loader import load_geojson_to_gdf
from src.utils.keyword_loader import load_keywords_from_txt_directory


GEOGRAPHIC_KEYWORDS = load_keywords_from_txt_directory("data/keywords/")
columns_to_keep = ["image_ids", "saved_path", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"]

def main():

    nlp = spacy.load("en_core_web_lg")
    
    # gdf = load_geojson_to_gdf(["mn-anoka-county", "wi-milwaukee-county"], ["image_ids", "saved_path", "deed_date", "seller", "buyer", "street_add", "cnty_name", "city", "state", "cov_text", "add_cov", "lot_cov", "block_cov"])
    gdf = load_geojson_to_gdf(["mn-dakota-county"], columns_to_keep)
    # gdf = load_geojson_to_gdf()
    # print(type(gdf))
    # print(gdf.head(3).to_dict(orient='list'))

    # rows, columns = gdf.shape
    # print(f"Number of rows: {rows}")
    # print(f"Number of columns: {columns}")
    # print(f"Column names: {gdf.columns.tolist()}") # Print the column names

    # Define the path to the keywords file
    # keywords_file_path = os.path.join("data", "geo_general.txt")  # or "keywords.json" for JSON

    # Load the keywords as a list of strings
      # or load_keywords_from_json

    # Print the loaded keywords to verify
    # print("Loaded keywords:", keywords)




    for _, row in tqdm(gdf[columns_to_keep].iterrows(), total=len(gdf)):
        # 提取 image_id（用于后续命名时备用）
        image_id = row['image_ids'][0] if isinstance(row['image_ids'], list) else row['image_ids']

        # 确保 saved_path 是 list
        path_list = row['saved_path'] if isinstance(row['saved_path'], list) else [row['saved_path']]
        print(path_list)
        # print(row['image_ids'][0])

        text = ""
        first_txt_path = None  # 用于计算 relative path
        

        for p in path_list:
            txt_path = Path(p)
            if not txt_path.exists():
                print(f"[Warning] File not found: {txt_path}")
                continue
            if first_txt_path is None:
                first_txt_path = txt_path
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content = re.sub(r"\s+", " ", content.strip())
                text += content + "\n"

        # spaCy 分词与标注
        # doc = nlp(text)
        entity_dict = {col: row[col] for col in entity_columns}


        relevant_sents = filter_relevant_sentences(text, entity_dict)
        for res in relevant_sents:
            print(res["sentence"])
            print(" - geo_class:", res["geo_class"])
            print(" - entity matches:", res["entity_matches"])
            print(" - keyword matches:", res["keyword_matches"])

        # bio = get_bio_labels(doc, entity_dict)
        # print(type(bio))
        # for token, label in bio:
        #     print(f"{token}\t{label}")

        # 构造输出路径
        try:
            relative_path = os.path.relpath(first_txt_path, start=input_base_dir)
            # print(relative_path) # mn-dakota-county/Abstract_Images_Books_Deeds 104-277 by Book and Page/DEEDS/doc_NONE_book_228_page_514.txt
        except ValueError:
            print(f"[Warning] Cannot resolve relative path for {first_txt_path}")
            continue

        output_path = output_base_dir / Path(relative_path).with_suffix(".bio.txt")
        os.makedirs(output_path.parent, exist_ok=True)
        # print(output_path) # /home/yaoyi/jiao0052/MapPrejudice/output/try01/mn-dakota-county/Abstract_Images_Books_Deeds 104-277 by Book and Page/DEEDS/doc_NONE_book_228_page_514.bio.txt

        # 写入 BIO 格式
        with open(output_path, "w", encoding="utf-8") as out_f:
            for token, label in zip(doc, labels):
                if not token.is_space:
                    out_f.write(f"{token.text}\t{label}\n")



if __name__ == "__main__":
    main()