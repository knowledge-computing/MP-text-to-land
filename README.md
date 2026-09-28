# MP-text-to-land

**Discover geographic entities in historical deed documents.**  
This project aims to automatically extract geographic entities—such as lot numbers, blocks, subdivisions, and more—from historical deed texts and output structured data for use in geographic information systems (GIS) and downstream analysis.

As a tool developer, my goal is to help the Map Prejudice team automate the process of extracting important information and try to approximate the same metadata format as in the existing `*.geojson` attributes: each feature item corresponds to a different deed's images, details, and its property location. Thus, the tools 3 - 9 aim at generating the `*.geojson`-similar format with the automatically extracted geo-info strings, hoping to achieve an effect similar to the volunteer labels or to effectively prompt during the crowdsourcing.

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Usage](#usage)  

## Features

- Named entity recognition (NER) for geographic entities in deed texts
- spaCy-based NLP pipeline with keyword and fuzzy matching
- Outputs structured data in JSON and BIO tagging formats

## Installation

### Prerequisites

- Python 3.10.16 (original Linux environment) or Python 3.11.9 (verified on macOS)
- Recommended: use `virtualenv` or `conda` for environment isolation

> **Note on requirements files**
> `requirements.txt` was exported from a Linux/conda environment. It contains
> conda-build wheel paths (`@ file:///croot/...`) built for Linux x86_64 / Python 3.10,
> so `pip install -r requirements.txt` may fail on macOS. On macOS, use
> `requirements-macos.txt`, which pins only the direct dependencies to versions
> verified on macOS. `requirements.txt` is kept for reference.

### Dependencies

- `spaCy`
- `GeoPandas`
- `RapidFuzz`
- `pandas`
- `scikit-learn`

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/knowledge-computing/MP-text-to-land.git
   cd MP-text-to-land
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   # macOS (recommended there; see the note above)
   pip install -r requirements-macos.txt

   # Original Linux/conda environment
   pip install -r requirements.txt
   ```

4. Download spaCy English model(s):
   ```bash
   python -m spacy download en_core_web_sm   # already included in requirements-macos.txt
   python -m spacy download en_core_web_md   # needed only for Tool 3's default --spacy_model_name
   ```

5. Create an `.env` file in the root of the MP-text-to-land folder by copying the template, then fill in your local paths:
   ```bash
   cp .env.example .env
   ```
   ```
   GEOJSON_PATH=""   # The path where you store the *.geojson files (per-county folders).
   OCRTXT_PATH=""    # The path where you store the *.txt files of the OCR results. Keep the trailing "/".
   S3_PATH=""        # (optional) The path to find the *.txt files of the OCR results from the S3 buckets. Keep the trailing "/".
   FOLDER_NAMES="mn-anoka-county,mn-dakota-county,mn-olmsted-county,mn-sherburne-county,mn-washington-county,wi-milwaukee-county" # County folder names as a comma-separated string
   ```
   `.env` is git-ignored and must not be committed. `FOLDER_NAMES` must be set even for tools that do not read GeoJSON, because importing `src` reads it. Run all tools from the repository root.

## Usage
### Tool 1: Identify deed sentences containing specified keywords (load *.txt files directly from the ocr results folder)
The tool 1 is designed for deed documents in `*.txt` format without volunteer tagged information.
This tool loads `*.txt` files of deed documents by iterating the folder of ocr results, and it supports identifying sentences with pre-defined keywords by string matching and save those sentences in an output `*.txt` file.

For example, assuming the user of this tool has collected a list of keywords in a `*.txt` file (each row is a single keyword) under the `data/keywords/` directory, then the user could identify "--keyword_threshold XX" "--item_threshold N" (XX is a number, suggested 90; N is a number, suggested > 1) while calling this tool. The sentences from the ocr `*.txt` files containing more than N keywords specified in the keywords list would be filtered out and saved at the output path.

#### 🔧 How to Use

1. **Prepare your data**
   - Place OCR `*.txt` deed files under the `data/` directory, using the same folder structure as in the S3 bucket.
   - Place all the keywords lists at the `data/keywords/` directory. All files endwith `*.txt` at the `data/keywords/` will be loaded. Each row in the `*.txt` files will be considered as a keyword. 
   - The `*.txt` files at the`data/keywords/not_using` directory would not be loaded anytime, so this is where the user can store the temporarily unnecessary keyword lists. Users can find the developer suggested keywords there too.

2. **Run the identifier**
```bash
python -m scripts.run_sentence_identifier_w_raw_txt \
    --root_path /root/path/covenants-deed-images/ocr/txt/ \
    --keyword_threshold 90 \
    --item_threshold 2 \
    --spacy_model_name en_core_web_sm \
    --output_path ./output/filtered_sentences_1.txt
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/filtered_sentences_1.txt`.

#### 🛠️ Arguments
- `--root_path`: Load ocr `*.txt` files from this path.
- `--keyword_threshold`: Fuzzy match threshold for keywords (0–100), recommended 80-90.
- `--item_threshold`: Minimum number of matched keywords to keep a sentence (>=0) recommended at least 2 for keyword matching.
- `--output_path`: Output `*.txt` file to save filtered sentences.
- `--spacy_model_name`: spaCy model to use for sentence tokenization and preprocessing. The options include: en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf. The model selection here for this tool does not affect the output because all of these would call the spaCy sentencizer.


### Tool 2: Identify deed sentences containing specified keywords/entities (load *.txt files by constructing paths with *.geojson image_ids)
The tool 2 is designed for deed documents that already have volunteer tagged information in its corresponding geojson file.
This tool loads ocr `*.txt` files of deed documents referring to the `*.txt` paths in geodataframe created with image_ids in geojson, and it supports identifying sentences with volunteer labeled attributes or pre-defined keywords by string matching and save those sentences in an individual `*.txt` file.

For example, assuming the volunteer labeled addresses are stored in the "street_add" column in the geojson, then the user could identify "--entity_columns street_add" "--entity_threshold XX" "--item_threshold 0" (XX is a number, suggested 90) while calling this tool. The sentences from the ocr `*.txt` files containing the volunteer labeled "street_add" strings would be filtered out and saved at the output path.
Another example is that, assuming the user of this tool has collected a list of keywords in a `*.txt` file (each row is a single keyword) under the `data/keywords/` directory, then the user could identify "--keyword_threshold XX" "--item_threshold N" (XX is a number, suggested 90; N is a number, suggested > 1) while calling this tool. The sentences from the ocr `*.txt` files containing more than N keywords specified in the keywords list would be filtered out and saved at the output path.

#### 🔧 How to Use

1. **Prepare your data**
   - Place OCR `*.txt` deed files under the `data/` directory, using the same folder structure as in the S3 bucket.
   - Place `*.geojson` files of the selected counties in the same `data/` folder. These should include fields like `image_ids`, and geographic metadata columns.
   - If you would like to filter out sentences containing volunteer labeled entities from `*.geojson`, please keep the `data/keywords/` directory with nothing endwith `*.txt`.
   - If you would like to filter out sentences containing specified keywords saved as `*.txt` files, please put all the keywords lists at the `data/keywords/` directory and not specify any `--entity_columns ` when calling this tool. All files endwith `*.txt` at the `data/keywords/` will be loaded. Each row in the `*.txt` files will be considered as a keyword. 
   - The `*.txt` files at the`data/keywords/not_using` directory would not be loaded anytime, so this is where the user can store the temporarily unnecessary keyword lists. Users can also find the developer suggested keywords there.
   - Please filter out only volunteer labeled entities or only user specified keywords every time. Otherwise, it would be difficult to define a reasonable `--item_threshold`.

2. **Run the identifier**
If you would like to identify sentences containing volunteer labeled entities from the deeds that already included in a `.geojson`:
```bash
python -m scripts.run_sentence_identifier_w_geojson \
    --counties mn-anoka-county \
    --entity_columns street_add add_cov \
    --entity_threshold 90 \
    --item_threshold 0 \
    --spacy_model_name en_core_web_sm \
    --output_path ./output/filtered_sentences_2.txt
```
If you would like to identify sentences containing specified keywords from the deeds that already included in a `*.geojson`:
```bash
python -m scripts.run_sentence_identifier_w_geojson \
    --counties mn-anoka-county \
    --keyword_threshold 90 \
    --item_threshold 2 \
    --spacy_model_name en_core_web_sm \
    --output_path ./output/filtered_sentences_2.txt
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/filtered_sentences_2.txt`.

#### 🛠️ Arguments
- `--counties`: List of counties to load `*.geojson` files from (must match folder names)
- `--entity_columns`: Geojson attribute fields to be used for fuzzy entity matching (e.g., `street_add`)
- `--entity_threshold`: Fuzzy match threshold for entities (0–100), recommended 80-90.
- `--keyword_threshold`: Fuzzy match threshold for keywords (0–100), recommended 80-90.
- `--item_threshold`: Minimum number of matched items (entities or keywords) to keep a sentence (>=0), recommended 0 for single entity matching, recommended at least 2 for keyword matching.
- `--output_path`: Output `*.txt` file to save filtered sentences
- `--spacy_model_name`: spaCy model to use for sentence tokenization and preprocessing. The options include: en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf. The model selection here for this tool does not affect the output because all of these would call the spaCy sentencizer.

### Note for Tools 3–9: records are identified by `county` + `image_ids`
`image_ids` are relative to the county folder, so the same id can exist in two counties. Since commit `55003a0`, every Tool 3–9 row therefore carries a `county` field, and records are identified by (`county`, `image_ids`):
- **Tool 3** writes `{"text", "county", "image_ids"}`. `county` is the first folder under `--root_path`, i.e. the county folder name, which is also the GeoJSON folder name. So `--root_path` must point to the folder that contains the county folders, not to a single county folder. Tool 3 never puts pages from different counties into one row.
- **Tools 4, 5, 6, 7 and 8** keep `county` in every output row and group sentences by `county` + `image_ids`.
- **Tool 9** merges rows by `county` + `image_ids` and writes `county` once per record.
- **Older files without `county`** are read as `"county": null`: they are grouped among themselves as before and never merged with rows that have a county.
- **Do not mix** Tool 3 outputs made before this change with newer downstream outputs. Rerun from Tool 3 onward.

An aggregate-only check (`python3 dev/check_county_id_collisions.py`, which prints counts only) found no cross-county collisions in the current data: 0 page groups mixing more than one county, and 0 county-relative ids present in more than one county. The `county` field prevents silent cross-county merges if future data has them. Pages of one deed that sit in different folders *within* one county are a separate open question and are not changed here.

### Tool 3: Identify deed sentences containing specified keywords and save the sentences with directory paths in *.jsonl (load *.txt files directly from the ocr results folder)
The tool 3 is designed for deed documents in `*.txt` format without volunteer tagged information.
This tool first groups the `*.txt` files that belong to the same deed document by comparing the file paths, then it loads `*.txt` files of deed documents by iterating the paths in the same group (which means these `*.txt` files jointly form a single deed document), and it supports identifying sentences with pre-defined keywords by string matching and save those sentences and the grouped file paths ('image_ids') in an output `*.jsonl` file.

For example, assuming the user of this tool has collected a list of keywords in a `*.txt` file (each row is a single keyword) under the `data/keywords/` directory, then the user could identify "--keyword_threshold XX" "--item_threshold N" (XX is a number, suggested 90; N is a number, suggested > 1) while calling this tool. The sentences from the ocr `*.txt` files containing more than N keywords specified in the keywords list would be filtered out and saved at the output path as `*.jsonl` file.

#### 🔧 How to Use

1. **Prepare your data**
   - Place OCR `*.txt` deed files under the `data/` directory, using the same folder structure as in the S3 bucket.
   - `--root_path` must point to the folder that **contains the county folders**, i.e. the same folder as `OCRTXT_PATH` in `.env` (e.g. `.../covenants-deed-images/ocr/txt/`, which contains `mn-anoka-county/`, `wi-milwaukee-county/`, ...). The first folder under it becomes each row's `county`. Do not point it at a single county folder: `county` would then be that county's first subfolder, and the first subfolder of every image_id would be dropped.
   - Place all the keywords lists at the `data/keywords/` directory. All files endwith `*.txt` at the `data/keywords/` will be loaded. Each row in the `*.txt` files will be considered as a keyword. 
   - The `*.txt` files at the`data/keywords/not_using` directory would not be loaded anytime, so this is where the user can store the temporarily unnecessary keyword lists. Users can find the developer suggested keywords there too.

2. **Run the identifier**
```bash
python -m scripts.run_identify_sentences_with_imageids_f_raw_txt_to_jsonl \
    --root_path /root/path/.../covenants-deed-images/ocr/txt/ \
    --keyword_threshold 90 \
    --item_threshold 2 \
    --spacy_model_name en_core_web_sm \
    --output_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/filtered_sentences_w_raw_txt_and_image_ids.jsonl`.
Each sentence are saved in this format:
`{"text": "Lots 1 in block one and lot 2 in block two.", "county": "mn-anoka-county", "image_ids": ["toy-book-a/page_001", "toy-book-a/page_002"]}`

Each `image_ids` entry is the OCR `*.txt` path **relative to its county folder**, without `.txt`, so it matches the `image_ids` in the county's `*.geojson`. For example, `<root_path>/mn-anoka-county/toy-book-a/page_001.txt` becomes `"toy-book-a/page_001"` with `"county": "mn-anoka-county"`. This does not depend on where `--root_path` is on disk, whether it ends with `/`, or whether it is a relative path.

> **Note:** Before commit `68a61cb`, Tool 3 derived image_ids with `split('/', 10)`, which only matched the GeoJSON ids at one specific folder depth. Do not mix Tool 3 outputs (or Tool 4–9 outputs built from them) created before and after this fix, because Tool 9 groups results by `image_ids`. Rerun Tool 3 → 7 → 9 from scratch. Likewise, outputs made before commit `55003a0` have no `county` field (read as `null`); do not mix them with newer downstream outputs either.

#### 🛠️ Arguments
- `--root_path`: Load ocr `*.txt` files from this path. It must be the folder that contains the county folders (the same as `OCRTXT_PATH`); the first folder under it becomes each row's `county`.
- `--keyword_threshold`: Fuzzy match threshold for keywords (0–100), recommended 80-90.
- `--item_threshold`: Minimum number of matched keywords to keep a sentence (>=0) recommended at least 2 for keyword matching.
- `--output_path`: Output `*.jsonl` file to save filtered sentences.
- `--spacy_model_name`: spaCy model to use for sentence tokenization and preprocessing. The options include: en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf. The model selection here for this tool does not affect the output because all of these would call the spaCy sentencizer.

### Tool 4: Identify state/county/city in deed sentences with trained named entity recognition (NER) model
The tool 4 is designed for state/county/city name entity recognition (NER) for sentences filtered out from deed documents that might contain geographic or parcel information.
This tool loads filtered sentences of deed documents saved in `*.jsonl` files referring to the `*.txt` paths created with image_ids associated with each sentences in the `*.jsonl`. It supports identifying multiple state/county/city names in an individual sentence if it contains many, and it leaves an empty list "[]" as the result if a sentence does not contain any. The input `*.jsonl` file format is exactly the same as the `*.jsonl` output from the tool 3, so please use tool 3 & 4 by sequence.

The model is trained from strach based on spaCy NER pipeline. The training data were geographic and parcel related sentences identified with tool 2 from the deed document ocr text results of six MN/WI counties, and then transformed to `*.jsonl` format and then labeled with developer collected state/county/city names and expressions referring to online public materials. The best performance model is saved at `./src/models/state_county_city_ner_model/model-best`. 

Here is an example scenario of applying the tool 4. Assuming we have already identified the sentences from the original ocr results with tool 3 and saved each sentence associated with its `image_ids` at `output/filtered_sentences_w_raw_txt_and_image_ids.jsonl`, then the user can declare use its root path as the "--file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl" and the model path `--model_path ./src/models/state_county_city_ner_model/model-best`. The model would recognize the state/county/city names from each of the sentences and save the extracted string by only extending the original dictionary items (adding a `{"NERpredicted_STATE": ["state_string"], "NERpredicted_CNTY": ["county_string"], "NERpredicted_CTY": ["city_string"]}` to each dictionary in the `*.jsonl`). 

#### 🔧 How to Use

1. **Prepare your data**
   - Use tool 3 to identify the geographic and parcel related sentences and save each sentence associate with its `image_ids` list. Place the `*.jsonl` result under the `output` or `data` folder.
   - Place the model under the `./src/models` directory.

2. **Run the identifier**
If you would like to recognize state/county/city from the identified sentences from the deeds that already included in the tool 3's `*.jsonl` output:
```bash
python -m scripts.run_state_county_city_ner_model \
    --file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl \
    --model_path ./src/models/state_county_city_ner_model/model-best \
    --output_path ./output/state_cnty_cty_w_txt_n_image_ids.jsonl
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/state_cnty_cty_w_txt_n_image_ids.jsonl`.
Each sentence and its ids and state/county/city are saved in this format:
`{"text": ".....parcel of land lying in the County of Anoka and State of Minnisota described as follows, to-wit: Block XX, XX Addition, to the City of Anoka, County of Anoka and State of Minnesota.", "county": "mn-anoka-county", "image_ids": ["toy-book-a/page_001", "toy-book-a/page_002"], "NERpredicted_STATE": ["State of Minnesota"], "NERpredicted_CNTY": ["County of Anoka", "County of Anoka"], "NERpredicted_CTY": ["City of Anoka"]}`.

`county` is copied from the Tool 3 input row (`null` for older files without it), and sentences are grouped by `county` + `image_ids` (see the note above Tool 3).

#### 🛠️ Arguments
- `--file_path`: The exact same format as the tool 3 output `*.jsonl` file
- `--model_path`: The pretrained spaCy NER model for state/county/city.
- `--output_path`: The output `*.jsonl` file to store the original filtered sentences associated with its `image_ids` list and recognized entities `"NERpredicted_STATE", "NERpredicted_CNTY", "NERpredicted_CTY"`.


### Tool 5: Identify subdivisions in deed sentences with trained subdivision named entity recognition (NER) model
The tool 5 is designed for subdivision name entity recognition (NER) for sentences filtered out from deed documents that might contain geographic or parcel information.
This tool loads filtered sentences of deed documents saved in `*.jsonl` files referring to the `*.txt` paths created with image_ids associated with each sentences in the `*.jsonl`. It supports identifying multiple subdivision names in an individual sentence if it contains many, and it leaves an empty list "[]" as the result if a sentence does not contain any subdivision name. The input `*.jsonl` file format is exactly the same as the `*.jsonl` output from the tool 3, so please use tool 3 & 5 by sequence.

The model is trained from strach based on spaCy NER pipeline. The training data were geographic and parcel related sentences identified with tool 2 from the deed document ocr text results of six counties, and then transformed to `*.jsonl` format and then labeled with the volunteer transcribed subdivision names from the `*.geojson`. The best performance model after 20 epochs is saved at `./src/models/subdivision_ner_model/model-best`. The model training python script is stored at `/src/training/subdivision_ner_training.py` FYI.

Here is an example scenario of applying the tool 5. Assuming we have already identified the sentences from the original ocr results with tool 3 and saved each sentence associated with its `image_ids` at `output/filtered_sentences_w_raw_txt_and_image_ids.jsonl`, then the user can declare use its root path as the "--file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl" and the model path `--model_path ./src/models/subdivision_ner_model/model-best`. The model would recognize the subdivision names from each of the sentences and save the extracted string by only extending the original dictionary items (adding a `{"NERpredicted_SUBD": ["Subd_name"]}` to each dictionary in the `*.jsonl`). 

#### 🔧 How to Use

1. **Prepare your data**
   - Use tool 3 to identify the geographic and parcel related sentences and save each sentence associate with its `image_ids` list. Place the `*.jsonl` result under the `output` or `data` folder.
   - Place the model under the `./src/models` directory.

2. **Run the identifier**
If you would like to recognize subdivision names from the identified sentences from the deeds that already included in the tool 3's `*.jsonl` output:
```bash
python -m scripts.run_subd_ner_model \
    --file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl \
    --model_path ./src/models/subdivision_ner_model/model-best \
    --output_path ./output/subd_w_txt_n_image_ids.jsonl
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/subd_w_txt_n_image_ids.jsonl`.
Each sentence and its ids and subdivision names are saved in this format:
`{"text": "Lots 1 in block one, An Example Addition, Minneapolis, Minnesota, and lot 2 in block two, Another Example Addition, St Paul, Minnesota.", "county": "mn-anoka-county", "image_ids": ["toy-book-a/page_001", "toy-book-a/page_002"], "NERpredicted_SUBD": ["An Example Addition", "Another Example Addition"]}`.

`county` is copied from the Tool 3 input row (`null` for older files without it), and sentences are grouped by `county` + `image_ids` (see the note above Tool 3).

#### 🛠️ Arguments
- `--file_path`: The exact same format as the tool 3 output `*.jsonl` file
- `--model_path`: The pretrained spaCy NER model for subdivision names.
- `--output_path`: The output `*.jsonl` file to store the original filtered sentences associated with its `image_ids` list and recognized subdivision names `"NERpredicted_SUBD"`

### Tool 6: Identify parcel information in deed sentences with trained named entity recognition (NER) model
The tool 6 is designed for parcel information (lot/block/unit/township/range/section/quarter) name entity recognition (NER) for sentences filtered out from deed documents that might contain geographic or parcel information.
This tool loads filtered sentences of deed documents saved in `*.jsonl` files referring to the `*.txt` paths created with image_ids associated with each sentences in the `*.jsonl`. It supports identifying multiple parcel strings in an individual sentence if it contains many, and it leaves an empty list "[]" as the result if a sentence does not contain any. The input `*.jsonl` file format is exactly the same as the `*.jsonl` output from the tool 3, so please use tool 3 & 6 by sequence.

Tool 6 uses the exact same trained model as tool 4. The model is trained from strach based on spaCy NER pipeline. The training data were geographic and parcel related sentences identified with tool 2 from the deed document ocr text results of six MN/WI counties, and then transformed to `*.jsonl` format. The parcel entities were labeled with developer collected parcel keywords and expressions after observing the deed dataset. The trained NER model does not achieve expected performance since the parcel descriptions in actual deed documents have many variants and creating exact human labels is expensive, but it is already somewhat capable of achieving its helpfulness for locating parcel information. The model is saved at `./src/models/state_county_city_ner_model/model-best`. 

Here is an example scenario of applying the tool 6. Assuming we have already identified the sentences from the original ocr results with tool 3 and saved each sentence associated with its `image_ids` at `output/filtered_sentences_w_raw_txt_and_image_ids.jsonl`, then the user can declare use its root path as the "--file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl" and the model path `--model_path ./src/models/state_county_city_ner_model/model-best`. The model would recognize the parcel entities from each of the sentences and save the extracted string by only extending the original dictionary items (adding a `{"NERpredicted_LOT": ["lot_string"], "NERpredicted_BLOCK": ["block_string"], "NERpredicted_UNIT": ["unit_string"], "NERpredicted_TOWNSHIP": ["township_string"], "NERpredicted_RANGE": ["range_string"], "NERpredicted_SECTION": ["section_string"], "NERpredicted_QUARTER": ["quarter_string"]}` to each dictionary in the `*.jsonl`). 

#### 🔧 How to Use

1. **Prepare your data**
   - Use tool 3 to identify the geographic and parcel related sentences and save each sentence associate with its `image_ids` list. Place the `*.jsonl` result under the `output` or `data` folder.
   - Place the model under the `./src/models` directory.

2. **Run the identifier**
If you would like to recognize parcel information (lot/block/unit/township/range/section/quarter) from the identified sentences from the deeds that already included in the tool 3's `*.jsonl` output:
```bash
python -m scripts.run_parcel_ner_model \
    --file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl \
    --model_path ./src/models/state_county_city_ner_model/model-best \
    --output_path ./output/parcel_w_txt_n_image_ids.jsonl
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/parcel_w_txt_n_image_ids.jsonl`.
Each sentence and its ids and parcel information (lot/block/unit/township/range/section/quarter) are saved in this format:

`{"text": "... the Southwest Quarter (SW-1) of the Northeast Quarter (NE') of Section One (1), in Township Thirty-two (32) and Range Ten (10), and .... -lot One (1) in said ...;", "county": "mn-anoka-county", "image_ids": ["toy-book-a/page_001", "toy-book-a/page_002"], "NERpredicted_LOT": ["lot One (1)"], "NERpredicted_BLOCK": [], "NERpredicted_UNIT": [], "NERpredicted_TOWNSHIP": ["Township Thirty"], "NERpredicted_RANGE": ["Range Ten"], "NERpredicted_SECTION": ["Section One"], "NERpredicted_QUARTER": ["Southwest Quarter", "Northeast Quarter", "NE"]}`

`county` is copied from the Tool 3 input row (`null` for older files without it), and sentences are grouped by `county` + `image_ids` (see the note above Tool 3).

#### 🛠️ Arguments
- `--file_path`: The exact same format as the tool 3 output `*.jsonl` file
- `--model_path`: Load the same pretrained spaCy NER model for state/county/city to locate parcel information (lot/block/unit/township/range/section/quarter) in the document.
- `--output_path`: The output `*.jsonl` file to store the original filtered sentences associated with its `image_ids` list and recognized entities `"NERpredicted_LOT", "NERpredicted_BLOCK", "NERpredicted_UNIT", "NERpredicted_TOWNSHIP", "NERpredicted_RANGE", "NERpredicted_SECTION", "NERpredicted_QUARTER"`.

### Tool 7: Identify all geo and parcel information in deed sentences with trained named entity recognition (NER) model
The tool 7 jointly uses the trained NER models applied by tool 4, 5, 6 to identify all known categories of information (state/county/city/subdivision/lot/block/unit/township/range/section/quarter) for sentences filtered out from deed documents.
This tool loads filtered sentences of deed documents saved in `*.jsonl` files referring to the `*.txt` paths created with image_ids associated with each sentences in the `*.jsonl`. Its function is a combination of tool 4, 5, and 6. The input `*.jsonl` file format is exactly the same as the `*.jsonl` output from the tool 3, so please use tool 3 & 7 by sequence.

#### 🔧 How to Use

1. **Prepare your data**
   - Use tool 3 to identify the geographic and parcel related sentences and save each sentence associate with its `image_ids` list. Place the `*.jsonl` result under the `output` or `data` folder.
   - Place the models under the `./src/models` directory.

2. **Run the identifier**
If you would like to recognize all known categories of information (state/county/city/subdivision/lot/block/unit/township/range/section/quarter) from the identified sentences from the deeds that already included in the tool 3's `*.jsonl` output:
```bash
python -m scripts.run_identify_all_geo_parcel \
    --file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl \
    --model_aut_path ./src/models/state_county_city_ner_model/model-best \
    --model_subd_path ./src/models/subdivision_ner_model/model-best \
    --output_path ./output/all_geo_parcel_w_txt_n_image_ids.jsonl
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/all_geo_parcel_w_txt_n_image_ids.jsonl`.
Each sentence and its ids and all known categories of geographic and parcel information (state/county/city/subdivision/lot/block/unit/township/range/section/quarter) are saved in this format:

`{"text": ".... of the County of Sherburne and State of Minnesota ... Stdte of Minnesota., described as follows, to-wit: Lot One (1) of Fake Lake Park Addition, being part of Government Lot Two(2) ; of Section Thirty One (31), Township Thirty One (31), Range Twenty One (21) according to ... Deeds in and for said Sherburne County.", "county": "mn-sherburne-county", "image_ids": ["toy-book-a/page_001", "toy-book-a/page_002"], "NERpredicted_STATE": ["State of Minnesota", "Minnesota"], "NERpredicted_CNTY": ["County of Sherburne", "County of Sherburne", "Sherburne County"], "NERpredicted_CTY": [], "NERpredicted_SUBD": ["Fake Lake Park Addition"], "NERpredicted_LOT": ["Lot One (1)", "Lot Two(2)"], "NERpredicted_BLOCK": [], "NERpredicted_UNIT": [], "NERpredicted_TOWNSHIP": ["Township Thirty"], "NERpredicted_RANGE": ["Range Twenty"], "NERpredicted_SECTION": ["Section Thirty"], "NERpredicted_QUARTER": []}`

`county` is copied from the Tool 3 input row (`null` for older files without it), and sentences are grouped by `county` + `image_ids` (see the note above Tool 3).

#### 🛠️ Arguments
- `--file_path`: The exact same format as the tool 3 output `*.jsonl` file
- `--model_aut_path`: Load the same pretrained spaCy NER model for state/county/city to located administrative boundary and parcel information (state/county/city/lot/block/unit/township/range/section/quarter) in the document.
- `--model_subd_path`: Load the same pretrained spaCy NER model for subdivision in the document.
- `--output_path`: The output `*.jsonl` file to store the original filtered sentences associated with its `image_ids` list and recognized entities `"NERpredicted_STATE", "NERpredicted_CNTY", "NERpredicted_CTY", "NERpredicted_SUBD", "NERpredicted_LOT", "NERpredicted_BLOCK", "NERpredicted_UNIT", "NERpredicted_TOWNSHIP", "NERpredicted_RANGE", "NERpredicted_SECTION", "NERpredicted_QUARTER"`.

### Tool 8: Identify all geo and parcel information in deed sentences with trained named entity recognition (NER) model and save with entity indices in sentence strings. 
The tool 8 is almost the same logic as tool 7. The only difference is the format of storing the output.
Tool 7 only saves the identified entity strings, while tool 8 saves the entity as well as its start index and end index in the original sentence.
The input `*.jsonl` file format is exactly the same as the `*.jsonl` output from the tool 3, so please use tool 3 & 8 by sequence.

#### 🔧 How to Use

1. **Prepare your data**
   - Use tool 3 to identify the geographic and parcel related sentences and save each sentence associate with its `image_ids` list. Place the `*.jsonl` result under the `output` or `data` folder.
   - Place the models under the `./src/models` directory.

2. **Run the identifier**
If you would like to recognize all known categories of information (state/county/city/subdivision/lot/block/unit/township/range/section/quarter) as well as the indices of its appearance from the identified sentences from the deeds that already included in the tool 3's `*.jsonl` output:
```bash
python -m scripts.run_identify_all_geo_parcel_with_index \
    --file_path ./output/filtered_sentences_w_raw_txt_and_image_ids.jsonl \
    --model_aut_path ./src/models/state_county_city_ner_model/model-best \
    --model_subd_path ./src/models/subdivision_ner_model/model-best \
    --output_path ./output/all_geo_parcel_w_txt_n_image_ids_n_index.jsonl
```

3. **Output**
The script saves filtered geographic-relevant sentences to `output/all_geo_parcel_w_txt_n_image_ids_n_index.jsonl`.
Each sentence and its ids and all known categories of geographic and parcel information (state/county/city/subdivision/lot/block/unit/township/range/section/quarter) are saved in this format:

`{"text": "Lot Three (3) in Block Two (2), Kenth Park according to ... ", "county": "mn-anoka-county", "image_ids": ["toy-book-a/page_001", "toy-book-a/page_002"], "NERpredicted_STATE": [], "NERpredicted_CNTY": [], "NERpredicted_CTY": [], "NERpredicted_SUBD": [["Kenth Park", 32, 42]], "NERpredicted_LOT": [["Lot Three (3)", 0, 13]], "NERpredicted_BLOCK": [["Block Two (2)", 17, 30]], "NERpredicted_UNIT": [], "NERpredicted_TOWNSHIP": [], "NERpredicted_RANGE": [], "NERpredicted_SECTION": [], "NERpredicted_QUARTER": []}`

`county` is copied from the Tool 3 input row (`null` for older files without it), and sentences are grouped by `county` + `image_ids` (see the note above Tool 3).

#### 🛠️ Arguments
- `--file_path`: The exact same format as the tool 3 output `*.jsonl` file
- `--model_aut_path`: Load the same pretrained spaCy NER model for state/county/city to located administrative boundary and parcel information (state/county/city/lot/block/unit/township/range/section/quarter) in the document.
- `--model_subd_path`: Load the same pretrained spaCy NER model for subdivision in the document.
- `--output_path`: The output `*.jsonl` file to store the original filtered sentences associated with its `image_ids` list and recognized entities `"NERpredicted_STATE", "NERpredicted_CNTY", "NERpredicted_CTY", "NERpredicted_SUBD", "NERpredicted_LOT", "NERpredicted_BLOCK", "NERpredicted_UNIT", "NERpredicted_TOWNSHIP", "NERpredicted_RANGE", "NERpredicted_SECTION", "NERpredicted_QUARTER"`. Each entity is stored in a list which contains its entity string, start index, and end index.

### Tool 9: Combine the NER results of all sentences that belong to the same deed document (same county and image_ids)
The tool 9 is designed for combining extracted entities from different sentences by tool 7. Given locating the deed document to real-world location is a document-level task and all the extracted entities from multiple sentences from the document would jointly decide the location of the deed document. Thus, it might be necessary to develop this tool 9.
The input `*.jsonl` file format is exactly the same as the `*.jsonl` output from the tool 7, so please use tool 7 & 9 by sequence. Rows are merged by `county` + `image_ids`, so deeds from different counties that share the same county-relative `image_ids` stay separate records.

#### 🔧 How to Use

1. **Prepare your data**
   - Use tool 7 to generated the entities and save them associate with their corresponding `image_ids` list. Place the `*.jsonl` output from tool 7 under the `output` or `data` folder.
   - Only combine outputs that come from the same Tool 3 run made after the image_id and county changes (see the note above Tool 3). Mixing older and newer outputs splits one deed into separate groups: older rows without `county` are read as `"county": null` and never merge with rows that have a county.

2. **Run the identifier**
If you would like to combine all known categories of information (state/county/city/subdivision/lot/block/unit/township/range/section/quarter) from the same `county` and set of `image_ids` from the tool 7's `*.jsonl` output:
```bash
python -m scripts.run_combine_ner_results \
    --file_path ./output/all_geo_parcel_w_txt_n_image_ids.jsonl \
    --output_path ./output/combine_ner_results.jsonl
```

3. **Output**
The script saves combined ner results to `output/combine_ner_results.jsonl`.
Each unique `county` + `image_ids` and all known categories of geographic and parcel information (state/county/city/subdivision/lot/block/unit/township/range/section/quarter) identified from all different sentences of that individual document are saved in this format:

`{"county": "mn-anoka-county", "image_ids": ["toy-book-a/page_001", "toy-book-a/page_002"], "NERpredicted_STATE": ["Minnesota", "State of Minnesota", "State of Minnesota", "State of Minnesota"], "NERpredicted_CNTY": ["Anoka County", "County of Anoka", "County of Anoka", "County of Anoka", "Anoka County"], "NERpredicted_CTY": [], ..., "NERpredicted_TOWNSHIP": ["Township Thirty"], "NERpredicted_RANGE": ["Range Twenty"], "NERpredicted_SECTION": ["Section Thirteen"], "NERpredicted_QUARTER": ["Northeast Quarter", "Southwest Quarter", "Northeast Quarter", "Northeast Quarter"]}`

`county` is written once per record (it is not merged like the entity lists). Records built from older files without `county` have `"county": null`.

#### 🛠️ Arguments
- `--file_path`: The exact same format as the tool 7 output `*.jsonl` file
- `--output_path`: The output `*.jsonl` file to store each unique `county` + `image_ids` and all the recognized entities `"NERpredicted_STATE", "NERpredicted_CNTY", "NERpredicted_CTY", "NERpredicted_SUBD", "NERpredicted_LOT", "NERpredicted_BLOCK", "NERpredicted_UNIT", "NERpredicted_TOWNSHIP", "NERpredicted_RANGE", "NERpredicted_SECTION", "NERpredicted_QUARTER"` from all of its sentences. Each entity is stored in a list.
