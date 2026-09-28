# PROJECT_MAP.md — MP-text-to-land

A reading map of this repository, written from the README, the source files, and a few
read-only smoke checks (2026-09-25). Writing the map changed no source code; later source
fixes are noted where they apply (e.g. the Tool 3 image_id fix, commit `68a61cb`). Items
marked *(verified)* were confirmed by running code in the local `mapprejudice/` venv;
everything else comes from reading the code.

---

## 1. Purpose

MP-text-to-land pulls **geographic and parcel entities out of OCR text of historical
property deeds** for the Mapping Prejudice project (racial-covenant research). Those
entities are state, county, city, subdivision/addition, lot, block, unit,
township, range, section and quarter. The goal is output shaped like the project's
volunteer-labelled `*.geojson` attributes, where each record is keyed by the deed's
`image_ids`. That output can either stand in for volunteer transcription or pre-fill
prompts for crowdsourcing.

The pipeline has two stages:

1. **Sentence filtering.** Split OCR text into sentences with spaCy, then keep the ones
   that fuzzy-match keywords or volunteer-labelled GeoJSON values (RapidFuzz).
2. **NER.** Run two spaCy NER models that ship in this repo over the kept sentences,
   then merge the results per deed.

---

## 2. Repository layout

```
MP-text-to-land/
├── .env                      # local paths; git-ignored, each user creates their own (§6)
├── .env.example              # template for .env with the required variable names
├── .gitignore
├── README.md                 # user guide for Tools 1–9
├── requirements.txt          # pipdeptree-style freeze from a Linux/conda env; may fail on macOS (see §7)
├── requirements-macos.txt    # clean direct-dependency pins verified on macOS (Python 3.11.9)
├── requirements-mac-working.txt   # git-ignored; full local `pip freeze` of the venv
├── mapprejudice/             # git-ignored 1.3 GB local Python 3.11.9 venv
├── data/
│   ├── test_keyword_label.txt     # 7 short parcel words (township, range, twp, ...)
│   └── keywords/
│       └── not_using/             # suggested keyword lists, NEVER loaded (see §3.3)
├── dev/                      # aggregate-only local checks, e.g. check_county_id_collisions.py (counts only)
├── scripts/                  # CLI entry points = "Tools" 1–9 (run with python -m)
├── src/
│   ├── __init__.py           # eagerly imports every submodule (side effects!)
│   ├── data_preprocessing/   # GeoJSON loading, path building, sentence filtering, grouping
│   ├── utils/                # .env loader, keyword loader, shapefile→txt extractor
│   ├── training/             # subdivision NER fine-tuning script (not a CLI tool)
│   └── models/
│       ├── state_county_city_ner_model/model-best   # 6.4 MB, spaCy 3.7.x
│       └── subdivision_ner_model/model-best         # 53 MB, fine-tuned en_core_web_md 3.7.1
└── test/                     # unittest toy-data tests (test_tool3_image_ids, test_pipeline_tool3_7_9,
                              #   test_county_collisions, test_dev_collision_script) + older manual scripts (see §8)
```

Git-ignored but referenced in code: `/output/`, `/data/raw/`, `/data/processed/`,
`/data/spacy_ner_subdivision_labels/`, `/data/subdivision_sample/`.

---

## 3. Expected data inputs

### 3.1 OCR `*.txt` files

- One text file per scanned page image, laid out the same way as the S3 bucket
  `covenants-deed-images/ocr/txt/<county-folder>/<...>/<image_id>.txt`.
- Two ways to find them:
  - **By directory walk** (Tools 1 and 3): `--root_path` is walked recursively with
    `rglob("*.txt")` or `os.walk`.
  - **By GeoJSON `image_ids`** (Tool 2): paths are built as
    `f"{OCRTXT_PATH}{county_folder}/{image_id}.txt"` in
    `src/data_preprocessing/path_generator.py:17`. This is plain string concatenation, so
    **`OCRTXT_PATH` must end in `/`**.
- Pages of one deed are recognised by filename. Tool 3 groups files whose basename is
  the same **up to the last `_`** (`group_files_by_prefix`); for example,
  `page_001` and `page_002` form one deed.
- Tools 1–3 (the ones that read OCR text) collapse whitespace (`re.sub(r"\s+", " ", ...)`) and joins a deed's pages
  with `\n`.

### 3.2 GeoJSON files (only Tool 2, the manual tests, and training-data prep use these)

- Layout: `GEOJSON_PATH/<county-folder>/*.geojson`, joined with `os.path.join`, so a
  trailing slash doesn't matter here.
- Required property: `image_ids`. It is a *string* that looks like a Python list, e.g.
  `"['123/abc_001', '123/abc_002']"`, and is parsed with
  `json.loads(x.replace("'", '"'))`.
- The loader keeps these columns by default:
  `image_ids, saved_path, deed_date, seller, buyer, street_add, cnty_name, city, state,
  cov_text, add_cov, lot_cov, block_cov, add_mod, block_mod, lot_mod, ph_dsc_mod, geometry,
  zip_code, cnty_pin, cnty_fips, workflow, doc_num, zn_subj_id, zn_dt_ret, cov_type,
  med_score, manual_cx, match_type, plat, dt_updated`.
  Tool 2 overrides this with a shorter list through `--columns_to_keep`.
- `saved_path` is a derived column (a list of local txt paths), not one read from the file.
- Known county folders: `mn-anoka-county, mn-dakota-county, mn-olmsted-county,
  mn-sherburne-county, mn-washington-county, wi-milwaukee-county`. The local `.env` lists
  9 folders.
- `mn-dakota-county` has a hard-coded fix-up for three specific `image_ids`
  (`geojson_loader.py:11-24`).

### 3.3 Keyword files

- `load_keywords_from_txt_directory("data/keywords/")` reads **every `*.txt` directly
  inside `data/keywords/`** (not recursively), one keyword per line.
- `data/keywords/not_using/` is never read. It stores the suggested lists: `geo_general`,
  `parcel_general`, `states_keywords`, `mn_county`, `mn_city`, `wi_county`, `wi_city`,
  `subd_keywords`, `subdivision_keywords`, `removed_keywords`, `test_keyword_label`.
- **Current state (verified): `data/keywords/` contains no `*.txt` files, so 0 keywords
  load.** Keyword mode in Tools 1–3 therefore writes nothing until someone copies lists up
  a level. (`data/test_keyword_label.txt` sits in `data/`, not `data/keywords/`, so it
  isn't loaded either.)
- Keywords are loaded **once, at import time**, into a module-level `KEYWORDS_LIST`
  (`sentence_identifier.py:5`), using a path relative to the working directory.

### 3.4 Model folders

| Folder | What it really is | Labels the scripts use |
|---|---|---|
| `src/models/state_county_city_ner_model/model-best` | Small `tok2vec + ner` pipeline trained with `spacy train` (its config points to `output/try003train.spacy`). Despite the name, it's **multi-label**: `BLOCK, CITY, COUNTY, LOT, QUARTER, RANGE, SECTION, STATE, SUBDIVISION, TOWNSHIP, UNIT`. meta.json reports ents_f of about 0.99. | STATE/COUNTY/CITY (Tool 4), the parcel labels (Tool 6), all but SUBDIVISION (Tools 7/8) |
| `src/models/subdivision_ner_model/model-best` | Full `en_core_web_md` 3.7.1 pipeline (tagger, parser, lemmatizer, ...) with a `SUBDIVISION` label added to NER. It keeps the stock OntoNotes labels (GPE, ORG, ...). The performance block in meta.json is **stock en_core_web_md numbers** and has no SUBDIVISION score. | SUBDIVISION only (Tools 5/7/8) |

- Both need **spaCy `>=3.7.x,<3.8.0`**; the local venv has spaCy 3.7.5 and thinc 8.2.5.
- *(verified)* The state/county/city model emits **one entity per token**. For
  "Lot Three (3)" it returns five separate `LOT` entities. The scripts rely on a
  merge-adjacent-spans heuristic to rebuild phrases (see §7). The subdivision model returns
  whole spans.
- Tools 1–3 also need a stock spaCy model for sentence splitting: `en_core_web_sm` by
  default (`en_core_web_md` for Tool 3). The venv has `sm` and `lg`, **not `md`**, so
  Tool 3's default fails there.

---

## 4. Scripts under `scripts/` (Tools 1–9)

Run each one from the repo root as a module, e.g. `python -m scripts.<name>`.
"Needs .env" means the script imports `src`, which reads `.env` at import time (§6).

| Tool | Script | Input → Output | Needs .env |
|---|---|---|---|
| 1 | `run_sentence_identifier_w_raw_txt.py` | walk OCR txt under `--root_path` → keyword-matched sentences as plain **`.txt`**, one per line | yes (only for import; values unused) |
| 2 | `run_sentence_identifier_w_geojson.py` | GeoJSON rows → their OCR pages → sentences matching GeoJSON attribute values (`--entity_columns`) and/or keywords → plain **`.txt`** | yes (`GEOJSON_PATH`, `OCRTXT_PATH`) |
| 3 | `run_identify_sentences_with_imageids_f_raw_txt_to_jsonl.py` | walk OCR txt, group pages into deeds, keyword-filter → **`.jsonl`** `{"text", "county", "image_ids"}` | yes (only for import) |
| 4 | `run_state_county_city_ner_model.py` | Tool 3 jsonl → adds `NERpredicted_STATE/CNTY/CTY`; keeps `county` | no |
| 5 | `run_subd_ner_model.py` | Tool 3 jsonl → adds `NERpredicted_SUBD`; keeps `county` | no |
| 6 | `run_parcel_ner_model.py` | Tool 3 jsonl → adds `NERpredicted_LOT/BLOCK/UNIT/TOWNSHIP/RANGE/SECTION/QUARTER` (uses the state/county/city model); keeps `county` | no |
| 7 | `run_identify_all_geo_parcel.py` | Tool 3 jsonl → all 11 `NERpredicted_*` fields as lists of strings, using both models; keeps `county` | no |
| 8 | `run_identify_all_geo_parcel_with_index.py` | same as Tool 7, but each entity is `[text, start_char, end_char]`; keeps `county` | no |
| 9 | `run_combine_ner_results.py` | Tool 7 jsonl → one row per unique (`county`, `image_ids`) with every sentence's entity lists concatenated (`text` dropped, `county` written once) | no |

Details per script:

- **Tool 1 (`run_sentence_identifier_w_raw_txt`)**: Each txt file is processed on its own
  (no page grouping). It calls `filter_relevant_sentences` with an empty entity dict and
  keeps a sentence when the number of keyword matches is `> --item_threshold`. The output
  loses track of which file each sentence came from. The default `--root_path` is the
  placeholder `/.../covenants-deed-images/ocr/txt/`.
- **Tool 2 (`run_sentence_identifier_w_geojson`)**: `load_geojson_to_gdf(--counties,
  --columns_to_keep)` loads the rows. For each row it reads every path in `saved_path`,
  builds `entity_dict = {col: row[col]}` from `--entity_columns`, and keeps a sentence when
  entity matches plus keyword matches `> --item_threshold`. Only string attribute values
  are kept. The README says Tool 2's output became NER training data.
- **Tool 3 (`run_identify_sentences_with_imageids_f_raw_txt_to_jsonl`)**:
  `group_files_by_prefix(root_path)` builds page groups. Each group is concatenated and
  filtered by keyword, and each kept sentence is written with
  `image_ids = [county_relative_image_id(p, root_path) for p in group]`: each path relative
  to its county folder (the first folder under `--root_path`), without `.txt`, so it
  matches GeoJSON `image_ids`. `--root_path` must therefore be the folder that contains the
  county folders, e.g. `OCRTXT_PATH`. Before commit `68a61cb` it used `split('/', 10)`
  (§7, item 13). Each row also carries `county`, the first folder under `--root_path`
  (`null` for a file directly under it). Page groups that span counties are split by
  county, so a row never mixes counties (§7, item 27). Its default output name
  `all_sentences_w_raw_txt_and_image_ids.jsonl` differs from the
  `filtered_sentences_...` name the README and Tools 4/5/6/8 expect.
- **Tools 4–8**: Each loads the jsonl and regroups sentences by (`county`, `image_ids`),
  with `county` `null` for old rows that lack it, and writes `county` in every row. It
  writes `image_ids` **sorted** and emits one line per sentence. Output order is
  group-first, so it can differ from input order. None of them re-read the OCR txt, even
  though the README suggests they do.
- **Tool 7 vs. Tool 8**: Tool 8 applies the adjacent-span merge to SUBDIVISION as well;
  Tool 7 doesn't. Tool 7 defaults to Tool 3's real default output name; Tool 8 defaults to
  the README name.
- **Tool 9 (`run_combine_ner_results`)**: `merge_dictionaries` concatenates list fields
  for rows that share the same (`county`, `image_ids`) key. `county` itself is written
  once per record, never concatenated, and is `null` for old rows. Duplicates are kept on purpose, since repeated
  mentions act as evidence. It imports spaCy but doesn't use it.

### Non-CLI modules the tools depend on

| Module | Role |
|---|---|
| `src/__init__.py` | Imports every submodule eagerly, so any `from src...` import runs `load_config()`, loads keywords, and imports geopandas. Sets `logging.basicConfig(INFO)`. |
| `src/utils/config_loader.py` | `load_config()` returns `(GEOJSON_PATH, FOLDER_NAMES.split(','), OCRTXT_PATH, S3_PATH)`. `load_openai_config()` reads `OPENAI_API_KEY`; nothing calls it. |
| `src/utils/keyword_loader.py` | Reads keyword `*.txt` files (§3.3). |
| `src/utils/extract_shp_attribute_to_txt.py` | Walks shapefiles (default `data/subdivision_sample/`) and writes the `PLAT/LEGAL_NAME/Name/PLATNAME` values to `output/subdivision_names_<shp>.txt`. Probably used to build subdivision keyword lists. No script calls it. |
| `src/data_preprocessing/geojson_loader.py` | `load_geojson_to_gdf()` concatenates the county GeoJSONs, parses `image_ids`, applies the Dakota fix, and adds `saved_path`. |
| `src/data_preprocessing/path_generator.py` | Builds S3 paths, local paths, and `aws s3 cp` commands (hard-coded `./aws/local/aws-cli/v2/2.26.0/bin/aws`) for downloading OCR text. `save_commands_as_txt` writes a de-duplicated command list. Only `test/test_path_generator.py` uses it. |
| `src/data_preprocessing/sentence_identifier.py` | The core filter: `normalize_text`, `match_entity_in_sentence` (`fuzz.partial_ratio` on the whole sentence), `match_keywords_in_sentence` (`fuzz.ratio` per **single word**), `classify_sentence`, and `filter_relevant_sentences` (iterates `doc.sents` and returns GEO sentences only). Also has `match_entity_in_sentence_for_subd_label`, a span-locating matcher with leftover debug `print`s. |
| `src/data_preprocessing/passage_identifier.py` | Near-copy of `sentence_identifier`. `filter_relevant_sentences2` classifies the whole document at once and uses the span-locating subdivision matcher. It looks like the tool used to generate subdivision NER labels. No script calls it. |
| `src/data_preprocessing/generate_image_ids_list_from_filenames.py` | `group_files_by_prefix(root)` (§3.1, §7). |
| `src/training/subdivision_ner_training.py` | Fine-tunes `en_core_web_md` NER on jsonl `{"text", "entities": [[s, e, label]]}` for 20 epochs and saves the model with the best dev loss. Train, dev, and output paths are `/root/path/...` placeholders. |

---

## 5. Order of running the tools

```
 (optional, once)  S3 ──aws s3 cp commands (path_generator / test_path_generator)──▶ local OCR txt
 (optional, once)  shapefiles ──extract_shp_attribute_to_txt──▶ subdivision keyword lists
 (setup)           copy chosen lists from data/keywords/not_using/ ─▶ data/keywords/   ← REQUIRED for keyword mode

 Main pipeline (deeds with no volunteer labels):

   OCR txt ──Tool 3──▶ filtered_sentences_w_raw_txt_and_image_ids.jsonl
                          │
             ┌────────────┼──────────────┬──────────────┐
           Tool 4       Tool 5         Tool 6     Tool 7 ──▶ all_geo_parcel_w_txt_n_image_ids.jsonl ──Tool 9──▶ combine_ner_results.jsonl
         (st/cnty/cty)  (subd)        (parcel)    Tool 8 ──▶ ..._n_index.jsonl   (same as 7, plus char offsets)

 Side tools:
   Tool 1  quick keyword look at raw OCR, writes a .txt with no ids
   Tool 2  labelled deeds (GeoJSON) → sentences that mention volunteer values (used to build training data)
   training/subdivision_ner_training.py  offline, retrains the subdivision model
```

In practice:

1. Set up the environment: on macOS, `pip install -r requirements-macos.txt`. It includes
   spaCy 3.7.5 and `en_core_web_sm`; add `en_core_web_md` if you keep Tool 3's default.
2. Create `.env` from the template (`cp .env.example .env`, then fill in your paths; see §6)
   and run everything **from the repo root**.
3. Put keyword lists in `data/keywords/`.
4. Run Tool 3 with `--root_path` set to the folder that **contains the county folders**
   (`OCRTXT_PATH`) and **with an explicit `--output_path`** so later tools can find the
   file. If you have Tool 3–9 outputs from before commit `68a61cb`, don't mix them with
   new ones: rerun Tool 3 → 7 → 9 from scratch. The same applies to outputs from before
  commit `55003a0`, which have no `county` field: rerun from Tool 3 onward.
5. Run Tool 7 (or 8), then Tool 9. Tools 4, 5 and 6 are single-category subsets of Tool 7.

README commands fixed in the documentation cleanup (every command now names an
existing script and uses only flags that script defines):

- Tool 1: added the missing `\` after `--root_path ...`.
- Tool 3: `scripts.identify_sentences_f_raw_txt_to_jsonl`, which doesn't exist, became
  `scripts.run_identify_sentences_with_imageids_f_raw_txt_to_jsonl`; added the missing `\`.
- Tool 9: `scripts.run_identify_all_geo_parcel_with_index` became
  `scripts.run_combine_ner_results`, and its heading no longer copies Tool 8's.

---

## 6. Required `.env` variables

`.env` was previously tracked in git. It was removed from Git tracking in commit
`c5a148f` and is git-ignored, so each user should copy `.env.example` to `.env` and fill
in local paths:

```bash
cp .env.example .env   # then edit the paths
```

`load_dotenv()` in `src/utils/config_loader.py` reads these variables:

| Variable | Required? | Used by | How |
|---|---|---|---|
| `FOLDER_NAMES` | **Yes, for any import of `src`** | `geojson_loader` (default county list), `test_path_generator` | Split on `,`, with no stripping, so `"a, b"` produces `" b"`. **If it's missing, `import src` crashes with `AttributeError: 'NoneType' object has no attribute 'split'` (verified).** |
| `GEOJSON_PATH` | Tool 2 and the tests | `geojson_loader.py:38` | `os.path.join(GEOJSON_PATH, county)` |
| `OCRTXT_PATH` | Tool 2 and the tests | `path_generator.py` | `f"{OCRTXT_PATH}{county}/{image_id}.txt"`. **Needs a trailing `/`.** |
| `S3_PATH` | Optional | `path_generator.create_s3_paths` / `create_command_paths` | `f"{S3_PATH}{county}/{image_id}.txt"`. Needs a trailing `/`. |
| `SHP_PATH` | Unused | (commented out in `config_loader`; not in `.env.example`) | none |
| `OPENAI_API_KEY` | Unused | `load_openai_config()` (never called) | none |

- Tools 1 and 3 need `FOLDER_NAMES` only so the import succeeds; they take paths from
  `--root_path`. Tools 4–9 don't read `.env`.
- Every path value in the local `.env` currently ends in `/`, and `FOLDER_NAMES` lists 9
  folders (verified).
- `.env` is found relative to how Python is started. Running from the repo root works;
  launching from another directory with `PYTHONPATH` set failed to find it (verified).

---

## 7. Fragile points and assumptions

### Environment and packaging
1. **`.env` was previously tracked.** It was removed from Git tracking in commit
   `c5a148f` and is git-ignored. Users should copy `.env.example` to `.env`. Two
   consequences:
   - Commits before `c5a148f` still contain the old `.env`, which held only local data
     paths, no secrets.
   - Pulling `c5a148f` **deletes an existing `.env`** from other checkouts, so those
     users need to recreate it from `.env.example`.

   The five `src/**/__pycache__/*.cpython-310.pyc` files that were committed were also
   untracked in `c5a148f`.
2. **Local-only files are ignored**: the `mapprejudice/` venv,
   `requirements-mac-working.txt`, `__pycache__/` and `*.pyc` (verified with
   `git check-ignore`).
3. **`requirements.txt` won't install as written.** It is a pipdeptree-style tree with
   indented duplicates and 9 `@ file:///croot/...` conda-build wheels built for Linux
   x86_64 cp310. It also lacks **python-dotenv**, **torch** (imported by the training
   script and not installed locally), and the `en_core_web_sm` wheel the README relies on.
   It is kept for reference. `requirements-macos.txt` pins only the direct dependencies,
   at the versions in the working macOS venv (Python 3.11.9); the README install section
   explains which file to use. It hasn't yet been tested in a fresh venv.
4. **Import-time side effects.** `import src` loads `.env`, reads `data/keywords/`
   relative to the working directory, imports geopandas, and sets global logging, even for
   Tools 1 and 3, which need none of that. Running from outside the repo root breaks it.

### Sentence filtering (Tools 1–3)
5. **No keywords are active (verified: 0 loaded).** With 0 keywords and
   `len(matches) > item_threshold`, Tools 1 and 3 write empty files and don't warn.
6. **About 465 multi-word keywords can never match.** `match_keywords_in_sentence`
   compares each keyword with *single words* using `fuzz.ratio`, so entries like
   `albert lea` or `city of` never reach the threshold. That includes 222 of the entries in
   `mn_city.txt` and 154 in `wi_city.txt`.
7. **Short keywords are noisy.** For example `st`, `tn`, `cty`: `states_keywords.txt` has
   75 entries of three characters or fewer. Blank lines become `""` keywords.
8. **`--item_threshold` is strict `>`.** The README calls it the "minimum number", which
   reads as `>=`, so users can be off by one. `0` means "at least one match".
9. **`keyword_set` / `KEYWORDS_LIST` are fixed at import.** `classify_sentence` always
   uses the global list, so keywords can't be passed in by the caller or by a test.
10. **Sentence splitting depends on the spaCy model.** The README says the model choice
    "does not affect the output because all of these would call the spaCy sentencizer".
    But `en_core_web_*` split sentences with the dependency parser (`senter` is disabled),
    so sm, md, lg and trf can produce different sentences. This needs checking.
11. **spaCy's `nlp.max_length` is 1,000,000 characters.** Tool 3 concatenates all pages in
    a group, so a very large group (see item 12) could raise `ValueError`.
12. **Page grouping is keyed by basename prefix only (verified).** `a/doc_1.txt` and
    `b/doc_2.txt` in *different folders or counties* merge into one "deed". Page order is
    lexicographic, so `_10` sorts before `_2`. Any filename with an `_` in it is treated as
    a paged document; `toy_book_page_003`, for example, is grouped with every other
    `toy_book_page_*` file. Since `55003a0`, Tool 3 splits such groups by county, so pages
    from different counties are never merged. Groups spanning several folders *within*
    one county are unchanged; an aggregate-only check found 1 such group in the current
    data. That remains a separate RQ.
13. **Fixed in `68a61cb`: Tool 3 `image_ids` used to depend on path depth.** The old rule,
    `file_path.split('/', 10)[-1]`, kept everything after the 10th `/` of the absolute
    path, so it matched GeoJSON ids only at one depth. With the real `OCRTXT_PATH` it
    dropped the first folder under each county, matching 0 of 5,034 GeoJSON ids. Tool 3
    now uses ids relative to the county folder, which reproduce all 5,034 *(verified,
    read-only)*. `test/test_tool3_image_ids.py` covers several depths, a trailing `/`,
    and relative roots. What's left:
    - `--root_path` must be the folder that contains the county folders. Pointed at a
      single county folder, every id loses its first subfolder, and nothing warns you.
    - Tool 3–9 outputs from before the fix have different ids. Don't mix them with new
      ones; rerun Tool 3 → 7 → 9 from scratch.
14. **Speed.** Keyword matching is O(#keywords × #words) per sentence. With all city lists
    enabled that is about 2,300 keywords. `get_fuzzy_match_index` is O(len(text) × len(phrase)).

### GeoJSON loading (Tool 2)
15. **One bad file stops all loading.** `load_geojson_to_gdf` returns `None` on the first
    exception, and Tool 2 then crashes on `gdf[...]`.
16. **`columns_to_keep` is hard-coded.** If any county lacks one of the columns, a
    `KeyError` follows. If no county folder exists, the empty frame raises a `KeyError`
    as well.
17. **`image_ids` parsing (`replace("'", '"')` then `json.loads`) breaks on ids that
    contain apostrophes or quotes.**
18. The Dakota County fix covers three specific ids. The same logic is copied in
    `test_path_generator.py`.
19. The `list_of_county_folder=folder_names` default is fixed when the module is imported.

### NER stage (Tools 4–9)
20. **Rebuilding phrases from token-level entities is heuristic.** Two same-label entities
    are merged when the gap between them is 0 or 1 character. Two separate mentions next to
    each other, like `Lot 1 Lot 2`, collapse into one; any wider gap (a double space, or
    OCR noise) splits a phrase. This only works because Tool 3 collapses whitespace first.
21. `ent.text.strip()` is paired with the *unstripped* `start_char`/`end_char` in Tool 8,
    so if an entity has edge whitespace, its text and offsets disagree.
22. Tools 4–8 write `image_ids` **sorted**, and grouping uses the unsorted tuple. The same
    set in a different order becomes a separate group, and output order differs from input.
23. The `SUBDIVISION` label from the state/county/city model is discarded. The
    subdivision model's other labels (GPE, ORG, ...) are also ignored. Choosing which model
    handles which label is a design decision nobody has written down.
24. Tool 9 on Tool 8 output concatenates `[text, start, end]` triples from different
    sentences, so the offsets lose their meaning.
25. Output files are opened with the default encoding plus `ensure_ascii=False`, and there
    are no context managers. A crash in the middle leaves partial files.
26. `os.makedirs(os.path.dirname(output_path))` fails when `--output_path` has no
    directory part (e.g. `out.jsonl`), because `dirname` is `""`.
27. **Fixed in `55003a0`: records were identified by `image_ids` alone.** `image_ids` are
    county-relative, so the same id in two counties merged into one Tool 7/9 record, and
    Tool 3 could build one mixed-county deed. Every Tool 3–9 row now carries `county`.
    Tools 4–8 group, and Tool 9 merges, by (`county`, `image_ids`). Old rows without
    `county` are read as `null` and never merge with rows that have one. The toy tests in
    `test/test_county_collisions.py` cover Tools 3–9. An aggregate-only check
    (`dev/check_county_id_collisions.py`, counts only) found no cross-county collisions in
    the current data: 0 page groups mixing more than one county, and 0 county-relative ids
    present in more than one county. So this prevents future silent cross-county merges
    rather than changing today's records. What's left:
    - `county` is the first folder under `--root_path`. Pointed at a single county folder,
      `county` becomes that county's first subfolder, and nothing warns you.
    - Tool 3–9 outputs from before `55003a0` lack `county`. Don't mix them with new ones;
      rerun from Tool 3 onward.

### Training script
28. `save_model` writes to the literal folder `"{output_dir}/model-last"` because the
    f-string prefix is missing.
29. `evaluate_model` calls `nlp.update(...)` on the dev set. In spaCy 3 that seems to
    **train on validation data** with a default optimizer, so the "best val loss" choice is
    biased. This needs checking.
30. It relies on the globals `nlp` and `output_dir`, uses placeholder paths, and imports
    `torch` only to print the device.
31. The README says both models were "trained from scratch", but the subdivision model is
    a fine-tuned `en_core_web_md`. Its meta.json metrics are the stock ones, so there is no
    recorded SUBDIVISION score.

---

## 8. What to test before changing anything

Automated `unittest` tests use only synthetic toy files in temp directories, never real
data. Run them all with `python -m unittest discover -s test` (35 tests; one opt-in test
with the bundled models runs only when `MP_RUN_MODEL_TESTS=1` is set):
- `test_tool3_image_ids.py`: Tool 3 image_ids at several root depths (item 13).
- `test_pipeline_tool3_7_9.py`: Tool 3 → 7 → 9 end to end, via each tool's real `main()`.
- `test_county_collisions.py`: `county` in Tools 3–9, grouping and merging by
  (`county`, `image_ids`), old rows, mixed old/new files (item 27).
- `test_dev_collision_script.py`: the aggregate-only checker runs on the standard library
  alone and prints counts only.

The other `test/*.py` files are manual `main()`
scripts that print output and contain no assertions. They need the real GeoJSON and OCR
data, and one of them writes to `/home/yaoyi/jiao0052/...`. They contain no test cases,
and importing them triggers `load_config()`.

Before changing code, set up a **golden baseline** plus some small unit tests.

### A. Environment smoke checks
- [ ] Fresh venv from `requirements-macos.txt`: `python -c "import src"` succeeds from
      the repo root.
- [ ] Both models load under spaCy 3.7.5: `spacy.load(".../model-best")`.
- [ ] Record current behaviour when `.env` / `FOLDER_NAMES` is missing (it crashes today).

### B. Unit tests with synthetic fixtures (no real data)
- [ ] `load_keywords_from_txt_directory`: loads only top-level `*.txt`, skips `not_using/`,
      and handles blank lines.
- [ ] `match_keywords_in_sentence`: single-word hits, the threshold boundary (89/90/91),
      and **pin down the current multi-word behaviour** (it never matches).
- [ ] `match_entity_in_sentence`: exact vs. fuzzy match; non-string or NaN values skipped.
- [ ] `filter_relevant_sentences`: returns only GEO sentences. Pin the `> item_threshold`
      semantics in the calling scripts.
- [ ] `group_files_by_prefix`: same-dir pages, **cross-dir basename collision**, `_10`
      vs `_2` ordering, and names with no `_`.
- [x] Tool 3 image_id derivation at several root depths, trailing `/`, and relative
      roots: `test/test_tool3_image_ids.py` (item 13).
- [ ] `path_generator`: trailing-slash assumption on `OCRTXT_PATH` / `S3_PATH`.
- [ ] `load_geojson_to_gdf` on a tiny 2–3-feature GeoJSON: string→list parsing of
      `image_ids`, empty or invalid ids dropped, `saved_path` built, a missing column, one
      bad file among good ones, and the Dakota fix.
- [ ] The adjacent-span merge in `extract_state_cnty_cty_kv`, on hand-built fake entity
      tuples: gap 0, gap 1, gap 2, and two adjacent separate mentions.
- [ ] `merge_dictionaries` (Tool 9): same ids merged, duplicates kept, `text` dropped,
      distinct ids kept apart.
- [x] `county` through Tools 3–9: same id in two counties stays separate, old rows get
      `null`, mixed old/new rows don't merge: `test/test_county_collisions.py` (item 27).

### C. End-to-end regression (golden files)
- [ ] Build a small fixture tree, e.g. `test/fixtures/ocr/txt/mn-anoka-county/...`, with a
      handful of multi-page deeds and a copied keyword list.
- [ ] Run Tool 3 → Tools 4, 5, 6, 7, 8 → Tool 9 with explicit paths and save every output
      as a **golden file**.
- [ ] After any change, diff the new outputs against the golden files. The jsonl key set,
      the key order, and the `image_ids` format are the contract between tools, and any
      downstream GeoJSON join depends on them.
- [ ] Check the NER output on a few sentences. The README examples ("Lot Three (3) in
      Block Two (2), Kenth Park ...") work well. Accept small differences only when they
      are expected, and pin spaCy, the models, and the Python version for these tests.

### D. Before touching behaviour specifically
- Changing the keyword matching, sentence splitter, or thresholds changes **which sentences
  reach NER**. Compare sentence counts and keyword-hit statistics on a real county sample
  before and after.
- Changing page grouping or `image_ids` derivation affects **how results join back to
  GeoJSON**. Check the join rate against `image_ids` in the real GeoJSON.
- Changing the NER post-processing (the merge heuristic or label routing) affects
  **entity strings**. Compare per-label counts and a manually reviewed sample.
