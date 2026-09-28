"""
Shared helpers for running Tool 3 -> Tool 7 -> Tool 9 on synthetic toy data.

Not a test module (unittest discovery only collects test*.py). Uses only toy files written
to temporary directories; no Mapping Prejudice data. Each tool's real main() is run with
only these stubbed:
  * Tool 3's keyword filter: data/keywords/ may be empty, so each page becomes one sentence.
  * spaCy model loading: Tool 7 gets tiny deterministic fake NER models unless
    real_models=True, which loads the bundled models from src/models/.
"""

import json
import os
import re
import sys
import types
from unittest import mock

# Importing `src` (via Tool 3) reads these at import time. Toy values only; load_dotenv()
# does not override variables that are already set, so a real .env is never used.
os.environ["FOLDER_NAMES"] = "mn-toy-county,wi-toy-county"
os.environ["GEOJSON_PATH"] = "/nonexistent/geojson/"
os.environ["OCRTXT_PATH"] = "/nonexistent/ocr/txt/"
os.environ["S3_PATH"] = "s3://nonexistent/ocr/txt/"

from scripts import run_identify_sentences_with_imageids_f_raw_txt_to_jsonl as tool3  # noqa: E402
from scripts import run_identify_all_geo_parcel as tool7  # noqa: E402
from scripts import run_combine_ner_results as tool9  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUT_MODEL_PATH = os.path.join(REPO_ROOT, "src/models/state_county_city_ner_model/model-best")
SUBD_MODEL_PATH = os.path.join(REPO_ROOT, "src/models/subdivision_ner_model/model-best")

NER_KEYS = ["NERpredicted_STATE", "NERpredicted_CNTY", "NERpredicted_CTY", "NERpredicted_SUBD",
            "NERpredicted_LOT", "NERpredicted_BLOCK", "NERpredicted_UNIT", "NERpredicted_TOWNSHIP",
            "NERpredicted_RANGE", "NERpredicted_SECTION", "NERpredicted_QUARTER"]


def write_tree(root, pages):
    """Write {relative path without .txt: text} as OCR txt files under root."""
    for rel, content in pages.items():
        path = os.path.join(root, rel + ".txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def fake_filter_relevant_sentences(text, **kwargs):
    """Stand-in for Tool 3's keyword filter: each page's text is one 'matched' sentence."""
    return [{"sentence": s.strip(), "keyword_matches": [("lot", "lot", 100)]}
            for s in text.split("\n") if s.strip()]


def fake_nlp(patterns):
    """A minimal stand-in for a spaCy pipeline: regex matches become entities."""
    def nlp(text):
        ents = [types.SimpleNamespace(label_=label, text=m.group(0), start_char=m.start(), end_char=m.end())
                for label, pat in patterns for m in re.finditer(pat, text)]
        return types.SimpleNamespace(ents=sorted(ents, key=lambda e: e.start_char))
    return nlp


FAKE_AUT_MODEL = fake_nlp([("LOT", r"Lot \d+"), ("BLOCK", r"Block \d+")])
FAKE_SUBD_MODEL = fake_nlp([("SUBDIVISION", r"Toy [A-Z][a-z]+")])


def run_tool3(root_path, out_path):
    with mock.patch.object(tool3, "filter_relevant_sentences", fake_filter_relevant_sentences), \
            mock.patch.object(tool3.spacy, "load", return_value=None), \
            mock.patch.object(sys, "argv", ["tool3", "--root_path", root_path, "--output_path", out_path]):
        tool3.main()
    return read_jsonl(out_path)


def run_tool7(in_path, out_path, real_models=False):
    argv = ["tool7", "--file_path", in_path, "--output_path", out_path]
    if real_models:
        with mock.patch.object(sys, "argv", argv + ["--model_aut_path", AUT_MODEL_PATH,
                                                    "--model_subd_path", SUBD_MODEL_PATH]):
            tool7.main()
    else:
        models = {"fake-aut": FAKE_AUT_MODEL, "fake-subd": FAKE_SUBD_MODEL}
        with mock.patch.object(tool7.spacy, "load", side_effect=models.__getitem__), \
                mock.patch.object(sys, "argv", argv + ["--model_aut_path", "fake-aut",
                                                       "--model_subd_path", "fake-subd"]):
            tool7.main()
    return read_jsonl(out_path)


def run_tool9(in_path, out_path):
    with mock.patch.object(sys, "argv", ["tool9", "--file_path", in_path, "--output_path", out_path]):
        tool9.main()
    return read_jsonl(out_path)


def run_pipeline(root_path, out_dir, tag, real_models=False):
    """Run Tool 3 -> 7 -> 9 on root_path; return (tool3_rows, tool7_rows, tool9_rows)."""
    p3, p7, p9 = (os.path.join(out_dir, f"{tag}_tool{n}.jsonl") for n in (3, 7, 9))
    rows3 = run_tool3(root_path, p3)
    rows7 = run_tool7(p3, p7, real_models=real_models)
    rows9 = run_tool9(p7, p9)
    return rows3, rows7, rows9
