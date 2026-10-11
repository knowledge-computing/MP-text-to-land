"""
Tests for dev/infer_try08.py on synthetic toy data only (fake OCR files, a blank spaCy pipeline with an
entity_ruler standing in for the try08 model).

    python -m unittest discover -s test -p "test_dev_infer_try08.py" -v
"""

import importlib.util
import io
import json
import os
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "dev", "infer_try08.py")
spec = importlib.util.spec_from_file_location("infer_try08", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

SECRET = "Toysecretword"
MSI = "/msi/home/someone/data/toy-ocr/"  # synthetic remote root; local root folder is also "toy-ocr"


def square(x0, y0, size=1.0):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x0 + size, y0], [x0 + size, y0 + size],
                                                [x0, y0 + size], [x0, y0]]]}


def toy_nlp():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "SUBDIVISION", "pattern": [{"LOWER": "maple"}]},
        {"label": "SUBDIVISION", "pattern": [{"LOWER": "addition"}]},  # separate ent, merged by extract_spans
        {"label": "SECTION", "pattern": [{"LOWER": "twelve"}]},
        {"label": "TOWNSHIP", "pattern": [{"LOWER": "thirty"}]},
        {"label": "COUNTY", "pattern": [{"LOWER": "anoka"}]},
    ])
    return nlp


@unittest.skipIf(spacy is None, "spaCy not installed")
class InferTry08Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.ocr = os.path.join(root, "toy-ocr")
        os.makedirs(os.path.join(self.ocr, "ocr", "anoka"))
        pages = {"ocr/anoka/p1.txt": f"Lot 3 of  Maple Addition,\n{SECRET} Anoka County",
                 "ocr/anoka/p2.txt": "in Section twelve, Township thirty",
                 "ocr/anoka/p3.txt": f"nothing here {SECRET}"}
        for rel, text in pages.items():
            with open(os.path.join(self.ocr, rel), "w") as f:
                f.write(text)
        deed_a = {"image_ids": "('toyimg_a1', 'toyimg_a2')", "cnty_name": "Anoka", "add_cov": SECRET,
                  "saved_path": f"('{MSI}ocr/anoka/p1.txt', '{MSI}ocr/anoka/p2.txt')"}
        deed_b = {"image_ids": "['b1']", "cnty_name": "Anoka",
                  "saved_path": f"['{MSI}ocr/anoka/p3.txt', '{MSI}ocr/anoka/missing.txt']"}
        feats = [
            {"type": "Feature", "geometry": square(0, 0), "properties": deed_a},
            {"type": "Feature", "geometry": square(1, 0), "properties": deed_a},  # touching -> 1 part
            {"type": "Feature", "geometry": square(5, 5), "properties": deed_b},
            {"type": "Feature", "geometry": square(9, 9), "properties": {**deed_b, "saved_path": "['x']"}},
        ]
        self.geojson = os.path.join(root, "toy_val.geojson")
        with open(self.geojson, "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        self.out = os.path.join(root, "pred", "toy_val_predicted.geojson")

    def tearDown(self):
        self.tmp.cleanup()

    def read_out(self):
        with open(self.out) as f:
            return json.load(f)["features"]

    # ---- helpers
    def test_remap_and_parse(self):
        self.assertEqual(script.remap_path(MSI + "ocr/x/y.txt", "/loc/toy-ocr"), script.Path("/loc/toy-ocr/ocr/x/y.txt"))
        self.assertIsNone(script.remap_path("/elsewhere/y.txt", "/loc/toy-ocr"))
        self.assertEqual(script.parse_list("('a', 'b')"), ["a", "b"])
        self.assertEqual(script.parse_list(None), [])
        self.assertEqual(script.parse_list("not a list ("), ["not a list ("])
        self.assertEqual(script.deed_key({"cnty_name": " Anoka", "image_ids": "('b','a')"}),
                         script.deed_key({"cnty_name": "anoka", "image_ids": ["a", "b"]}))

    def test_extract_spans_merge_rule(self):
        doc = toy_nlp()("Maple Addition and Maple")
        spans = script.extract_spans(doc)
        self.assertEqual([s[0] for s in spans["SUBDIVISION"]], ["Maple Addition", "Maple"])
        self.assertEqual(spans["SUBDIVISION"][0][1:], (0, 14))

    # ---- end to end
    def test_run_groups_deeds_and_keeps_offsets(self):
        result = script.run(self.geojson, self.out, self.ocr, toy_nlp())
        c = result["counts"]
        self.assertEqual((c["rows_in"], c["deeds"], c["deeds_written"]), (4, 2, 2))
        self.assertEqual(c["rows_merged_into_existing_deed"], 2)
        self.assertEqual(c["rows_with_different_saved_path"], 1)
        self.assertEqual((c["pages"], c["pages_missing"], c["deeds_with_missing_page"]), (4, 1, 1))
        self.assertEqual(c["deeds_multi_part"], 1)
        self.assertEqual(result["deeds_with_any_plss"], 1)
        a, b = self.read_out()
        self.assertEqual(a["properties"]["SUBDIVISION_p"], ["Maple Addition"])
        self.assertEqual(a["properties"]["SECTION_p"], ["twelve"])
        self.assertEqual((a["properties"]["n_rows"], a["properties"]["n_parts"]), (2, 1))
        self.assertEqual((b["properties"]["n_rows"], b["properties"]["n_parts"]), (2, 2))
        labels = [s[0] for s in a["properties"]["ner_spans"]]
        self.assertEqual(sorted(labels), ["COUNTY", "SECTION", "SUBDIVISION", "TOWNSHIP"])
        starts = [s[2] for s in a["properties"]["ner_spans"] if s[0] in ("SECTION", "TOWNSHIP")]
        self.assertLess(starts[0], starts[1])  # offsets give document order
        self.assertNotIn("saved_path", a["properties"])
        self.assertEqual(b["properties"]["SUBDIVISION_p"], [])

    def test_limit(self):
        result = script.run(self.geojson, self.out, self.ocr, toy_nlp(), limit=1)
        self.assertEqual(result["counts"]["deeds_written"], 1)

    def test_main_prints_counts_only(self):
        model_dir = os.path.join(self.tmp.name, "toy_model")
        toy_nlp().to_disk(model_dir)
        summary = os.path.join(self.tmp.name, "summary.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            script.main(["--geojson", self.geojson, "--model", model_dir, "--ocr-root", self.ocr,
                         "--out", self.out, "--summary", summary])
        printed = buf.getvalue()
        for leak in (SECRET, "Maple", "twelve", "toyimg_a1", "ocr/anoka"):
            self.assertNotIn(leak, printed)
        with open(summary) as f:
            self.assertEqual(json.load(f)["counts"]["deeds"], 2)

    # ---- guards
    def test_test_split_refused(self):
        test_named = os.path.join(self.tmp.name, "toy_test6_2_2.geojson")
        os.rename(self.geojson, test_named)
        with self.assertRaises(SystemExit):
            script.run(test_named, self.out, self.ocr, toy_nlp())
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                unittest.mock.patch("sys.stderr", io.StringIO()):
            script.main(["--split", "test", "--out", self.out])

    def test_out_inside_repo_refused(self):
        with self.assertRaises(SystemExit):
            script.main(["--geojson", self.geojson, "--out", os.path.join(REPO_ROOT, "results", "x.geojson")])


if __name__ == "__main__":
    unittest.main()
