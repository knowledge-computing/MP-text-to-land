"""
Tests for dev/recompute_table3.py on synthetic toy data only (fake PLSS grid, fake deeds).
Skipped when geopandas is unavailable.

    mapprejudice/bin/python -m unittest discover -s test -p "test_dev_recompute_table3.py" -v
"""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "dev", "recompute_table3.py")
spec = importlib.util.spec_from_file_location("recompute_table3", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError:  # pragma: no cover
    gpd = None


class NormalizerTests(unittest.TestCase):
    def test_words_and_digits(self):
        n = script.normalize_number
        self.assertEqual(n("Township One hundred and six North"), 106)
        self.assertEqual(n("Range Twenty-four (24) West"), 24)
        self.assertEqual(n("Sec. 5"), 5)
        self.assertEqual(n("Section Thirty one"), 31)
        self.assertEqual(n("Township numbered Eight (8) North"), 8)
        self.assertIsNone(n("Section"))

    def test_direction(self):
        self.assertEqual(script.range_direction(["Range 14 West"]), "W")
        self.assertEqual(script.range_direction(["Range 22 E."]), "E")
        self.assertEqual(script.range_direction(["Range 22"]), "")


@unittest.skipIf(gpd is None, "geopandas not installed")
class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        # one township (106, 14, W) with sections 1 and 2 side by side; another (7, 21, E) section 1
        plss = gpd.GeoDataFrame(
            {"SEC": [1, 2, 1], "TWP": [106, 106, 7], "RNG": [14, 14, 21], "DIR": ["W", "W", "E"]},
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10), box(100, 100, 110, 110)], crs="EPSG:5070")
        plss.to_file(os.path.join(d, "plss.geojson"), driver="GeoJSON")
        deeds = gpd.GeoDataFrame({
            "image_ids": ["a", "b", "c", "d", "e"],
            "TRIPLE_p": [
                str([(1, 106, 14, "W")]),        # a: right section, with DIR  -> step1 hit
                str([(2, 106, 14, "")]),         # b: wrong section, no DIR    -> step3 (6x6) hit
                str([(2, 106, 14, "W")]),        # c: wrong section, with DIR  -> original never retries 6x6
                str([]),                         # d: no triple                -> not in denominator
                str([(1, 106, 14, "W")]),        # e: straddles sections 1|2   -> contains fails, intersects hits
            ],
            "SECTION_p": [["Section One"], ["Sec 2"], ["Section Two"], [], ["Section 1"]],
            "TOWNSHIP_p": [["Township One hundred and six"]] * 3 + [[], ["Twp 106"]],
            "RANGE_p": [["Range Fourteen West"], ["Range 14"], ["Range 14 West"], [], ["Range 14 W."]],
            "SUBDIVISION_p": [["Toy Park"], [], ["Toyland Addn"], [], []],
            "add_cov": ["Toy Park", "Other", "Nowhere", None, None],
            "CITY_p": [[], [], [], [], []], "city": [None] * 5,
            "COUNTY_p": [["Toy County"]] * 5, "cnty_name": ["Toy"] * 5,
            "STATE_p": [["Minnesota"]] * 5, "state": ["MN"] * 5,
        }, geometry=[box(2, 2, 4, 4), box(2, 2, 4, 4), box(2, 2, 4, 4), box(2, 2, 4, 4), box(8, 2, 12, 4)], crs="EPSG:5070")
        deeds.to_file(os.path.join(d, "pred.geojson"), driver="GeoJSON")
        subd = gpd.GeoDataFrame({"SubdName": ["Toy Park", "Far Away"]}, geometry=[box(0, 0, 5, 5), box(500, 500, 501, 501)], crs="EPSG:5070")
        subd.to_file(os.path.join(d, "subd.geojson"), driver="GeoJSON")

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *extra):
        d = self.tmp.name
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = script.main(["--pred", f"{d}/pred.geojson", "--raw", f"{d}/pred.geojson", "--plss", f"{d}/plss.geojson",
                              "--outdir", f"{d}/out", *extra])
        self.assertEqual(rc, 0)
        return json.loads(out.getvalue()), out.getvalue()

    def test_original_cascade_and_fixed(self):
        res, _ = self.run_script("--tag", "t")
        t3 = res["table3"]
        self.assertEqual(t3["deeds_total"], 5)
        self.assertEqual(t3["with_triple"], 4)
        self.assertEqual(t3["orig_step1_dir"], 1)        # a
        self.assertEqual(t3["orig_step3_tuple"], 1)      # b
        self.assertEqual(t3["orig_1x1"], 1)
        self.assertEqual(t3["orig_6x6"], 2)
        self.assertEqual(t3["orig_failed_with_dir_never_retried"], 2)   # c, e
        self.assertEqual(t3["fixed_1x1_contains"], 1)    # a
        self.assertEqual(t3["fixed_6x6_contains"], 4)    # a, b, c, e (township polygon)
        self.assertEqual(t3["fixed_1x1_intersects"], 2)  # a, e
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "out", "t.json")))

    def test_regex_mode_and_table2_and_subd(self):
        res, text = self.run_script("--triples", "regex", "--subd", f"{self.tmp.name}/subd.geojson")
        self.assertEqual(res["table3"]["with_triple"], 4)
        self.assertEqual(res["table3"]["fixed_1x1_contains"], 1)
        self.assertEqual(res["table2"]["Subdivision"], {"TP": 1, "FP": 1, "FN": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(res["subdivision"]["deeds_with_subdivision_prediction"], 2)
        self.assertEqual(res["subdivision"]["deeds_candidate_intersects_deed"], 1)
        for leak in ("Toy Park", "Toyland", "Township One"):
            self.assertNotIn(leak, text)


if __name__ == "__main__":
    unittest.main()
