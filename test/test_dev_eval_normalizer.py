"""
Tests for dev/eval_normalizer.py and dev/diagnose_low_ceiling.py on synthetic toy data only
(fake PLSS grid, fake deeds, fake GPT batch). Skipped without geopandas.

    mapprejudice/bin/python -m unittest discover -s test -p "test_dev_eval_normalizer.py" -v
"""

import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, box
except ImportError:  # pragma: no cover
    gpd = None


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "dev", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(fn, argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = fn(argv)
    return code, out.getvalue()


# deed: (key, county, S spans, T spans, R spans, deed box, gold S, GPT triples, GPT RANGE_p)
DEEDS = [
    ("a", "Anoka County", ["Section 1"], ["Township 31 North"], ["Range 22 West"], (1, 1, 2, 2), 1,
     [(1, 31, 22)], ["Range 22 West"]),                                                   # all right
    ("b", "Anoka County", ["Section 2"], ["Township 31"], ["Range 22"], (3, 3, 4, 4), 1,
     [(2, 31, 22)], ["Range 22"]),                                                        # NER says S2, deed in S1
    ("c", "Anoka County", [], [], [], (5, 5, 6, 6), 1, [], []),                           # no PLSS span
    ("d", "Milwaukee County", ["Sec. 5"], ["T. 7 N."], ["R. 21 E."], (201, 201, 202, 202), 5,
     [(5, 7, 21)], ["R. 21 E."]),                                                         # all right (WI)
    ("e", "Anoka County", ["Lot 1, Block 2 of Section 1"], ["Township 13"], ["Range 22"], (6, 1, 7, 2), 1,
     [(1, 31, 22)], ["Range 22"]),                                                        # T typo; GPT right
]


@unittest.skipIf(gpd is None, "geopandas not installed")
class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.p = lambda name: os.path.join(self.tmp.name, name)
        crs = "EPSG:5070"
        # S1 and its eastern neighbour S2 (T31 R22 W), and one Milwaukee section; deed b lies in S1
        sec = [(1, 31, 22, "W", (0, 0, 10, 10)), (2, 31, 22, "W", (10, 0, 20, 10)), (5, 7, 21, "E", (200, 200, 210, 210))]
        gpd.GeoDataFrame({"SEC": [s[0] for s in sec], "TWP": [s[1] for s in sec], "RNG": [s[2] for s in sec],
                          "DIR": [s[3] for s in sec]}, geometry=[box(*s[4]) for s in sec], crs=crs).to_file(
            d("plss.geojson"), driver="GeoJSON")
        gpd.GeoDataFrame({"state": ["MN", "MN", "WI"], "counties": ["anoka", "anoka", "milwaukee"],
                          "T": [31, 31, 7], "Tdir": ["N"] * 3, "R": [22, 22, 21], "Rdir": ["W", "W", "E"], "S": [1, 2, 5]},
                         geometry=[box(*s[4]) for s in sec], crs=crs).to_file(d("sections.gpkg"), layer="sections", driver="GPKG")
        with open(d("valid.json"), "w") as f:
            json.dump({"valid": {"MN": {"anoka": [[31, "N", 22, "W", 1], [31, "N", 22, "W", 2]]},
                                 "WI": {"milwaukee": [[7, "N", 21, "E", 5]]}}}, f)
        common = {"image_ids": [f"['{x[0]}']" for x in DEEDS], "cnty_name": [x[1] for x in DEEDS]}
        geoms = [MultiPolygon([box(*x[5])]) for x in DEEDS]
        gpd.GeoDataFrame({**common, "SECTION_p": [str(x[2]) for x in DEEDS], "TOWNSHIP_p": [str(x[3]) for x in DEEDS],
                          "RANGE_p": [str(x[4]) for x in DEEDS]}, geometry=geoms, crs=crs).to_file(d("raw.geojson"), driver="GeoJSON")
        gpd.GeoDataFrame({**common, "TRIPLE_p": [str(x[7]) for x in DEEDS], "RANGE_p": [str(x[8]) for x in DEEDS]},
                         geometry=geoms, crs=crs).to_file(d("batch.geojson"), driver="GeoJSON")
        with open(d("gold.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "single", "status", "county", "deed_key", "T", "Tdir", "R", "Rdir", "S"])
            for x in DEEDS:
                mil = x[1].startswith("Milwaukee")
                w.writerow(["test", "True", "ok", "milwaukee" if mil else "anoka", x[0], 7 if mil else 31, "N",
                            21 if mil else 22, "E" if mil else "W", x[6]])

    def tearDown(self):
        self.tmp.cleanup()

    def test_eval_normalizer(self):
        script = load("eval_normalizer")
        p = self.p
        code, out = run(script.main, ["--raw", p("raw.geojson"), "--batch", p("batch.geojson"), "--plss", p("plss.geojson"),
                                      "--gold", p("gold.csv"), "--valid", p("valid.json"), "--summary", p("s.json")])
        self.assertEqual(code, 0)
        with open(p("s.json")) as f:
            s = json.load(f)
        self.assertEqual((s["inputs"]["joined_deeds"], s["inputs"]["deeds_with_span"]), (5, 4))
        self.assertEqual(s["errors"], {})
        rule = s["results"]["rule_v2[county]"]["ALL"]["with_span"]
        self.assertEqual((rule["deeds"], rule["covered"], rule["top1_1x1"], rule["top1_6x6"]), (4, 3, 2, 3))
        self.assertEqual((rule["oracle_TRS"], rule["field_exact_TRS"]), (2, 2))
        self.assertEqual(rule["share_of_oracle"]["field_exact_TRS"], 1.0)
        self.assertEqual(rule["miss:gold_not_in_spans"], 1)          # b
        self.assertEqual(rule["miss:abstain:not_in_plss"], 1)        # e
        self.assertEqual(s["results"]["rule_v2[county]"]["ALL"]["all"]["reason:no_spans"], 1)  # c
        gpt = s["results"]["gpt4o_auto"]["ALL"]["with_span"]
        self.assertEqual((gpt["covered"], gpt["top1_1x1"], gpt["top1_6x6"]), (4, 3, 4))
        regex = s["results"]["regex_v1"]["ALL"]["with_span"]
        self.assertEqual((regex["covered"], regex["top1_1x1"], regex["miss:gold_not_in_spans"]), (4, 2, 2))
        mil = s["results"]["rule_v2[county]"]["milwaukee"]["with_span"]
        self.assertEqual((mil["deeds"], mil["top1_1x1"]), (1, 1))
        self.assertEqual(s["paper_scoring"]["gpt4o_auto"]["with_triple"], 4)
        self.assertEqual(s["paper_scoring"]["gpt4o_auto"]["orig_1x1"], 3)
        self.assertNotIn("Section", out)

    def test_diagnose_low_ceiling(self):
        script = load("diagnose_low_ceiling")
        p = self.p
        code, out = run(script.main, ["--raw", p("raw.geojson"), "--gold", p("gold.csv"), "--valid", p("valid.json"),
                                      "--sections", p("sections.gpkg"), "--summary", p("d.json")])
        self.assertEqual(code, 0)
        r = json.loads(out)
        a = r["counties"]["anoka"]
        self.assertEqual(a["deeds"], 3)  # a, b, e (c has no span)
        self.assertEqual(a["deeds_with_valid_section_reading"], 2)  # a, b
        self.assertEqual(a["nearest_same_section"], 1)
        self.assertEqual(a["nearest_adjacent"], 1)
        self.assertEqual(a["top1_adjacent"], 1)
        self.assertEqual((a["S_with_plat_kw"], a["kw_lot"], a["kw_block"], a["deeds_with_plat_kw"]), (1, 1, 1, 1))
        self.assertEqual(a["T_with_plss_kw"], 3)  # "Township" in a, b and e
        self.assertEqual(r["counties"]["milwaukee"]["nearest_same_section"], 1)
        self.assertEqual(r["errors"], {})


if __name__ == "__main__":
    unittest.main()
