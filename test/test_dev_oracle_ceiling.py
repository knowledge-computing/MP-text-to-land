"""
Tests for dev/oracle_ceiling.py on synthetic toy data only. Skipped without geopandas.

    mapprejudice/bin/python -m unittest discover -s test -p "test_dev_oracle_ceiling.py" -v
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
    from shapely.geometry import box
except ImportError:  # pragma: no cover
    gpd = None


@unittest.skipIf(gpd is None, "geopandas not installed")
class OracleCeilingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        spec = importlib.util.spec_from_file_location("oracle_ceiling", os.path.join(REPO_ROOT, "dev", "oracle_ceiling.py"))
        self.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.script)
        rows = [  # (ids, county, S spans, T spans, R spans)
            ("['a']", "Anoka County", ["Section 5"], ["Township 31 North"], ["Range 22 West"]),     # all recoverable
            ("['b']", "Anoka County", ["Section 5 of Township 31"], [], ["Range 22"]),               # T only via S span
            ("['c']", "Anoka County", ["Section 6"], ["Township 31"], ["Range 23 East"]),            # wrong R, wrong dir
            ("['d']", "Milwaukee County", [], [], []),                                               # no span
            ("['e']", "Milwaukee County", ["Sec. 9"], ["T. 7 N."], ["R. 21 E."]),                    # not in gold
        ]
        gpd.GeoDataFrame({
            "image_ids": [r[0] for r in rows], "cnty_name": [r[1] for r in rows],
            "SECTION_p": [str(r[2]) for r in rows], "TOWNSHIP_p": [str(r[3]) for r in rows],
            "RANGE_p": [str(r[4]) for r in rows],
        }, geometry=[box(i, 0, i + 1, 1) for i in range(len(rows))], crs="EPSG:5070").to_file(
            os.path.join(d, "raw.geojson"), driver="GeoJSON")
        with open(os.path.join(d, "gold.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "single", "status", "county", "deed_key", "T", "R", "S", "Rdir"])
            for key, county in (("a", "anoka"), ("b", "anoka"), ("c", "anoka"), ("d", "milwaukee")):
                w.writerow(["test", "True", "ok", county, key, 31 if county == "anoka" else 7, 22 if county == "anoka" else 21,
                            5, "W" if county == "anoka" else "E"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts(self):
        d = self.tmp.name
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = self.script.main(["--raw", os.path.join(d, "raw.geojson"), "--gold", os.path.join(d, "gold.csv"),
                                     "--summary", os.path.join(d, "o.json")])
        self.assertEqual(code, 0)
        r = json.loads(out.getvalue())
        self.assertEqual(r["inputs"]["joined"], 4)
        allc, spanc = r["counts"]["ALL"]["all"], r["counts"]["ALL"]["with_span"]
        self.assertEqual((allc["deeds"], spanc["deeds"]), (4, 3))
        self.assertEqual(spanc["T_in_same_label_spans"], 2)   # a, c
        self.assertEqual(spanc["T_in_any_span"], 3)           # b via its Section span
        self.assertEqual(spanc["R_in_same_label_spans"], 2)   # a, b
        self.assertEqual(spanc["TRS_in_same_label_spans"], 1)
        self.assertEqual(spanc["TRS_in_any_span"], 2)
        self.assertEqual(spanc["R_spans_state_gold_Rdir"], 1)
        self.assertEqual(spanc["R_spans_state_other_Rdir"], 1)
        self.assertEqual(r["counts"]["milwaukee"]["all"]["deeds"], 1)
        self.assertNotIn("milwaukee", [k for k, v in r["counts"].items() if v.get("with_span", {}).get("deeds")])
        self.assertEqual(spanc["pct"]["TRS_in_any_span"], 66.67)


if __name__ == "__main__":
    unittest.main()
