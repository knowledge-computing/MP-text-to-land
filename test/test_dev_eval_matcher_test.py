"""
Tests for dev/eval_matcher_test.py on synthetic toy data only.

    python -m unittest discover -s test -p "test_dev_eval_matcher_test.py" -v
"""

import json
import os
import sys
import tempfile
import unittest
from collections import Counter

from shapely.geometry import box

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dev"))
import eval_matcher_test as et  # noqa: E402

from src.matcher import NameNormalizer, PlatIndex  # noqa: E402

PLATS = [
    {"county": "anoka", "name": "OAK RIDGE", "aliases": [], "geometry": box(0, 0, 100, 100), "sections": set()},
    {"county": "anoka", "name": "ELM PARK", "aliases": [], "geometry": box(100, 0, 200, 100), "sections": set()},
    {"county": "dakota", "name": "Pine Hills", "aliases": [], "geometry": box(5000, 0, 5100, 100), "sections": set()},
]


def deed(county, preds, geom, parts=1):
    return {"county": county, "props": {"SUBDIVISION_p": preds}, "geometry": geom, "n_parts": parts}


class EvalMatcherTestTest(unittest.TestCase):
    def setUp(self):
        self.index = PlatIndex(PLATS, NameNormalizer({}))
        self.deeds = [
            deed("anoka", ["Oak Ridge"], box(10, 10, 20, 20)),          # centroid in, correct; raw case fails paper
            deed("anoka", ["ELM PARK"], box(10, 10, 20, 20)),           # wrong plat; paper hit (exact raw string)
            deed("anoka", ["Pine Hills"], box(150, 10, 160, 20)),       # own-county: no anoka candidate >= 70
            deed("anoka", [], box(150, 10, 160, 20)),                    # no prediction
            deed("anoka", ["OAK RIDGE"], box(1, 1, 2, 2), parts=2),     # multi-part: not eligible
        ]

    def test_run_metrics(self):
        errors = Counter()
        res = et.run(self.deeds, PLATS, self.index, errors)
        self.assertEqual(errors, Counter())
        m = res["matcher"]["ALL"]
        self.assertEqual((m["counts"]["b_all"], m["counts"]["a_with_pred"]), (4, 3))
        self.assertEqual(m["centroid_in_pct_b"], 25.0)
        self.assertEqual(m["centroid_in_pct_a"], 33.33)
        self.assertEqual(m["plat_acc_a"], 33.33)
        self.assertEqual((m["intersects_pct_a"], m["contains_pct_a"]), (33.33, 33.33))
        p = res["paper_original"]["ALL"]
        # raw, all counties: "ELM PARK" exact, "Pine Hills" matches dakota, "Oak Ridge" (mixed case) fails,
        # multi-part "OAK RIDGE" counts in the all-deed denominator only
        self.assertEqual(p["counts"]["all_with_pred"], 4)
        self.assertEqual(p["counts"]["all_hit"], 3)
        self.assertEqual((p["counts"]["a_with_pred"], p["counts"]["a_hit"]), (3, 2))
        self.assertEqual(p["counts"]["a_hit_intersects"], 0)  # ELM PARK plat does not touch deed 2; Pine Hills far away

    def test_load_test_deeds_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "toy_preds.geojson")
            sq = lambda x: [[[x, 45.0], [x + 0.001, 45.0], [x + 0.001, 45.001], [x, 45.001], [x, 45.0]]]  # noqa: E731
            feats = [{"type": "Feature", "properties": {"cnty_name": "Anoka", "SUBDIVISION_p": ["A"]},
                      "geometry": {"type": "MultiPolygon", "coordinates": [sq(-93.2)]}},
                     {"type": "Feature", "properties": {"cnty_name": "Anoka", "SUBDIVISION_p": []},
                      "geometry": {"type": "MultiPolygon", "coordinates": [sq(-93.2), sq(-93.1)]}}]
            with open(path, "w") as f:
                json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                           "features": feats}, f)
            deeds = et.load_test_deeds(path)
        self.assertEqual([d["n_parts"] for d in deeds], [1, 2])
        self.assertEqual(deeds[0]["county"], "anoka")

    def test_run_once_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = os.path.join(tmp, "test_v1.json")
            open(summary, "w").close()
            with self.assertRaises(SystemExit):
                et.main(["--summary", summary])


if __name__ == "__main__":
    unittest.main()
