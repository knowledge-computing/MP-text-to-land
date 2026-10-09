"""
Tests for dev/eval_matcher_val.py and dev/weak_labels.py on synthetic toy plats and deeds only.

    python -m unittest discover -s test -p "test_dev_eval_matcher.py" -v
"""

import os
import sys
import unittest
from collections import Counter

from shapely.geometry import MultiPolygon, box

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dev"))
import eval_matcher_val as ev  # noqa: E402
import weak_labels as wl  # noqa: E402

from src.matcher import MatchConfig, NameNormalizer, PlatIndex  # noqa: E402

PLATS = [
    {"county": "anoka", "name": "OAK RIDGE", "aliases": [], "geometry": box(0, 0, 100, 100), "sections": {(1, 31, 24)}},
    {"county": "anoka", "name": "OAK RIDGE 2ND", "aliases": [], "geometry": box(100, 0, 200, 100), "sections": {(2, 31, 24)}},
    {"county": "dakota", "name": "OAK RIDGE", "aliases": [], "geometry": box(5000, 0, 5100, 100), "sections": set()},
]


def deed(county, preds, geom, parts=1):
    return {"county": county, "props": {"SUBDIVISION_p": preds}, "geometry": geom, "n_parts": parts}


class EvalMatcherTest(unittest.TestCase):
    def setUp(self):
        self.index = PlatIndex(PLATS, NameNormalizer({}))
        self.deeds = [
            deed("anoka", ["Oak Ridge"], box(10, 10, 20, 20)),           # correct, inside
            deed("anoka", "['Oak Ridge 2nd']", box(10, 10, 20, 20)),     # wrong plat, adjacent (100 m away)
            deed("anoka", "[]", box(150, 10, 160, 20)),                   # no prediction
            deed("washington", ["Oak Ridge"], box(10, 10, 20, 20)),       # no plat layer -> not eligible
            deed("anoka", ["Oak Ridge"], MultiPolygon([box(1, 1, 2, 2), box(5, 5, 6, 6)]), parts=2),
        ]
        self.labels = wl.weak_labels(self.deeds, PLATS)

    def test_weak_labels(self):
        self.assertEqual([l["status"] for l in self.labels],
                         ["eligible", "eligible", "eligible", "no_plat_layer", "multi_part"])
        self.assertEqual([l["plat"] for l in self.labels[:3]], [0, 0, 1])
        self.assertAlmostEqual(self.labels[0]["max_frac"], 1.0)

    def test_preds_parsing(self):
        self.assertEqual(ev.preds_of({"SUBDIVISION_p": "['A', 'None', '']"}), ["A"])
        self.assertEqual(ev.preds_of({"SUBDIVISION_p": "nan"}), [])

    def test_evaluate_counts(self):
        errors = Counter()
        res = ev.evaluate(self.deeds, self.labels, [frozenset()] * 5, self.index, MatchConfig(norm="n1"), errors)
        a = res["ALL"]
        self.assertEqual(errors, Counter())
        self.assertEqual((a["counts"]["b_all"], a["counts"]["a_with_pred"]), (3, 2))
        self.assertEqual(a["counts"]["plat_correct"], 1)
        self.assertEqual((a["plat_acc_a"], a["plat_acc_b"]), (50.0, 33.33))
        self.assertEqual(a["contains_pct_a"], 50.0)
        self.assertEqual(a["intersects_pct_a"], 50.0)
        self.assertEqual(a["centroid_m"]["min,p10,p25,p50,p75,p90,max"][-1], 85.0)  # deed centroid x=15 -> plat at 100
        self.assertEqual(a["original_pct_a"], 100.0)

    def test_original_raw_all_counties(self):
        res = ev.evaluate(self.deeds, self.labels, [frozenset()] * 5, self.index, MatchConfig(norm="raw", own_county=False), Counter())
        self.assertEqual(res["ALL"]["original_pct_a"], 0.0)   # case-sensitive: no candidate >= 70... for mixed case
        self.assertEqual(res["ALL"]["counts"].get("top1", 0), 0)

    def test_grid_names_unique(self):
        names = [c.name() for c in ev.configs()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 39)


if __name__ == "__main__":
    unittest.main()
