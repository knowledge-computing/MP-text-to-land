"""
Tests for dev/check_split_overlap.py on synthetic toy GeoJSON only.

    python -m unittest discover -s test -p "test_dev_split_overlap.py" -v
"""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "dev", "check_split_overlap.py")
spec = importlib.util.spec_from_file_location("check_split_overlap", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

SECRET = "TOYSECRETID"


def feature(ids, county="Anoka"):
    return {"type": "Feature", "geometry": None, "properties": {"image_ids": ids, "cnty_name": county}}


SPLITS = {
    # train: deed A twice (two geometries), deed B, deed C (shares a page with test deed T2)
    "train": [feature("('a/1', 'a/2')"), feature("('a/1', 'a/2')"), feature("['b/1']"),
              feature(f"('c/1', '{SECRET}/9')"), feature(None), feature("not a list (")],
    "val": [feature("['v/1']", "Milwaukee"), feature("('b/1',)")],  # val deed identical to train deed B
    # test: T1 clean, T2 overlaps train deed C via SECRET/9, same id in another county is NOT shared
    "test": [feature("['t/1']"), feature(f"('t/2', '{SECRET}/9')"), feature("['b/1']", "Dakota")],
}


class SplitOverlapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = {}
        for name, feats in SPLITS.items():
            p = os.path.join(self.tmp.name, f"{name}.geojson")
            with open(p, "w") as f:
                json.dump({"type": "FeatureCollection", "features": feats}, f)
            self.paths[name] = p

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts(self):
        out = script.summarize(self.paths)
        tr = out["splits"]["train"]
        self.assertEqual(tr["features"], 6)
        self.assertEqual(tr["features_empty_ids"], 1)
        self.assertEqual(tr["features_unparseable_ids"], 1)
        self.assertEqual(tr["deeds"], 3)
        self.assertEqual(tr["deeds_with_multiple_features"], 1)
        self.assertEqual(out["splits"]["test"]["by_county"], {"anoka": {"deeds": 2, "features": 2},
                                                               "dakota": {"deeds": 1, "features": 1}})
        self.assertEqual(out["pairs"]["train~val"]["identical_deeds"], 1)
        self.assertEqual(out["test_leakage"], {"test_deeds": 3, "test_deeds_identical_to_a_non_test_deed": 0,
                                               "test_deeds_sharing_any_page_with_non_test": 1,
                                               "test_pages_seen_outside_test": 1})
        self.assertEqual(out["all_splits_unique_deeds"], 7)

    def test_prints_no_ids(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = script.main([f"--split={n}={p}" for n, p in self.paths.items()])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertNotIn(SECRET, text)
        self.assertNotIn("a/1", text)
        self.assertNotIn(self.tmp.name, text)


if __name__ == "__main__":
    unittest.main()
