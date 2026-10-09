"""
Tests for dev/check_page_grouping.py on synthetic toy data only (fake OCR tree and fake
GeoJSON files in a temporary directory).

The script must run with the Python standard library alone and print aggregate counts
only. Run from the repository root:

    python -m unittest discover -s test -p "test_dev_grouping_check.py" -v
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

from test_dev_collision_script import BLOCKED, RUNNER

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "dev", "check_page_grouping.py")

spec = importlib.util.spec_from_file_location("check_page_grouping", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

MN, WI = "mn-toy-county", "wi-toy-county"
OCR_PAGES = {
    MN: ["a/deed_001", "a/deed_002",          # one deed, one folder
         "b/scan_001", "c/scan_002",          # two deeds, same prefix, two folders
         "d/part_001", "e/part_002",          # one deed stored across two folders
         "f/toy_batch_sheet_001", "f/toy_batch_sheet_002",  # two deeds, same prefix, one folder
         "g/lonely"],                         # no GeoJSON deed
    WI: ["h/leaf_001"],
}
GEOJSON_IMAGE_IDS = {
    MN: ["['a/deed_001', 'a/deed_002']", "['b/scan_001']", ["c/scan_002"], "['d/part_001', 'e/part_002']",
         "['f/toy_batch_sheet_001']", "['f/toy_batch_sheet_002']",
         "['x/missing_001']",                 # GeoJSON deed without OCR pages
         "not a list", ""],                   # unparseable, empty
    WI: [["h/leaf_001"], None],
}
EXPECTED = {
    "counties": 2, "ocr_only_counties": 0, "geojson_only_folders": 0,
    "ocr_files": 10, "geojson_deeds": 8, "unparseable": 1,
    "geojson_ids": 10, "geojson_ids_missing": 1, "multi_folder_deeds": 1,
    "current": {"groups": 6, "multi_folder_groups": 2, "labeled_groups": 5, "over_merged_groups": 2,
                "multi_folder_consistent": 1, "multi_folder_over_merged": 1, "multi_folder_unlabeled": 0,
                "deeds_with_pages": 7, "split_deeds": 0, "exact_deeds": 3},
    "by_folder": {"groups": 8, "multi_folder_groups": 0, "labeled_groups": 7, "over_merged_groups": 1,
                  "multi_folder_consistent": 0, "multi_folder_over_merged": 0, "multi_folder_unlabeled": 0,
                  "deeds_with_pages": 7, "split_deeds": 1, "exact_deeds": 4},
}


class GroupingCheckTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.ocr = os.path.join(tmp.name, "ocr", "txt")
        self.geo = os.path.join(tmp.name, "geojson")
        for county, pages in OCR_PAGES.items():
            for rel in pages:
                path = os.path.join(self.ocr, county, rel + ".txt")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("toy text")
        for county, values in GEOJSON_IMAGE_IDS.items():
            os.makedirs(os.path.join(self.geo, county), exist_ok=True)
            features = [{"type": "Feature", "geometry": None, "properties": {"image_ids": v}} for v in values]
            with open(os.path.join(self.geo, county, "toy.geojson"), "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": features}, f)

    def run_stdlib_only(self, *args):
        return subprocess.run([sys.executable, "-c", RUNNER, ",".join(BLOCKED), SCRIPT, *args],
                              capture_output=True, text=True, cwd=self.tmp,
                              env={**os.environ, "OCRTXT_PATH": "", "GEOJSON_PATH": ""})

    def test_counts_on_toy_data(self):
        # The "diagnosis" section is covered by test_grouping_mismatches.py.
        s = script.summary(self.ocr, self.geo)
        self.assertEqual({k: v for k, v in s.items() if k != "diagnosis"}, EXPECTED)

    def test_folder_grouping_trades_over_merges_for_split_deeds(self):
        # The toy data has one over-merge that folder grouping fixes (b/scan vs c/scan),
        # one real multi-folder deed that it would split (d/part + e/part), and one
        # same-folder over-merge that it cannot fix (f/toy_batch_sheet_*).
        s = script.summary(self.ocr, self.geo)
        self.assertEqual(s["current"]["over_merged_groups"] - s["by_folder"]["over_merged_groups"], 1)
        self.assertEqual(s["by_folder"]["split_deeds"] - s["current"]["split_deeds"], 1)

    def test_runs_without_third_party_packages_and_prints_counts_only(self):
        proc = self.run_stdlib_only("--ocr-root", self.ocr, "--geojson-root", self.geo)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OCR txt files: 10 | GeoJSON deeds (features with image_ids): 8", proc.stdout)
        self.assertIn("current multi-folder groups -> consistent with one GeoJSON deed: 1 | over-merged: 1", proc.stdout)
        # No roots, county names, ids or file names (single-letter toy folders are too short to check).
        for text in [self.ocr, self.geo, MN, WI] + [x for pages in OCR_PAGES.values() for rel in pages
                                                    for x in (rel, rel.split("/")[-1])]:
            self.assertNotIn(text, proc.stdout)

    def test_missing_root_fails_without_printing_a_path(self):
        missing = os.path.join(self.tmp, "does-not-exist")
        proc = self.run_stdlib_only("--ocr-root", missing, "--geojson-root", self.geo)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn(missing, proc.stdout + proc.stderr)

    def test_parse_image_ids(self):
        self.assertEqual(script.parse_image_ids("['p/a_1', 'p/a_2']"), ["p/a_1", "p/a_2"])
        self.assertEqual(script.parse_image_ids(["p/a_1", "", 3]), ["p/a_1"])
        self.assertEqual(script.parse_image_ids(""), [])
        self.assertEqual(script.parse_image_ids(None), [])
        self.assertIsNone(script.parse_image_ids("not a list"))
        self.assertIsNone(script.parse_image_ids('{"a": 1}'))


if __name__ == "__main__":
    unittest.main()
