"""
Tool 3 grouping mismatches against GeoJSON deeds, and how dev/check_page_grouping.py
classifies them (counts only).

Definitions used by the checker. A GeoJSON "deed" is one feature's image_ids list, and
a "group" is one Tool 3 (county, prefix) page group:
  * over-merged group: a group with >=1 page listed in GeoJSON, where no single deed
    contains all of the group's GeoJSON-listed pages. Pages no deed lists are ignored.
      - disjoint: the deeds it touches share no ids (separate deeds merged by prefix).
      - overlapping: some page belongs to >1 of those deeds (GeoJSON itself chains them).
  * split deed: a deed with >=1 page present as an OCR file whose present pages fall into
    >1 group. Missing pages are ignored. Primary reason: some pages have no "_" (each such
    page is its own group), or the pages have different basename prefixes (differing only
    in digits, or in other text).

Synthetic toy data in temporary directories only. Run from the repository root:

    python -m unittest discover -s test -p "test_grouping_mismatches.py" -v
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

import toy_pipeline as tp
from test_dev_collision_script import BLOCKED, RUNNER

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "dev", "check_page_grouping.py")
spec = importlib.util.spec_from_file_location("check_page_grouping", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

MN, WI = "mn-toy-county", "wi-toy-county"
# One synthetic case per category: (OCR pages, GeoJSON deeds as image_ids lists).
CASES = {
    "ok":            (MN, ["a/ok_001", "a/ok_002"], [["a/ok_001", "a/ok_002"]]),
    "disjoint":      (MN, ["b/same_001", "b/same_002"], [["b/same_001"], ["b/same_002"]]),
    "overlapping":   (MN, ["c/chain_001", "c/chain_002", "c/chain_003"],
                      [["c/chain_001", "c/chain_002"], ["c/chain_002", "c/chain_003"]]),
    "three_deeds":   (MN, ["d/tri_001", "d/tri_002", "d/tri_003", "d/tri_004"],   # tri_004: unreferenced
                      [["d/tri_001"], ["d/tri_002"], ["d/tri_003"]]),
    "digits_prefix": (MN, ["e/toydoc_101_sheet_1", "e/toydoc_102_sheet_1"],
                      [["e/toydoc_101_sheet_1", "e/toydoc_102_sheet_1"]]),
    "text_prefix":   (MN, ["f/front_1", "f/back_1"], [["f/front_1", "f/back_1"]]),
    "no_underscore": (MN, ["g/leafa", "g/leafb"], [["g/leafa", "g/leafb"]]),
    "folders_missing": (MN, ["h/alpha_1", "i/beta_1"], [["h/alpha_1", "i/beta_1", "x/gone_1"]]),
    "three_groups":  (MN, ["j/p_1", "j/q_1", "j/r_1"], [["j/p_1", "j/q_1", "j/r_1"]]),
    "split_and_over": (MN, ["k/mix_1", "k/mix_2", "k/solo_1"], [["k/mix_1", "k/solo_1"], ["k/mix_2"]]),
    "duplicate_deed": (WI, ["l/dup_001"], [["l/dup_001"], ["l/dup_001"]]),
    "repeated_id":   (WI, ["m/twice_001"], [["m/twice_001", "m/twice_001"]]),
    "all_missing":   (WI, [], [["z/none_001"]]),
    "unreferenced":  (WI, ["n/orphan_001"], []),
}
EXPECTED_DIAGNOSIS = {
    "hygiene": {"deeds": 19, "deeds_with_pages": 18, "deeds_some_pages_missing": 1, "deeds_all_pages_missing": 1,
                "ids_in_multiple_deeds": 2, "deeds_sharing_ids": 4, "exact_duplicate_deeds": 1,
                "ids_repeated_within_deed": 1, "unreferenced_ocr_pages": 2},
    "groups": {"groups": 20, "labeled_groups": 19, "groups_touching_multiple_deeds": 5,
               "groups_with_unreferenced_and_geojson_pages": 1, "groups_with_only_unreferenced_pages": 1},
    "over_merged": {"total": 4, "disjoint_deeds": 3, "overlapping_deeds": 1, "touching_2_deeds": 3,
                    "touching_3plus_deeds": 1, "multi_folder": 0, "with_unreferenced_pages": 1},
    "split": {"total": 6, "no_underscore_pages": 1, "different_prefixes": 5, "prefixes_differ_only_in_digits": 1,
              "prefixes_differ_in_text": 4, "same_prefix": 0, "multi_folder": 1, "into_2_groups": 5,
              "into_3plus_groups": 1, "in_over_merged_group": 1, "some_pages_missing": 1},
}


def write_case_tree(base, cases):
    ocr, geo = os.path.join(base, "ocr", "txt"), os.path.join(base, "geojson")
    features = {MN: [], WI: []}
    for county, pages, deeds in cases.values():
        tp.write_tree(os.path.join(ocr, county), {p: "toy text" for p in pages})
        features[county] += [{"type": "Feature", "geometry": None, "properties": {"image_ids": str(d)}} for d in deeds]
    for county, feats in features.items():
        os.makedirs(os.path.join(ocr, county), exist_ok=True)
        os.makedirs(os.path.join(geo, county), exist_ok=True)
        with open(os.path.join(geo, county, "toy.geojson"), "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
    return ocr, geo


class CheckerDiagnosisTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.ocr, self.geo = write_case_tree(tmp.name, CASES)

    def test_each_category_is_counted(self):
        self.assertEqual(checker.summary(self.ocr, self.geo)["diagnosis"], EXPECTED_DIAGNOSIS)

    def test_diagnosis_totals_match_the_headline_counts(self):
        s = checker.summary(self.ocr, self.geo)
        d, cur = s["diagnosis"], s["current"]
        self.assertEqual(d["over_merged"]["total"], cur["over_merged_groups"])
        self.assertEqual(d["split"]["total"], cur["split_deeds"])
        self.assertEqual(d["over_merged"]["disjoint_deeds"] + d["over_merged"]["overlapping_deeds"], d["over_merged"]["total"])
        self.assertEqual(d["split"]["no_underscore_pages"] + d["split"]["different_prefixes"] + d["split"]["same_prefix"],
                         d["split"]["total"])

    def test_cli_prints_the_diagnosis_as_counts_only(self):
        proc = subprocess.run([sys.executable, "-c", RUNNER, ",".join(BLOCKED), SCRIPT,
                               "--ocr-root", self.ocr, "--geojson-root", self.geo],
                              capture_output=True, text=True, cwd=self.tmp,
                              env={**os.environ, "OCRTXT_PATH": "", "GEOJSON_PATH": ""})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("over-merged groups: 4", proc.stdout)
        self.assertIn("merging disjoint deeds (no shared ids): 3 | merging overlapping deeds (deeds share an id): 1",
                      proc.stdout)
        self.assertIn("GeoJSON deeds split across >1 group: 6", proc.stdout)
        self.assertIn("differ only in digits: 1 | differ in other text: 4", proc.stdout)
        for county, pages, deeds in CASES.values():
            for rel in pages + [i for d in deeds for i in d]:
                self.assertNotIn(rel, proc.stdout)
                self.assertNotIn(rel.split("/")[-1], proc.stdout)
        for text in (self.ocr, self.geo, MN, WI):
            self.assertNotIn(text, proc.stdout)


class Tool3MismatchMechanismTests(unittest.TestCase):
    """The same mechanisms seen through Tool 3's real main()."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def tool3_groups(self, pages, tag):
        root = os.path.join(self.tmp, tag, "txt")
        tp.write_tree(root, {f"{MN}/{p}": f"Lot {n} toy page" for n, p in enumerate(pages)})
        rows = tp.run_tool3(root, os.path.join(self.tmp, f"{tag}.jsonl"))
        return sorted({tuple(r["image_ids"]) for r in rows})

    def test_one_deed_whose_prefixes_differ_only_in_digits_is_split(self):
        self.assertEqual(self.tool3_groups(["e/toydoc_101_sheet_1", "e/toydoc_102_sheet_1"], "digits"),
                         [("e/toydoc_101_sheet_1",), ("e/toydoc_102_sheet_1",)])

    def test_one_deed_whose_prefixes_differ_in_text_is_split(self):
        self.assertEqual(self.tool3_groups(["f/front_1", "f/back_1"], "text"), [("f/back_1",), ("f/front_1",)])

    def test_pages_without_underscore_are_each_their_own_group(self):
        self.assertEqual(self.tool3_groups(["g/leafa", "g/leafb"], "no_us"), [("g/leafa",), ("g/leafb",)])

    def test_separate_deeds_sharing_a_prefix_are_merged(self):
        self.assertEqual(self.tool3_groups(["b/same_001", "b/same_002"], "merged"), [("b/same_001", "b/same_002")])


if __name__ == "__main__":
    unittest.main()
