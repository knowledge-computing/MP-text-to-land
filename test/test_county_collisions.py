"""
RQ: do Tools 3, 7 and 9 need an explicit county field to prevent cross-county image_id
collisions, now that Tool 3 writes image_ids relative to the county folder?

Where county information goes:
  * group_files_by_prefix (used by Tool 3) groups pages by basename prefix only, across
    all folders and counties. Files without "_" are keyed by their full path instead.
  * Tool 3 writes {"text", "image_ids"}. The ids are county-relative and there is no
    county field, so county is dropped here.
  * Tool 7 regroups by tuple(image_ids) and writes a fixed dict, dropping extra fields.
  * Tool 9 merges by tuple(image_ids) and concatenates every other non-text field.

CountyCollisionCurrentBehaviorTests show what happens today (the evidence).
CountyFieldDesiredBehaviorTests describe the contract for a county-aware fix. They are
@unittest.expectedFailure until that fix exists; then remove the decorators and update
the current-behavior tests.

Synthetic toy data in temporary directories only. Run from the repository root:

    python -m unittest discover -s test -p "test_county_collisions.py" -v
"""

import os
import tempfile
import unittest

import toy_pipeline as tp

MN, WI = "mn-toy-county", "wi-toy-county"

# Two deeds that do not collide: distinct basenames in each county.
CONTROL = {
    f"{MN}/toy-book-c/mnonly_001": "Lot 1 of Toy Alpha",
    f"{WI}/toy-book-d/wionly_001": "Lot 2 of Toy Beta",
}
# Same county-relative id in both counties, basename without "_" (grouped by full path).
SAME_ID_NO_UNDERSCORE = {
    f"{MN}/toy-book-x/sharedpage": "Lot 11 of Toy Gamma",
    f"{WI}/toy-book-x/sharedpage": "Lot 22 of Toy Delta",
}
# Same county-relative id in both counties, basename with "_" (grouped by prefix "page").
SAME_ID_WITH_UNDERSCORE = {
    f"{MN}/toy-book-y/page_001": "Lot 31 of Toy Epsilon",
    f"{WI}/toy-book-y/page_001": "Lot 42 of Toy Zeta",
}
# Different ids, but the same basename prefix "scan" in both counties.
SAME_PREFIX_DIFFERENT_IDS = {
    f"{MN}/toy-book-m/scan_001": "Lot 51 of Toy Eta",
    f"{WI}/toy-book-w/scan_002": "Lot 62 of Toy Theta",
}


class CountyTestBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.join(tmp.name, "ocr", "txt")
        self.out = os.path.join(tmp.name, "out")

    def pipeline(self, pages):
        tp.write_tree(self.root, pages)
        return tp.run_pipeline(self.root + "/", self.out, "p")


class CountyCollisionCurrentBehaviorTests(CountyTestBase):
    """What happens today. Update these when a county field is added."""

    def test_control_distinct_basenames_stay_separate(self):
        _, _, rows9 = self.pipeline(CONTROL)
        self.assertEqual(sorted(r["image_ids"] for r in rows9),
                         [["toy-book-c/mnonly_001"], ["toy-book-d/wionly_001"]])

    def test_no_county_field_in_tool3_7_9_outputs(self):
        rows3, rows7, rows9 = self.pipeline(CONTROL)
        for r in rows3 + rows7 + rows9:
            self.assertNotIn("county", r)

    def test_same_id_without_underscore_collides_in_tool7_and_tool9(self):
        rows3, rows7, rows9 = self.pipeline(SAME_ID_NO_UNDERSCORE)
        # Tool 3 still sees two separate deeds, but gives them identical image_ids...
        self.assertEqual(len(rows3), 2)
        self.assertEqual(rows3[0]["image_ids"], rows3[1]["image_ids"])
        self.assertNotEqual(rows3[0]["text"], rows3[1]["text"])
        # ...so Tool 9 merges both counties into one record.
        self.assertEqual(len(rows9), 1)
        self.assertEqual(rows9[0]["image_ids"], ["toy-book-x/sharedpage"])
        self.assertEqual(sorted(rows9[0]["NERpredicted_LOT"]), ["Lot 11", "Lot 22"])
        self.assertEqual(sorted(rows9[0]["NERpredicted_SUBD"]), ["Toy Delta", "Toy Gamma"])

    def test_same_id_with_underscore_is_merged_already_by_tool3(self):
        rows3, _, rows9 = self.pipeline(SAME_ID_WITH_UNDERSCORE)
        # Both counties' pages form one "deed" whose image_ids repeat the same id.
        self.assertEqual(len(rows3), 2)
        for r in rows3:
            self.assertEqual(r["image_ids"], ["toy-book-y/page_001", "toy-book-y/page_001"])
        self.assertEqual(len(rows9), 1)
        self.assertEqual(sorted(rows9[0]["NERpredicted_LOT"]), ["Lot 31", "Lot 42"])

    def test_same_prefix_different_ids_forms_one_cross_county_deed(self):
        rows3, _, rows9 = self.pipeline(SAME_PREFIX_DIFFERENT_IDS)
        for r in rows3:
            self.assertEqual(r["image_ids"], ["toy-book-m/scan_001", "toy-book-w/scan_002"])
        self.assertEqual(len(rows9), 1)
        self.assertEqual(sorted(rows9[0]["NERpredicted_LOT"]), ["Lot 51", "Lot 62"])

    def test_tool7_drops_a_county_field_added_upstream(self):
        # Adding "county" to Tool 3 alone would not be enough.
        p_in, p_out = os.path.join(self.out, "in.jsonl"), os.path.join(self.out, "out7.jsonl")
        tp.write_jsonl(p_in, [{"text": "Lot 7 of Toy Iota", "image_ids": ["toy-book-q/page_001"], "county": MN}])
        rows7 = tp.run_tool7(p_in, p_out)
        self.assertEqual(len(rows7), 1)
        self.assertNotIn("county", rows7[0])

    def test_tool9_concatenates_an_extra_string_field(self):
        # Tool 9 merges every non-text/image_ids field with "+", so a plain county string
        # passed through unchanged would be corrupted on merge.
        merged = tp.tool9.merge_dictionaries([
            {"image_ids": ["toy-book-q/page_001"], "county": MN, "NERpredicted_LOT": ["Lot 1"]},
            {"image_ids": ["toy-book-q/page_001"], "county": MN, "NERpredicted_LOT": ["Lot 2"]},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["county"], MN + MN)
        self.assertEqual(merged[0]["NERpredicted_LOT"], ["Lot 1", "Lot 2"])


class CountyFieldDesiredBehaviorTests(CountyTestBase):
    """Contract for a county-aware fix: records are identified by (county, image_ids)."""

    @unittest.expectedFailure
    def test_tool3_rows_carry_their_county(self):
        rows3, _, _ = self.pipeline(CONTROL)
        for r in rows3:
            self.assertIn("county", r)
        self.assertEqual(sorted((r["county"], r["image_ids"][0]) for r in rows3),
                         [(MN, "toy-book-c/mnonly_001"), (WI, "toy-book-d/wionly_001")])

    @unittest.expectedFailure
    def test_tool7_and_tool9_keep_county(self):
        _, rows7, rows9 = self.pipeline(CONTROL)
        for r in rows7 + rows9:
            self.assertIn("county", r)
        self.assertEqual(sorted(r["county"] for r in rows9), [MN, WI])

    @unittest.expectedFailure
    def test_same_id_in_two_counties_stays_two_records(self):
        _, _, rows9 = self.pipeline(SAME_ID_NO_UNDERSCORE)
        self.assertEqual(len(rows9), 2)
        by_county = {r.get("county"): r for r in rows9}
        self.assertEqual(by_county[MN]["NERpredicted_LOT"], ["Lot 11"])
        self.assertEqual(by_county[WI]["NERpredicted_LOT"], ["Lot 22"])

    @unittest.expectedFailure
    def test_tool3_never_groups_pages_from_different_counties(self):
        for name, pages in [("same id", SAME_ID_WITH_UNDERSCORE), ("same prefix", SAME_PREFIX_DIFFERENT_IDS)]:
            with self.subTest(name):
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                root = os.path.join(tmp.name, "txt")
                tp.write_tree(root, pages)
                rows3, _, rows9 = tp.run_pipeline(root, os.path.join(tmp.name, "out"), "g")
                for r in rows3:
                    self.assertEqual(len(r["image_ids"]), 1)
                self.assertEqual(len(rows9), 2)


if __name__ == "__main__":
    unittest.main()
