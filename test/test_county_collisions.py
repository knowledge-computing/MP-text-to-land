"""
Contract tests for the explicit "county" field in Tools 3-9.

image_ids are relative to the county folder, so they are only unique within a county.
Every row therefore carries "county" (the OCR county folder name, which is also the GeoJSON
folder name), and records are identified by (county, image_ids):
  * Tool 3 writes {"text", "county", "image_ids"} and never puts pages from different
    counties into one row. group_files_by_prefix groups by basename prefix across
    folders; Tool 3 splits such groups by county. Grouping within a county is unchanged.
  * Tools 4/5/6/7/8 regroup by (county, image_ids) and write "county" in every row.
  * Tool 9 merges by (county, image_ids) and writes "county" once per record; the field
    is never concatenated.
  * Old rows without "county" are treated as county None: they are grouped among
    themselves as before, written with "county": null, and never merged with rows that
    have a county.

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
# Same county-relative id in both counties, basename with "_" (same prefix "page").
SAME_ID_WITH_UNDERSCORE = {
    f"{MN}/toy-book-y/page_001": "Lot 31 of Toy Epsilon",
    f"{WI}/toy-book-y/page_001": "Lot 42 of Toy Zeta",
}
# Different ids, but the same basename prefix "scan" in both counties.
SAME_PREFIX_DIFFERENT_IDS = {
    f"{MN}/toy-book-m/scan_001": "Lot 51 of Toy Eta",
    f"{WI}/toy-book-w/scan_002": "Lot 62 of Toy Theta",
}
# A two-page deed within one county: must still be one row (grouping within a county unchanged).
TWO_PAGE_DEED = {
    f"{MN}/toy-book-t/twopage_001": "Lot 71 of Toy Iota",
    f"{MN}/toy-book-t/twopage_002": "Lot 72 of Toy Iota",
}

SHARED_IDS = ["toy-book-q/page_001"]
# Tool 3-style rows with the same image_ids in two counties plus an old-format row (no county).
MIXED_ROWS = [
    {"text": "Lot 1 of Toy Kappa County of Mnland", "county": MN, "image_ids": SHARED_IDS},
    {"text": "Lot 2 of Toy Lambda County of Wiland", "county": WI, "image_ids": SHARED_IDS},
    {"text": "Lot 3 of Toy Kappa County of Mnland", "county": MN, "image_ids": SHARED_IDS},
    {"text": "Lot 4 of Toy Mu County of Oldland", "image_ids": SHARED_IDS},
]
EXPECTED_COUNTY_BY_TEXT = {r["text"]: r.get("county") for r in MIXED_ROWS}


class CountyTestBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.out = os.path.join(tmp.name, "out")

    def pipeline(self, pages, tag="p"):
        root = os.path.join(self.tmp, tag, "ocr", "txt")
        tp.write_tree(root, pages)
        return tp.run_pipeline(root + "/", self.out, tag)

    def tool3(self, pages, tag="t3"):
        root = os.path.join(self.tmp, tag, "ocr", "txt")
        tp.write_tree(root, pages)
        return tp.run_tool3(root + "/", os.path.join(self.out, f"{tag}.jsonl"))

    def jsonl(self, name, rows):
        path = os.path.join(self.out, name)
        tp.write_jsonl(path, rows)
        return path


class Tool3CountyTests(CountyTestBase):
    def test_rows_have_text_county_image_ids(self):
        rows3 = self.tool3(CONTROL)
        for r in rows3:
            self.assertEqual(list(r), ["text", "county", "image_ids"])
        self.assertEqual(sorted((r["county"], r["image_ids"]) for r in rows3),
                         [(MN, ["toy-book-c/mnonly_001"]), (WI, ["toy-book-d/wionly_001"])])

    def test_image_ids_themselves_are_unchanged(self):
        rows3 = self.tool3({**CONTROL, **TWO_PAGE_DEED})
        self.assertEqual(sorted(tuple(r["image_ids"]) for r in rows3), [
            ("toy-book-c/mnonly_001",),
            ("toy-book-d/wionly_001",),
            ("toy-book-t/twopage_001", "toy-book-t/twopage_002"),
            ("toy-book-t/twopage_001", "toy-book-t/twopage_002"),  # one row per page sentence
        ])

    def test_never_mixes_counties_in_one_row(self):
        for name, pages in [("same id, no '_'", SAME_ID_NO_UNDERSCORE),
                            ("same id, with '_'", SAME_ID_WITH_UNDERSCORE),
                            ("same prefix, different ids", SAME_PREFIX_DIFFERENT_IDS)]:
            with self.subTest(name):
                rows3 = self.tool3(pages, tag=f"mix{len(name)}")
                expected = sorted((rel.split("/", 1)[0], [rel.split("/", 1)[1]], text) for rel, text in pages.items())
                self.assertEqual(sorted((r["county"], r["image_ids"], r["text"]) for r in rows3), expected)

    def test_grouping_within_a_county_is_unchanged(self):
        rows3 = self.tool3(TWO_PAGE_DEED)
        self.assertEqual(len(rows3), 2)
        for r in rows3:
            self.assertEqual(r["county"], MN)
            self.assertEqual(r["image_ids"], ["toy-book-t/twopage_001", "toy-book-t/twopage_002"])

    def test_file_directly_under_root_gets_county_null(self):
        rows3 = self.tool3({"loose_page_1": "Lot 5 of Toy Nu"})
        self.assertEqual(rows3, [{"text": "Lot 5 of Toy Nu", "county": None, "image_ids": ["loose_page_1"]}])


class NerToolsCountyTests(CountyTestBase):
    """Tools 4, 5, 6, 7 and 8 on the same Tool 3-style input."""

    def test_group_by_county_and_image_ids_and_keep_county(self):
        p_in = self.jsonl("mixed.jsonl", MIXED_ROWS)
        for name, run in tp.NER_TOOL_RUNNERS.items():
            with self.subTest(name):
                rows = run(p_in, os.path.join(self.out, f"{name}.jsonl"))
                self.assertEqual(len(rows), len(MIXED_ROWS))
                for r in rows:
                    self.assertEqual(list(r)[:3], ["text", "county", "image_ids"])
                    self.assertEqual(r["image_ids"], SHARED_IDS)
                    # A county-blind grouping would label every row with the first row's county.
                    self.assertEqual(r["county"], EXPECTED_COUNTY_BY_TEXT[r["text"]])

    def test_old_rows_without_county_still_work_and_output_null(self):
        old_rows = [{"text": "Lot 8 of Toy Xi", "image_ids": ["toy-book-o/page_001"]},
                    {"text": "Lot 9 of Toy Xi", "image_ids": ["toy-book-o/page_001"]}]
        p_in = self.jsonl("old.jsonl", old_rows)
        for name, run in tp.NER_TOOL_RUNNERS.items():
            with self.subTest(name):
                rows = run(p_in, os.path.join(self.out, f"old_{name}.jsonl"))
                self.assertEqual([(r["text"], r["county"], r["image_ids"]) for r in rows],
                                 [(o["text"], None, o["image_ids"]) for o in old_rows])

    def test_ner_output_fields_are_otherwise_unchanged(self):
        p_in = self.jsonl("one.jsonl", MIXED_ROWS[:1])
        rows7 = tp.run_tool7(p_in, os.path.join(self.out, "one7.jsonl"))
        self.assertEqual(set(rows7[0]) - {"text", "county", "image_ids"}, set(tp.NER_KEYS))
        self.assertEqual(rows7[0]["NERpredicted_LOT"], ["Lot 1"])
        self.assertEqual(rows7[0]["NERpredicted_CNTY"], ["County of Mnland"])
        rows4 = tp.run_tool4(p_in, os.path.join(self.out, "one4.jsonl"))
        self.assertEqual(list(rows4[0]), ["text", "county", "image_ids",
                                          "NERpredicted_STATE", "NERpredicted_CNTY", "NERpredicted_CTY"])


class Tool9CountyTests(CountyTestBase):
    def test_merges_by_county_and_image_ids(self):
        rows = [dict(r, **{"NERpredicted_LOT": [r["text"].split(" of ")[0]]}) for r in MIXED_ROWS]
        merged = {r["county"]: r for r in tp.run_tool9(self.jsonl("m7.jsonl", rows), os.path.join(self.out, "m9.jsonl"))}
        self.assertEqual(set(merged), {MN, WI, None})
        self.assertEqual(merged[MN]["NERpredicted_LOT"], ["Lot 1", "Lot 3"])
        self.assertEqual(merged[WI]["NERpredicted_LOT"], ["Lot 2"])
        self.assertEqual(merged[None]["NERpredicted_LOT"], ["Lot 4"])  # old row: not merged into MN or WI

    def test_county_written_once_and_never_concatenated(self):
        rows = [{"text": "a", "county": MN, "image_ids": SHARED_IDS, "NERpredicted_LOT": ["Lot 1"]},
                {"text": "b", "county": MN, "image_ids": SHARED_IDS, "NERpredicted_LOT": ["Lot 2"]}]
        out = tp.run_tool9(self.jsonl("c7.jsonl", rows), os.path.join(self.out, "c9.jsonl"))
        self.assertEqual(out, [{"county": MN, "image_ids": SHARED_IDS, "NERpredicted_LOT": ["Lot 1", "Lot 2"]}])

    def test_old_rows_without_county_merge_as_before_with_county_null(self):
        rows = [{"text": "a", "image_ids": SHARED_IDS, "NERpredicted_LOT": ["Lot 1"]},
                {"text": "b", "image_ids": SHARED_IDS, "NERpredicted_LOT": ["Lot 2"]},
                {"text": "c", "image_ids": ["toy-book-r/page_002"], "NERpredicted_LOT": []}]
        out = tp.run_tool9(self.jsonl("o7.jsonl", rows), os.path.join(self.out, "o9.jsonl"))
        self.assertEqual(out, [{"county": None, "image_ids": SHARED_IDS, "NERpredicted_LOT": ["Lot 1", "Lot 2"]},
                               {"county": None, "image_ids": ["toy-book-r/page_002"], "NERpredicted_LOT": []}])


class EndToEndCountyTests(CountyTestBase):
    def test_control_distinct_deeds_stay_separate(self):
        _, _, rows9 = self.pipeline(CONTROL)
        self.assertEqual(sorted((r["county"], r["image_ids"]) for r in rows9),
                         [(MN, ["toy-book-c/mnonly_001"]), (WI, ["toy-book-d/wionly_001"])])

    def test_same_id_in_two_counties_gives_separate_records(self):
        for name, pages in [("no '_'", SAME_ID_NO_UNDERSCORE), ("with '_'", SAME_ID_WITH_UNDERSCORE)]:
            with self.subTest(name):
                _, rows7, rows9 = self.pipeline(pages, tag=f"e2e{len(name)}")
                self.assertEqual(len(rows9), 2)
                by_county = {r["county"]: r for r in rows9}
                self.assertEqual(set(by_county), {MN, WI})
                for county in (MN, WI):
                    rel, text = next((k, v) for k, v in pages.items() if k.startswith(county + "/"))
                    self.assertEqual(by_county[county]["image_ids"], [rel.split("/", 1)[1]])
                    self.assertEqual(by_county[county]["NERpredicted_LOT"], [text.split(" of ")[0]])
                self.assertEqual({r["county"] for r in rows7}, {MN, WI})

    def test_same_prefix_in_two_counties_gives_separate_records(self):
        _, _, rows9 = self.pipeline(SAME_PREFIX_DIFFERENT_IDS)
        self.assertEqual(sorted((r["county"], r["image_ids"], r["NERpredicted_LOT"]) for r in rows9),
                         [(MN, ["toy-book-m/scan_001"], ["Lot 51"]), (WI, ["toy-book-w/scan_002"], ["Lot 62"])])

    def test_tool8_to_tool9_keeps_county(self):
        root = os.path.join(self.tmp, "t8", "ocr", "txt")
        tp.write_tree(root, SAME_ID_NO_UNDERSCORE)
        p3, p8, p9 = (os.path.join(self.out, f"t8_{n}.jsonl") for n in (3, 8, 9))
        tp.run_tool3(root, p3)
        rows8 = tp.run_tool8(p3, p8)
        self.assertEqual(sorted(r["county"] for r in rows8), [MN, WI])
        rows9 = tp.run_tool9(p8, p9)
        self.assertEqual(sorted(r["county"] for r in rows9), [MN, WI])
        self.assertEqual({r["county"]: r["NERpredicted_LOT"][0][0] for r in rows9}, {MN: "Lot 11", WI: "Lot 22"})

    def test_mixed_old_and_new_rows_with_same_ids_do_not_merge(self):
        # e.g. an old Tool 7 file (no county) concatenated with a new one.
        new_rows = [dict(r, NERpredicted_LOT=["Lot 1"]) for r in MIXED_ROWS[:1]]
        old_rows = [{"text": "old", "image_ids": SHARED_IDS, "NERpredicted_LOT": ["Lot 99"]}]
        out = tp.run_tool9(self.jsonl("mix7.jsonl", old_rows + new_rows), os.path.join(self.out, "mix9.jsonl"))
        self.assertEqual(sorted((str(r["county"]), r["NERpredicted_LOT"]) for r in out),
                         [("None", ["Lot 99"]), (MN, ["Lot 1"])])


if __name__ == "__main__":
    unittest.main()
