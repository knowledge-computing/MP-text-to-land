"""
Regression tests for how Tool 3 groups OCR pages into deeds.

Investigated (2026-09-29): should Tool 3 group by (county, folder, prefix) instead of
(county, prefix)? Decision: no, keep (county, prefix). The aggregate-only check
dev/check_page_grouping.py found that the one real multi-folder group is consistent with a
single GeoJSON deed, and folder-level grouping would split it without reducing over-merged
groups. These tests pin that intended behavior.

How Tool 3 groups pages into deeds:
  * group_files_by_prefix keys every page by the basename text before its last "_"
    (e.g. "scan" for "scan_001"), ignoring the folder. Files without "_" are keyed by
    their full path, so they are always single-page deeds.
  * Tool 3 then splits groups by county (commit 55003a0). The effective key is
    (county, prefix).
So a single-county group spans several folders exactly when files in different folders of
that county share the same basename prefix.

A file-name rule cannot tell "one deed stored across folders" from "two deeds whose names
share a prefix": both produce the same tree shape. Only the GeoJSON image_ids lists can,
which is what dev/check_page_grouping.py measures (counts only). Over-merged groups (also
within one folder) and GeoJSON deeds split across groups remain separate future RQs.

Synthetic toy data in temporary directories only. Run from the repository root:

    python -m unittest discover -s test -p "test_folder_grouping.py" -v
"""

import os
import tempfile
import unittest

import toy_pipeline as tp

C = "mn-toy-county"


class FolderGroupingTests(unittest.TestCase):
    """Tool 3's intended (county, prefix) grouping."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def tool3(self, pages, tag):
        root = os.path.join(self.tmp, tag, "txt")
        tp.write_tree(root, pages)
        return tp.run_tool3(root, os.path.join(self.tmp, f"{tag}.jsonl"))

    def groups(self, rows):
        return sorted({(r["county"], tuple(r["image_ids"])) for r in rows})

    def test_pages_in_one_folder_form_one_deed(self):
        rows = self.tool3({f"{C}/toy-book-a/deed_001": "Lot 1 page one",
                           f"{C}/toy-book-a/deed_002": "Lot 1 page two"}, "one_folder")
        self.assertEqual(self.groups(rows), [(C, ("toy-book-a/deed_001", "toy-book-a/deed_002"))])

    def test_same_prefix_in_two_folders_forms_one_deed(self):
        # Two folders, same basename prefix "scan": merged into one multi-folder deed.
        rows = self.tool3({f"{C}/toy-book-b/scan_001": "Lot 2 of Toy Alpha",
                           f"{C}/toy-book-c/scan_007": "Lot 3 of Toy Beta"}, "two_folders")
        self.assertEqual(self.groups(rows), [(C, ("toy-book-b/scan_001", "toy-book-c/scan_007"))])
        # Both pages' text is attributed to the same image_ids.
        self.assertEqual({r["text"] for r in rows}, {"Lot 2 of Toy Alpha", "Lot 3 of Toy Beta"})

    def test_separate_deeds_and_one_split_deed_look_identical(self):
        # Case A: two separate deeds whose names share a prefix.
        # Case B: one deed whose two pages were stored in two folders.
        # Same tree shape, so the same grouping: a file-name rule cannot tell them apart.
        a = self.tool3({f"{C}/toy-folder-1/page_001": "deed A", f"{C}/toy-folder-2/page_002": "deed B"}, "a")
        b = self.tool3({f"{C}/toy-folder-1/page_001": "deed C p1", f"{C}/toy-folder-2/page_002": "deed C p2"}, "b")
        self.assertEqual(self.groups(a), self.groups(b))
        self.assertEqual(len(self.groups(a)), 1)

    def test_multi_folder_deed_ends_up_as_one_tool9_record(self):
        root = os.path.join(self.tmp, "e2e", "txt")
        tp.write_tree(root, {f"{C}/toy-book-b/scan_001": "Lot 2 of Toy Alpha",
                             f"{C}/toy-book-c/scan_007": "Lot 3 of Toy Beta"})
        _, _, rows9 = tp.run_pipeline(root, os.path.join(self.tmp, "out"), "e2e")
        self.assertEqual(len(rows9), 1)
        self.assertEqual(sorted(rows9[0]["NERpredicted_LOT"]), ["Lot 2", "Lot 3"])
        self.assertEqual(sorted(rows9[0]["NERpredicted_SUBD"]), ["Toy Alpha", "Toy Beta"])

    def test_pages_of_a_multi_folder_deed_are_ordered_by_folder_then_name(self):
        # Pages are sorted by full path, so folder order wins over page number.
        rows = self.tool3({f"{C}/toy-z-folder/sheet_001": "p1", f"{C}/toy-a-folder/sheet_002": "p2"}, "order")
        self.assertEqual(rows[0]["image_ids"], ["toy-a-folder/sheet_002", "toy-z-folder/sheet_001"])

    def test_names_without_underscore_are_never_grouped(self):
        rows = self.tool3({f"{C}/toy-book-d/leaf": "x", f"{C}/toy-book-e/leaf": "y"}, "no_underscore")
        self.assertEqual(self.groups(rows), [(C, ("toy-book-d/leaf",)), (C, ("toy-book-e/leaf",))])

    def test_prefix_grouping_can_also_merge_within_one_folder(self):
        # Not fixed by folder-level grouping: e.g. "toy_batch_sheet_001" and "toy_batch_sheet_002"
        # share the prefix "toy_batch_sheet" even if they are different deeds.
        rows = self.tool3({f"{C}/toy-book-f/toy_batch_sheet_001": "deed X",
                           f"{C}/toy-book-f/toy_batch_sheet_002": "deed Y"}, "same_folder")
        self.assertEqual(self.groups(rows), [(C, ("toy-book-f/toy_batch_sheet_001", "toy-book-f/toy_batch_sheet_002"))])


if __name__ == "__main__":
    unittest.main()
