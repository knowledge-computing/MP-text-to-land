"""
Tests for how Tool 3 (scripts/run_identify_sentences_with_imageids_f_raw_txt_to_jsonl.py)
builds the `image_ids` it writes to its jsonl output.

Rule: each image_id is the OCR txt path relative to its county folder (the first folder
under --root_path), without ".txt", so it matches the 'image_ids' in the GeoJSON files.
It must not depend on how deep --root_path sits on disk. (Tool 3 previously used
`file_path.split('/', 10)[-1]`, which was only correct at one absolute depth.)

Only synthetic toy files in a temporary directory are used; no Mapping Prejudice data and
no spaCy model are needed. Run from the repository root:

    python -m unittest discover -s test -p "test_tool3_image_ids.py" -v
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

# Importing `src` reads these at import time (FOLDER_NAMES is mandatory). Set toy values so
# the tests never depend on a real .env; load_dotenv() does not override existing variables.
os.environ["FOLDER_NAMES"] = "mn-toy-county,wi-toy-county"
os.environ["GEOJSON_PATH"] = "/nonexistent/geojson/"
os.environ["OCRTXT_PATH"] = "/nonexistent/ocr/txt/"
os.environ["S3_PATH"] = "s3://nonexistent/ocr/txt/"

from scripts import run_identify_sentences_with_imageids_f_raw_txt_to_jsonl as tool3  # noqa: E402

# Synthetic toy OCR tree under <root>/ (county folder / county-relative id, no ".txt").
# The ids are made up and deliberately not modelled on real ones. Basename prefixes are
# distinct across folders so the known grouping bug (grouping by basename prefix only)
# does not affect these tests.
TOY_FILES = {
    "mn-toy-county/toy-book-a/page_001": "Lot 1 Block 2 toy page one",
    "mn-toy-county/toy-book-a/page_002": "Lot 1 Block 2 toy page two",
    "mn-toy-county/toy-abstracts/other/year/abstract_page_003": "Lot 3 toy abstract",
    "wi-toy-county/toy-milwaukee-book/toy_page_004": "Lot 4 toy page",
}

# Expected image_ids per deed: relative to the county folder.
EXPECTED = [
    ["toy-abstracts/other/year/abstract_page_003"],
    ["toy-book-a/page_001", "toy-book-a/page_002"],
    ["toy-milwaukee-book/toy_page_004"],
]


def fake_filter_relevant_sentences(text, **kwargs):
    """Stand-in for the keyword filter: one 'matched' sentence per deed."""
    return [{"sentence": text.strip(), "keyword_matches": [("lot", "lot", 100)]}]


class Tool3ImageIdTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = tmp.name
        self.out_path = os.path.join(self.base, "out", "tool3.jsonl")

    def make_root(self, root_slashes, files=TOY_FILES):
        """Create the toy tree under a root path containing exactly `root_slashes` '/'.

        For comparison, a typical absolute OCRTXT_PATH such as
        /Users/<user>/<project>/<data-dir>/<bucket>/ocr/txt has 7.
        """
        pads = root_slashes - self.base.count("/")
        if pads < 0:
            self.skipTest(f"temp dir {self.base!r} is too deep for a {root_slashes}-slash root")
        root = os.path.join(self.base, *[f"pad{i}" for i in range(pads)])
        for rel, content in files.items():
            path = os.path.join(root, rel + ".txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        self.assertEqual(root.count("/"), root_slashes)
        return root

    def run_tool3(self, root_path):
        """Run Tool 3's real main() on root_path and return the image_ids lists, sorted."""
        argv = ["tool3", "--root_path", root_path, "--output_path", self.out_path]
        with mock.patch.object(tool3, "filter_relevant_sentences", fake_filter_relevant_sentences), \
                mock.patch.object(tool3.spacy, "load", return_value=None), \
                mock.patch.object(sys, "argv", argv):
            tool3.main()
        with open(self.out_path, encoding="utf-8") as f:
            return sorted(json.loads(line)["image_ids"] for line in f)

    def test_root_like_real_setup(self):
        # 7-slash root with a trailing '/', like a typical OCRTXT_PATH and the README example.
        root = self.make_root(7)
        self.assertEqual(self.run_tool3(root + "/"), EXPECTED)

    def test_trailing_slash_on_root_path_makes_no_difference(self):
        root = self.make_root(7)
        self.assertEqual(self.run_tool3(root), EXPECTED)
        self.assertEqual(self.run_tool3(root + "/"), EXPECTED)

    def test_root_one_level_deeper(self):
        # The only depth where the old split('/', 10) rule happened to be correct.
        root = self.make_root(8)
        self.assertEqual(self.run_tool3(root), EXPECTED)

    def test_root_two_levels_deeper(self):
        # Old rule kept the county folder here ("mn-toy-county/toy-book-a/...").
        root = self.make_root(9)
        self.assertEqual(self.run_tool3(root), EXPECTED)

    def test_deeply_nested_root(self):
        root = self.make_root(14)
        self.assertEqual(self.run_tool3(root), EXPECTED)

    def test_relative_root_path(self):
        # Old rule kept only the file name here ("abstract_page_003").
        root = self.make_root(7)
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(os.path.dirname(root))
        self.assertEqual(self.run_tool3(os.path.basename(root)), EXPECTED)
        self.assertEqual(self.run_tool3("./" + os.path.basename(root)), EXPECTED)

    def test_file_directly_under_root_keeps_its_name(self):
        # No county folder to strip: fall back to the file name instead of failing.
        root = self.make_root(7, files={"loose_page_1": "Lot 5 toy page with no county folder"})
        self.assertEqual(self.run_tool3(root), [["loose_page_1"]])


if __name__ == "__main__":
    unittest.main()
