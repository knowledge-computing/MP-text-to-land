"""
End-to-end check of the Tool 3 -> Tool 7 -> Tool 9 flow on synthetic toy data.

  Tool 3  scripts/run_identify_sentences_with_imageids_f_raw_txt_to_jsonl.py
          OCR txt tree -> {"text", "county", "image_ids"} per kept sentence
  Tool 7  scripts/run_identify_all_geo_parcel.py
          adds NERpredicted_* lists; regroups rows by (county, image_ids), writes ids sorted
  Tool 9  scripts/run_combine_ner_results.py
          merges rows with the same (county, image_ids) into one record per deed

County handling in detail (including Tools 4/5/6/8 and old rows without county) is
covered by test_county_collisions.py.

Each tool's real main() runs; see toy_pipeline.py for the stubs. Set MP_RUN_MODEL_TESTS=1
to also run one smoke test with the real bundled models in src/models/ (repository files,
not Mapping Prejudice data). Only synthetic files in temporary directories are used.
Run from the repository root:

    python -m unittest discover -s test -p "test_pipeline_tool3_7_9.py" -v
"""

import json
import os
import re
import tempfile
import unittest

import toy_pipeline as tp

# Synthetic OCR tree: <county folder>/<county-relative id>, one text per page.
# Basename prefixes are distinct across folders and counties (pages are grouped by basename
# prefix only; see test_county_collisions.py).
TOY_PAGES = {
    "mn-toy-county/toy-book-a/page_001": "Lot 1 Block 2 of Toy Addition, page one.",
    "mn-toy-county/toy-book-a/page_002": "Lot 2 Block 2 of Toy Addition, page two.",
    "mn-toy-county/toy-abstracts/other/year/abstract_page_003": "Lot 3 Block 9 of Toy Park.",
    "wi-toy-county/toy-milwaukee-book/toy_page_004": "Lot 4 Block 1 of Toy Heights.",
}

# GeoJSON-style ids (relative to the county folder), grouped per deed, with each deed's county.
EXPECTED_COUNTY_BY_GROUP = {
    ("toy-book-a/page_001", "toy-book-a/page_002"): "mn-toy-county",
    ("toy-abstracts/other/year/abstract_page_003",): "mn-toy-county",
    ("toy-milwaukee-book/toy_page_004",): "wi-toy-county",
}
EXPECTED_GROUPS = set(EXPECTED_COUNTY_BY_GROUP)


class PipelineTestBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = tmp.name
        self.out = os.path.join(self.base, "out")

    def make_root(self, root_slashes):
        """Write TOY_PAGES under a root path containing exactly `root_slashes` '/'."""
        pads = root_slashes - self.base.count("/")
        if pads < 1:
            self.skipTest(f"temp dir {self.base!r} is too deep for a {root_slashes}-slash root")
        root = os.path.join(self.base, "ocr", *[f"pad{i}" for i in range(pads - 1)])
        tp.write_tree(root, TOY_PAGES)
        self.assertEqual(root.count("/"), root_slashes)
        return root


class PipelineImageIdTests(PipelineTestBase):
    def test_tool3_ids_are_county_relative(self):
        rows3, _, _ = tp.run_pipeline(self.make_root(7) + "/", self.out, "t3")
        self.assertEqual({tuple(r["image_ids"]) for r in rows3}, EXPECTED_GROUPS)
        self.assertEqual(len(rows3), len(TOY_PAGES))  # one sentence per page
        for r in rows3:
            self.assertEqual(r["county"], EXPECTED_COUNTY_BY_GROUP[tuple(r["image_ids"])])

    def test_tool7_passes_ids_through_unchanged(self):
        rows3, rows7, _ = tp.run_pipeline(self.make_root(7) + "/", self.out, "t7")
        self.assertEqual(len(rows7), len(rows3))
        self.assertEqual(sorted((r["text"], r["county"], tuple(r["image_ids"])) for r in rows7),
                         sorted((r["text"], r["county"], tuple(r["image_ids"])) for r in rows3))
        for r in rows7:
            self.assertEqual(r["image_ids"], sorted(r["image_ids"]))
            self.assertEqual(set(r) - {"text", "county", "image_ids"}, set(tp.NER_KEYS))

    def test_tool9_gives_one_record_per_deed_with_merged_entities(self):
        _, _, rows9 = tp.run_pipeline(self.make_root(7) + "/", self.out, "t9")
        by_ids = {tuple(r["image_ids"]): r for r in rows9}
        self.assertEqual(set(by_ids), EXPECTED_GROUPS)
        self.assertEqual(len(rows9), len(EXPECTED_GROUPS))
        for ids, r in by_ids.items():
            self.assertEqual(r["county"], EXPECTED_COUNTY_BY_GROUP[ids])
        two_page = by_ids[("toy-book-a/page_001", "toy-book-a/page_002")]
        self.assertEqual(two_page["NERpredicted_LOT"], ["Lot 1", "Lot 2"])
        self.assertEqual(two_page["NERpredicted_SUBD"], ["Toy Addition", "Toy Addition"])
        self.assertEqual(by_ids[("toy-milwaukee-book/toy_page_004",)]["NERpredicted_BLOCK"], ["Block 1"])
        for r in rows9:
            self.assertNotIn("text", r)

    def test_pipeline_result_does_not_depend_on_root_depth(self):
        # With the old split('/', 10) rule these three runs gave three different id sets.
        results = [tp.run_pipeline(self.make_root(7) + "/", self.out, "d7")[2]]
        root9 = os.path.join(self.base, "deeper", "x", "y")
        tp.write_tree(root9, TOY_PAGES)
        results.append(tp.run_pipeline(root9, self.out, "d9")[2])
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(os.path.dirname(root9))
        results.append(tp.run_pipeline(os.path.basename(root9), self.out, "rel")[2])
        canon = [sorted(json.dumps(r, sort_keys=True) for r in res) for res in results]
        self.assertEqual(canon[0], canon[1])
        self.assertEqual(canon[0], canon[2])


class NoOldIdRuleTests(unittest.TestCase):
    def test_no_active_split_slash_10_in_scripts_or_src(self):
        # Guard against reintroducing the depth-dependent split('/', 10) id rule.
        pat = re.compile(r"""(?<![r\w])split\(\s*['"]/['"]\s*,\s*10\s*\)""")
        hits = []
        for folder in ("scripts", "src"):
            for dirpath, _, files in os.walk(os.path.join(tp.REPO_ROOT, folder)):
                for fn in files:
                    if fn.endswith(".py"):
                        path = os.path.join(dirpath, fn)
                        with open(path, encoding="utf-8") as f:
                            for n, line in enumerate(f, 1):
                                if pat.search(line.split("#", 1)[0]):
                                    hits.append(f"{os.path.relpath(path, tp.REPO_ROOT)}:{n}")
        self.assertEqual(hits, [])


@unittest.skipUnless(os.environ.get("MP_RUN_MODEL_TESTS") == "1",
                     "set MP_RUN_MODEL_TESTS=1 to run Tool 7 with the real bundled spaCy models")
class PipelineWithBundledModelsTests(PipelineTestBase):
    def test_real_models_keep_ids_and_output_shape(self):
        _, rows7, rows9 = tp.run_pipeline(self.make_root(7) + "/", self.out, "real", real_models=True)
        self.assertEqual({tuple(r["image_ids"]) for r in rows7}, EXPECTED_GROUPS)
        self.assertEqual({tuple(r["image_ids"]) for r in rows9}, EXPECTED_GROUPS)
        for r in rows9:
            self.assertEqual(set(r) - {"county", "image_ids"}, set(tp.NER_KEYS))
            self.assertEqual(r["county"], EXPECTED_COUNTY_BY_GROUP[tuple(r["image_ids"])])
            self.assertTrue(all(isinstance(r[k], list) for k in tp.NER_KEYS))


if __name__ == "__main__":
    unittest.main()
