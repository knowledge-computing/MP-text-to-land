"""
Tests for dev/matcher_ablation_val.py on synthetic toy plats and deeds only (no real data).

    mapprejudice/bin/python -m unittest discover -s test -p "test_matcher_ablation.py" -v
"""
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "dev"))
import matcher_ablation_val as ab  # noqa: E402

CFG = yaml.safe_load(ab.CONFIG.read_text())


def outcome(centroid_in=False, top1=False):
    return {"has_query": True, "top1": top1, "plat_correct": centroid_in, "centroid_in": centroid_in,
            "contains": False, "intersects": centroid_in}


class CategorizeTest(unittest.TestCase):
    def test_correct_split(self):
        self.assertEqual(ab.categorize(outcome(True, True), ["x"], ["X"], True, True, True), "correct_gold_in_gazetteer")
        self.assertEqual(ab.categorize(outcome(True, True), ["x"], ["X"], False, False, False),
                         "correct_gold_not_in_gazetteer")
        self.assertEqual(ab.categorize(outcome(True, True), ["x"], [], False, False, False), "correct_no_gold")

    def test_failure_order(self):
        # no extraction wins over everything else
        self.assertEqual(ab.categorize(outcome(), [], [], False, False, False), "no_extraction")
        self.assertEqual(ab.categorize(outcome(), [], ["X"], False, False, False), "no_extraction")
        self.assertEqual(ab.categorize(outcome(), ["x"], [], False, False, False), "gold_no_gold")
        # the oracle failing wins over name agreement
        self.assertEqual(ab.categorize(outcome(), ["x"], ["X"], False, False, True), "gold_not_in_gazetteer")
        self.assertEqual(ab.categorize(outcome(), ["x"], ["X"], False, True, True), "gold_in_gazetteer_oracle_fails")
        self.assertEqual(ab.categorize(outcome(top1=False), ["x"], ["X"], True, True, True), "right_name_no_top1")
        self.assertEqual(ab.categorize(outcome(top1=True), ["x"], ["X"], True, True, True), "right_name_wrong_plat")
        self.assertEqual(ab.categorize(outcome(top1=False), ["x"], ["X"], True, True, False), "wrong_name_no_top1")
        self.assertEqual(ab.categorize(outcome(top1=True), ["x"], ["X"], True, True, False), "wrong_name_wrong_plat")

    def test_config_matches_code(self):
        self.assertEqual(CFG["errors"]["order"], ab.FAIL_ORDER)
        self.assertEqual(CFG["errors"]["correct_split"], ab.CORRECT)
        self.assertEqual(CFG["errors"]["name_agreement"]["threshold"], 90)
        b = ab.fuzzy_config(CFG["matchers"]["B"])
        self.assertEqual((b.scorer, b.norm, b.own_county, b.aliases, b.section, b.cutoff),
                         ("ratio", "n1", True, False, "none", 70.0))
        self.assertEqual(ab.fuzzy_config(CFG["extra_rows"]["B_raw"]).norm, "raw")


class NameAgreeTest(unittest.TestCase):
    def test_threshold(self):
        self.assertTrue(ab.name_agree(["Oak Ridge."], ["OAK RIDGE"], 90))         # n1 removes case and punctuation
        self.assertTrue(ab.name_agree(["Oak Rdge"], ["OAK RIDGE"], 90))           # ratio 94.1
        self.assertFalse(ab.name_agree(["Oak Ridge Second"], ["OAK RIDGE 2ND"], 90))  # ratio 82.8
        self.assertFalse(ab.name_agree([], ["OAK RIDGE"], 90))
        self.assertTrue(ab.name_agree(["Elm", "Oak Ridge"], ["X", "OAK RIDGE"], 90))  # any pair


class ExactN1Test(unittest.TestCase):
    def setUp(self):
        plats, norm3, *_ = ab.toy_inputs()
        self.idx = ab.ExactN1Index(plats, norm3)

    def test_punctuation_and_case(self):
        self.assertEqual(self.idx.top1(["oak-ridge"], "anoka").ident, ("anoka", "OAK RIDGE"))
        self.assertIsNone(self.idx.top1(["Oak Rdge"], "anoka"))                   # exact only
        self.assertIsNone(self.idx.top1(["Pine Hill"], "anoka"))                  # own county only
        self.assertIsNone(self.idx.top1(["..."], "anoka"))                        # empty n1 form never matches

    def test_first_string_wins(self):
        self.assertEqual(self.idx.top1(["Elm Park", "Oak Ridge"], "anoka").ident, ("anoka", "ELM PARK"))


class ToyRunTest(unittest.TestCase):
    def run_toy(self, tamper=False):
        out = Path(tempfile.mkdtemp()) / "out"
        code = ab.run_toy(out, tamper=tamper)
        return code, out

    def test_toy_run_writes_and_reproduces(self):
        code, out = self.run_toy()
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["val_cells.csv", "val_errors.csv", "val_paired.csv", "val_summary.json"])

    def test_tampered_target_refuses(self):
        code, out = self.run_toy(tamper=True)
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    def test_toy_numbers(self):
        plats, norm3, deeds, gold, sources = ab.toy_inputs()
        res = ab.run(CFG, plats, norm3, deeds, gold, sources, Counter())
        cell = {(r["source"], r["matcher"], r["county"]): r for r in res["cells"]}
        ci = {m: cell[("try08_ner", m, "ALL")]["centroid_in"] for m in ab.LADDER + ("B_raw",)}
        self.assertEqual(ci, {"A": 2, "A_prime": 3, "B": 5, "C": 5, "B_raw": 0})
        self.assertEqual(cell[("oracle_add_mod", "C", "ALL")]["centroid_in"], 7)
        self.assertEqual(cell[("try08_ner", "C", "ALL")]["eligible_b"], 9)
        err = {(r["source"], r["matcher"]): r for r in res["errors"] if r["county"] == "ALL"}
        a = err[("try08_ner", "A")]
        self.assertEqual((a["right_name_no_top1"], a["wrong_name_no_top1"], a["no_extraction"], a["gold_no_gold"],
                          a["gold_not_in_gazetteer"], a["wrong_name_wrong_plat"]), (2, 1, 1, 1, 1, 1))
        for r in res["errors"]:                                                  # categories partition the deeds
            self.assertEqual(r["correct"] + r["failed"], r["eligible_b"])
        self.assertEqual(res["consistency_mismatches"], [])
        self.assertEqual(res["gold"]["ALL"]["eligible_with_gold"], 8)


if __name__ == "__main__":
    unittest.main()
