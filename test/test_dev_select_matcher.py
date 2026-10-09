"""
Tests for dev/select_matcher.py on synthetic data only.

    python -m unittest discover -s test -p "test_dev_select_matcher.py" -v
"""

import os
import sys
import unittest
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dev"))
import select_matcher as sm  # noqa: E402


class SelectMatcherTest(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertEqual(sm.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(sm.mcnemar_exact(0, 6), 2 / 64)          # 0.03125
        self.assertAlmostEqual(sm.mcnemar_exact(1, 6), 2 * 8 / 128)      # 0.125
        self.assertEqual(sm.mcnemar_exact(5, 5), 1.0)

    def test_decision_rule(self):
        self.assertEqual(sm.decide({"p_exact_two_sided": 0.03, "A_only": 0, "B_only": 6}), "B")
        self.assertEqual(sm.decide({"p_exact_two_sided": 0.03, "A_only": 6, "B_only": 0}), "A")  # B worse
        self.assertEqual(sm.decide({"p_exact_two_sided": 0.05, "A_only": 0, "B_only": 6}), "A")  # not < 0.05

    def test_compare(self):
        a = {0: (1, 1), 1: (1, 0), 2: (0, 0), 3: (0, 1)}
        b = {0: (1, 1), 1: (0, 0), 2: (1, 1), 3: (1, 1)}
        r = sm.compare(a, b, 0)
        self.assertEqual((r["A_only"], r["B_only"], r["A_success"], r["B_success"]), (1, 2, 2, 3))

    def test_dedupe_by_deed_key(self):
        d = lambda c, ids, p: {"props": {"cnty_name": c, "image_ids": ids, "SUBDIVISION_p": p}}  # noqa: E731
        stats = Counter()
        out = sm.dedupe([[d("Anoka", "('a', 'b')", ["X"]), d("Anoka", "['c']", [])],
                         [d("anoka", ["b", "a"], ["X"]), d("Dakota", "['c']", [])]], stats)
        self.assertEqual(len(out), 3)
        self.assertEqual((stats["duplicate_deeds_dropped"], stats["duplicate_same_subdivision_pred"]), (1, 1))

    def test_candidates_fixed(self):
        a, b = sm.CANDIDATES["A"], sm.CANDIDATES["B"]
        self.assertEqual((a.scorer, a.norm, a.own_county, a.aliases, a.section), ("ratio", "n3", True, False, "none"))
        self.assertEqual((b.section, b.k), ("soft", 5))
        self.assertEqual(sm.ALPHA, 0.05)


if __name__ == "__main__":
    unittest.main()
