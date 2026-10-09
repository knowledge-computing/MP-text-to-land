"""
Tests for src/matcher on synthetic toy plats and names only.

    python -m unittest discover -s test -p "test_matcher.py" -v
"""

import unittest

from shapely.geometry import box

from src.matcher import MatchConfig, NameNormalizer, PlatIndex, geom_metrics, learn_rules, n1


def toy_index():
    pairs = [("Oak Ridge Second Addition", "OAK RIDGE 2ND ADD")] * 5 + \
            [("Elm Subdivision", "ELM SUB")] * 5 + [("Lot 7 Plat", "Lot 8 Plat")] * 9
    norm3 = NameNormalizer(learn_rules(pairs, min_freq=5))
    plats = [
        {"county": "anoka", "name": "OAK RIDGE 2ND ADD", "aliases": [], "geometry": box(0, 0, 10, 10), "sections": {(1, 31, 24)}},
        {"county": "anoka", "name": "Oak Ridge 3rd Addition", "aliases": [], "geometry": box(20, 0, 30, 10), "sections": {(2, 31, 24)}},
        {"county": "anoka", "name": "OAK RIDGE 2ND ADD", "aliases": [], "geometry": box(10, 0, 12, 10), "sections": {(1, 31, 24)}},
        {"county": "dakota", "name": "Oak Ridge Second Addition", "aliases": ["ORSA"], "geometry": box(100, 0, 110, 10), "sections": {(9, 27, 22)}},
        {"county": "anoka", "name": "Maple Park", "aliases": ["Maple Pk"], "geometry": box(0, 50, 10, 60), "sections": {(5, 31, 24)}},
        {"county": "anoka", "name": "Maple Park Replat", "aliases": [], "geometry": box(0, 70, 10, 80), "sections": {(6, 31, 24)}},
    ]
    return PlatIndex(plats, norm3)


class NamesTest(unittest.TestCase):
    def test_n1(self):
        self.assertEqual(n1(" Jane's  2nd-Addition. "), "JANE S 2ND ADDITION")

    def test_learned_classes(self):
        rules = learn_rules([("Oak Second Addition", "OAK 2ND ADD")] * 5 + [("A Lot 7", "A Lot 8")] * 9, min_freq=5)
        norm = NameNormalizer(rules)
        self.assertEqual(norm("oak second addition"), norm("OAK 2ND ADD"))
        self.assertEqual(norm("Third"), "3RD")                 # ordinal table always applies
        self.assertNotIn("7", rules["canon"])                   # number swaps are not spelling variants
        self.assertNotEqual(norm("Lot 7"), norm("Lot 8"))

    def test_rare_rule_ignored(self):
        rules = learn_rules([("X Terrace", "X TER")] * 4, min_freq=5)
        self.assertNotEqual(NameNormalizer(rules)("Terrace"), NameNormalizer(rules)("TER"))


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.idx = toy_index()

    def test_case_sensitive_raw_fails_n1_works(self):
        raw = MatchConfig(norm="raw", own_county=False)
        self.assertIsNone(self.idx.top1(["oak ridge 2nd add"], "anoka", raw))
        m = self.idx.top1(["oak ridge 2nd add"], "anoka", MatchConfig(norm="n1"))
        self.assertEqual(m.ident, ("anoka", "OAK RIDGE 2ND ADD"))
        self.assertEqual(sorted(m.plats), [0, 2])               # both polygons of the identity

    def test_own_county_and_n3(self):
        m = self.idx.top1(["Oak Ridge Second Addition"], "anoka", MatchConfig(norm="n3"))
        self.assertEqual(m.ident, ("anoka", "OAK RIDGE 2ND ADD"))
        m_all = self.idx.top1(["Oak Ridge Second Addition"], "anoka", MatchConfig(norm="n1", own_county=False))
        self.assertEqual(m_all.ident[0], "dakota")

    def test_aliases(self):
        cfg = MatchConfig(norm="n1", aliases=True)
        self.assertEqual(self.idx.top1(["MAPLE PK"], "anoka", cfg).ident, ("anoka", "MAPLE PARK"))

    def test_section_hard_and_soft(self):
        q = ["Oak Ridge 4th Addition"]                          # 2ND and 3RD score alike
        base = self.idx.top1(q, "anoka", MatchConfig())
        hard = self.idx.top1(q, "anoka", MatchConfig(section="hard"), frozenset({(2, 31, 24)}))
        soft = self.idx.top1(q, "anoka", MatchConfig(section="soft", k=5), frozenset({(2, 31, 24)}))
        self.assertEqual(hard.ident, ("anoka", "OAK RIDGE 3RD ADDITION"))
        self.assertEqual(soft.ident, ("anoka", "OAK RIDGE 3RD ADDITION"))
        self.assertEqual(soft.final, soft.score + 5)
        # hard filter with no surviving candidate falls back to no filter
        self.assertEqual(self.idx.top1(q, "anoka", MatchConfig(section="hard"), frozenset({(36, 1, 1)})).ident, base.ident)

    def test_cutoff_and_any_candidate(self):
        self.assertIsNone(self.idx.top1(["zzzz"], "anoka", MatchConfig()))
        self.assertFalse(self.idx.any_candidate(["zzzz"], "anoka", MatchConfig()))
        self.assertTrue(self.idx.any_candidate(["OAK RIDGE 2ND ADD"], "anoka", MatchConfig(norm="raw", own_county=False)))

    def test_tie_goes_to_first_prediction(self):
        m = self.idx.top1(["Maple Park", "Maple Park Replat"], "anoka", MatchConfig())
        self.assertEqual(m.ident, ("anoka", "MAPLE PARK"))


class GeometryTest(unittest.TestCase):
    def test_metrics(self):
        plat, deed = box(0, 0, 10, 10), box(8, 0, 12, 2)        # half of deed inside
        g = geom_metrics(plat, deed)
        self.assertFalse(g["contains"])
        self.assertTrue(g["intersects"])
        self.assertFalse(g["centroid_in"])                     # centroid on x=10 boundary -> not inside
        self.assertAlmostEqual(g["covered"], 0.5)
        self.assertAlmostEqual(g["iou"], 4 / (100 + 4))
        far = geom_metrics(plat, box(20, 0, 22, 2))
        self.assertEqual((far["intersects"], far["covered"], far["iou"]), (False, 0.0, 0.0))
        self.assertAlmostEqual(far["centroid_m"], 11.0)
        inside = geom_metrics(plat, box(1, 1, 2, 2))
        self.assertTrue(inside["contains"] and inside["centroid_in"])
        self.assertEqual(inside["centroid_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
