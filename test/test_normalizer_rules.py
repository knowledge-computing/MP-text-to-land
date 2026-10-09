"""
Tests for the Phase 1 rule-based PLSS normalizer (numbers, candidates, constraints, validate, select)
and the regex v1 baseline wrapper. Synthetic data only.

    mapprejudice/bin/python -m unittest discover -s test -p "test_normalizer_rules.py" -v
"""

import importlib.util
import os
import unittest

from src.normalizer import TRS, Span, synth
from src.normalizer.baselines import RegexV1Normalizer
from src.normalizer.candidates import candidate_values, read_span, read_spans
from src.normalizer.constraints import Constraints
from src.normalizer.numbers import parse_number_words
from src.normalizer.rule_based import RuleBasedNormalizer

TOY = synth.toy_valid()
C = Constraints.from_trs(TOY)
NORM = RuleBasedNormalizer(C)


def words(text):
    return [w.value for w in parse_number_words(text)]


def vals(label, text):
    return [(c.value, c.kind) for c in read_span(Span(label, text)).values]


def top(spans, county="toyka", state="MN", norm=NORM):
    r = norm.normalize(spans, county, state)
    return (r.top.T, r.top.R, r.top.S, r.top.Rdir) if r.top else r.reason


class NumberWordTests(unittest.TestCase):
    def test_exact_and_compound(self):
        self.assertEqual(words("Township One hundred and six North"), [106])
        self.assertEqual(words("Range Twenty-four (24) West"), [24])
        self.assertEqual(words("Section Thirty one"), [31])
        self.assertEqual(words("twentyfour"), [24])
        self.assertEqual(words("onehundredseventeen"), [117])
        self.assertEqual(words("SECTIONTwenty-Seven (27)"), [27])
        self.assertEqual(words("Twenty North"), [20])

    def test_ocr_damage(self):
        self.assertEqual(words("f0rty"), [40])
        self.assertEqual(words("e1even"), [11])
        self.assertEqual(words("Sec eightecn"), [18])
        self.assertEqual(words("Twp Sev en(7) North"), [7])
        self.assertEqual(words("twenty-t hree"), [23])
        self.assertEqual(words("Twenty-Nne"), [29])
        self.assertEqual(words("Tvventynne"), [29])
        self.assertEqual(words("twentywo"), [22])
        self.assertEqual(words("one hundred ancl six"), [106])

    def test_structure_and_ordinals(self):
        self.assertEqual(words("Sections One ancl Seven"), [1, 7])
        self.assertEqual(words("twelve twelve"), [12, 12])
        self.assertEqual(words("of the Fourth Principal Meridian"), [])
        self.assertEqual(words("West of the Fifth P.M."), [])


class CandidateTests(unittest.TestCase):
    def test_agreement_and_disagreement(self):
        self.assertEqual(vals("S", "Section TWELVE (12)"), [(12, "words+digits")])
        got = read_span(Span("R", "Range Twenty-four (42) West")).values
        self.assertEqual([(c.value, c.disagree, c.hint) for c in got], [(24, True, "R"), (42, True, "R")])
        self.assertGreater(got[0].conf, got[1].conf)  # words preferred on disagreement

    def test_compact_and_glued(self):
        self.assertEqual(vals("T", "T31N")[0], (31, "compact"))
        self.assertEqual(vals("T", "TOWNSHIP27"), [(27, "compact")])
        self.assertEqual(vals("T", "Town5bip7"), [(7, "compact")])
        self.assertEqual(vals("R", "R24W"), [(24, "compact")])
        self.assertEqual(vals("T", "Township nnmbered 2B")[0], (28, "digits_repaired"))
        self.assertEqual(read_span(Span("T", "Township 8north")).dirs[0][0], "N")

    def test_skips(self):
        self.assertEqual(vals("S", "NE 1/4 of Section 5"), [(5, "digits")])
        self.assertEqual(vals("R", "Range 24 West of the 4th P.M."), [(24, "digits")])
        self.assertEqual(vals("S", "SEC1ON nineteen"), [(19, "words")])  # OCR'd label word is not 10

    def test_hints_and_split_digits(self):
        got = read_span(Span("S", "Section 5 of Township 31")).values
        self.assertEqual([(c.value, c.hint) for c in got], [(5, "S"), (31, "T")])
        got = read_span(Span("S", "Sections 5 and 6")).values
        self.assertEqual([(c.value, c.joined) for c in got], [(5, False), (6, True)])
        self.assertIn((22, "split_digits"), vals("R", "Range 2 2 East"))

    def test_directions(self):
        self.assertEqual([d[0] for d in read_span(Span("R", "R. 22 vve5t")).dirs], ["W"])
        self.assertEqual([d[0] for d in read_span(Span("T", "Twp SEVEN (7) NOORTH")).dirs], ["N"])

    def test_empty_spans_do_not_shift_pairing(self):
        rs = read_spans([Span("R", "Government"), Span("R", "Range 22")])
        self.assertEqual([r.index for r in rs], [-1, 0])
        self.assertEqual(candidate_values(rs, "R"), {22})


class ConstraintTests(unittest.TestCase):
    def test_exact_sets_not_min_max(self):
        sc = C.scope("toyka", "MN", "county")
        self.assertTrue(sc.section_ok(27, 22, 5, "W"))
        self.assertTrue(sc.section_ok(114, 22, 5, "W"))
        self.assertFalse(sc.section_ok(50, 22, 5, "W"))   # between the two meridians' T ranges
        self.assertFalse(sc.section_ok(28, 23, 33, "W"))  # partial township
        self.assertTrue(sc.section_ok(27, 22, 5, ""))
        self.assertEqual(sc.default_rdir(), "W")

    def test_levels(self):
        self.assertEqual(C.scope("toywaukee", "WI", "county").directions(), ["E"])
        self.assertTrue(C.scope("toyka", "MN", "state").name.startswith("state"))
        none = C.scope("toyka", "MN", "none")
        self.assertTrue(none.section_ok(50, 22, 5))
        self.assertFalse(none.section_ok(50, 22, 37))
        self.assertTrue(C.scope("unknown", "MN", "county").name.startswith("state"))


class NormalizerTests(unittest.TestCase):
    def test_clean_and_default_direction(self):
        self.assertEqual(top([Span("S", "Section 5"), Span("T", "Township 27 North"), Span("R", "Range 22 West")]),
                         (27, 22, 5, "W"))
        self.assertEqual(top([Span("S", "Sec. 5"), Span("T", "T. 7"), Span("R", "R. 21")], "toywaukee", "WI"),
                         (7, 21, 5, "E"))

    def test_plss_settles_disagreement(self):
        # words say 24, digits say 42; only R22/23 exist, so neither... use R where words are wrong
        spans = [Span("S", "Section 5"), Span("T", "Township 27"), Span("R", "Range Thirty-two (23)")]
        self.assertEqual(top(spans), (27, 23, 5, "W"))

    def test_direction_conflict_uses_county(self):
        self.assertEqual(top([Span("S", "Sec 5"), Span("T", "T 27"), Span("R", "R 22 East")]), (27, 22, 5, "W"))

    def test_township_only_and_reasons(self):
        self.assertEqual(top([Span("T", "Township 27 North"), Span("R", "Range 22")]), (27, 22, None, "W"))
        self.assertEqual(top([Span("S", "Section 5"), Span("T", "Township 99"), Span("R", "Range 22")]), "not_in_plss")
        self.assertEqual(top([Span("S", "Section 5")]), "no_candidate")
        self.assertEqual(top([]), "no_spans")
        self.assertEqual(top([Span("S", "Section 5"), Span("T", "Township 999"), Span("R", "Range 22")]), "filtered_out")

    def test_cross_slot_recovery(self):
        self.assertEqual(top([Span("S", "Section 5 of Township 27"), Span("R", "Range 22 West")]), (27, 22, 5, "W"))

    def test_multi_section_and_pairing(self):
        r = NORM.normalize([Span("S", "Sections 5 and 6"), Span("T", "T. 28 N."), Span("R", "R. 23 W.")], "toyka", "MN")
        self.assertEqual({t.S for t in r.top_k()}, {5, 6})
        spans = [Span("S", "Section 5"), Span("T", "Township 27"), Span("R", "Range 22"),
                 Span("S", "Section 9"), Span("T", "Township 114"), Span("R", "Range 23")]
        r = NORM.normalize(spans, "toyka", "MN")
        self.assertEqual({(t.T, t.R, t.S) for t in r.top_k()}, {(27, 22, 5), (114, 23, 9)})

    def test_keep_ratio_limits_extra_triples(self):
        spans = [Span("S", "Section 5"), Span("T", "Township 27"), Span("R", "Range 22"), Span("R", "Range 23")]
        loose = RuleBasedNormalizer(C, keep_ratio=0.0).normalize(spans, "toyka", "MN")
        tight = NORM.normalize(spans, "toyka", "MN")
        self.assertGreater(len(loose.ranked), len(tight.ranked))
        self.assertEqual(tight.top_k(), [TRS(27, "N", 22, "W", 5)])

    def test_pluggable_scorer(self):
        class PreferLargeS:
            name = "large_s"

            def score(self, cand):
                return cand.trs.S or 0

        spans = [Span("S", "Sections 5 and 6"), Span("T", "T. 28"), Span("R", "R. 23")]
        r = RuleBasedNormalizer(C, scorer=PreferLargeS()).normalize(spans, "toyka", "MN")
        self.assertEqual(r.top.S, 6)
        self.assertIn("valid_readings", r.info)


class RegexV1BaselineTests(unittest.TestCase):
    def test_wrapper(self):
        r = RegexV1Normalizer().normalize([Span("S", "Section 5"), Span("T", "Township Thirty-one"),
                                           Span("R", "Range 22 West")])
        self.assertEqual(r.top, TRS(31, "", 22, "W", 5))
        r = RegexV1Normalizer().normalize([Span("S", "Section 5"), Span("R", "Range 22")])
        self.assertEqual((r.top.T, r.top.R, r.top.S), (None, 22, 5))  # zip_longest padding kept as-is


class SyntheticRegressionTests(unittest.TestCase):
    """Floors on the toy synthetic corpora (n=300); exact values are reported by dev/eval_synth.py."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "eval_synth", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev", "eval_synth.py"))
        cls.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ev)
        cls.sets = synth.corpora(TOY, 300, seed=5)

    def test_floors_and_ceiling(self):
        regex = RegexV1Normalizer()
        floors = {"clean": 0.99, "ocr": 0.9, "span": 0.75, "ocr+span": 0.7}
        for name, ex in self.sets.items():
            rule = self.ev.score(ex, NORM)
            base = self.ev.score(ex, regex)
            orc = self.ev.oracle(ex)
            self.assertGreaterEqual(rule["exact_top1"], floors[name], name)
            self.assertGreater(rule["exact_top1"], base["exact_top1"], name)
            self.assertLessEqual(rule["exact_top1_ignore_dir"], orc["oracle_TRS"] + 1e-9, name)
            self.assertLess(rule["triples_per_covered"], 1.4, name)


if __name__ == "__main__":
    unittest.main()
