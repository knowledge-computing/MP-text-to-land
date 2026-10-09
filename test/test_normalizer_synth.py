"""
Tests for the synthetic T/R/S generator and shared normalizer types (synthetic data only).

    mapprejudice/bin/python -m unittest discover -s test -p "test_normalizer_synth.py" -v
"""

import io
import json
import os
import random
import re
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout

from src.normalizer import TRS, Result, Span, spans_from_fields
from src.normalizer import synth
from src.normalizer.numbers import int_to_words
from src.normalizer.ocr import TO_DIGIT, ocr_noise

TOY = synth.toy_valid()


def squash(text):
    return re.sub(r"[\s\-]|\band\b", "", text.lower())


class NumberWordTests(unittest.TestCase):
    def test_int_to_words(self):
        self.assertEqual(int_to_words(7), "seven")
        self.assertEqual(int_to_words(40), "forty")
        self.assertEqual(int_to_words(24), "twenty-four")
        self.assertEqual(int_to_words(24, joiner=" "), "twenty four")
        self.assertEqual(int_to_words(24, joiner=""), "twentyfour")
        self.assertEqual(int_to_words(117), "one hundred seventeen")
        self.assertEqual(int_to_words(117, hundred_and=True), "one hundred and seventeen")
        self.assertEqual(int_to_words(100, hundred_and=True), "one hundred")
        with self.assertRaises(ValueError):
            int_to_words(0)


class OcrTests(unittest.TestCase):
    def test_to_digit_maps_only_letters(self):
        self.assertTrue(all(not k.isdigit() for k in TO_DIGIT))
        self.assertEqual(TO_DIGIT["l"], "1")
        self.assertEqual(TO_DIGIT["O"], "0")
        self.assertEqual(TO_DIGIT["S"], "5")

    def test_noise_is_seeded_and_zero_p_is_identity(self):
        text = "Section Twenty-four (24), Township 31 North"
        self.assertEqual(ocr_noise(text, random.Random(1), 0.0), text)
        a = ocr_noise(text, random.Random(5), 0.2)
        b = ocr_noise(text, random.Random(5), 0.2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, text)


class SynthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets = synth.corpora(TOY, 400, seed=11)

    def test_deterministic_and_seed_dependent(self):
        a = [e.to_dict() for e in synth.generate(TOY, 50, seed=3, ocr_p=0.05, span_p=0.2)]
        b = [e.to_dict() for e in synth.generate(TOY, 50, seed=3, ocr_p=0.05, span_p=0.2)]
        c = [e.to_dict() for e in synth.generate(TOY, 50, seed=4, ocr_p=0.05, span_p=0.2)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_offsets_match_text_in_every_set(self):
        for name, examples in self.sets.items():
            for e in examples:
                for s in e.spans:
                    self.assertEqual(e.text[s.start:s.end], s.text, f"{name} #{e.id}")
                    self.assertIn(s.label, ("S", "T", "R"))

    def test_noisy_sets_pair_with_clean(self):
        clean = self.sets["clean"]
        for name in ("ocr", "span", "ocr+span"):
            for c, n in zip(clean, self.sets[name]):
                self.assertEqual((c.layout, c.gold, c.county), (n.layout, n.gold, n.county))
        self.assertTrue(all(not e.noise for e in clean))
        self.assertGreater(sum(bool(e.noise) for e in self.sets["ocr"]), 100)
        self.assertGreater(sum(any(":" in x for x in e.noise) for e in self.sets["span"]), 50)

    def test_clean_spans_state_their_numbers(self):
        checked = 0
        for e in self.sets["clean"]:
            if e.layout not in ("STR", "TRS", "TR", "compact"):
                continue
            g = e.gold[0]
            for s in e.spans:
                n = {"S": g.S, "T": g.T, "R": g.R}[s.label]
                style = e.styles[s.label]
                if style in ("digits", "words_digits", "compact"):
                    self.assertIn(str(n), s.text)
                elif style == "words":
                    self.assertIn(squash(int_to_words(n)), squash(s.text))
                checked += 1
        self.assertGreater(checked, 500)

    def test_gold_is_valid_and_directions_follow_state(self):
        valid = {k: set(v) for k, v in TOY.items()}
        for e in self.sets["clean"]:
            for g in e.gold:
                self.assertEqual(g.Rdir, "E" if e.state == "WI" else "W")
                self.assertEqual(g.Tdir, "N")
                if g.S is None:
                    self.assertEqual(e.layout, "TR")
                else:
                    self.assertIn(g, valid[(e.state, e.county)])
        multi = [e for e in self.sets["clean"] if e.layout in ("multi_section", "multi_triple")]
        self.assertTrue(multi and all(len(e.gold) == 2 for e in multi))

    def test_all_layouts_and_styles_appear(self):
        layouts = Counter(e.layout for e in self.sets["clean"])
        self.assertEqual(set(layouts), {name for name, _ in synth.LAYOUTS})
        styles = Counter(v for e in self.sets["clean"] for v in e.styles.values())
        self.assertTrue({"digits", "words", "words_digits", "numbered", "compact"} <= set(styles))

    def test_meridian_suffix_matches_state(self):
        for e in self.sets["clean"]:
            if "Principal Meridian" in e.text:
                self.assertIn("Fourth" if e.state == "WI" else "Fifth", e.text)

    def test_valid_json_roundtrip_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "valid.json")
            data = {"meta": {}, "valid": {st: {c: [[t.T, t.Tdir, t.R, t.Rdir, t.S] for t in rows]}
                                          for (st, c), rows in TOY.items()}}
            with open(path, "w") as f:
                json.dump(data, f)
            self.assertEqual(synth.load_valid_json(path), TOY)
            buf = io.StringIO()
            with redirect_stdout(buf):
                synth.main(["--valid-json", path, "--n", "4", "--seed", "2"])
            lines = buf.getvalue().strip().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertIn("gold", json.loads(lines[0]))


class TypeTests(unittest.TestCase):
    def test_result_and_trs(self):
        t = TRS(31, "N", 22, "W", 5)
        self.assertEqual(t.paper_triple(), (5, 31, 22, "W"))
        self.assertEqual(t.township(), TRS(31, "N", 22, "W", None))
        r = Result([(t, 0.9), (TRS(31, "N", 22, "W", 6), 0.5)])
        self.assertEqual(r.top, t)
        self.assertEqual(len(r.top_k(1)), 1)
        self.assertIsNone(Result(reason="no_candidate").top)

    def test_spans_from_fields(self):
        props = {"SECTION_p": ["Section 5", ""], "TOWNSHIP_p": "['Township 31 North']", "RANGE_p": None}
        self.assertEqual(spans_from_fields(props), [Span("S", "Section 5"), Span("T", "Township 31 North")])


if __name__ == "__main__":
    unittest.main()
