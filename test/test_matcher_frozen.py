"""
Guard for the frozen matcher_v1.0 values (any change needs a new version name).

    python -m unittest discover -s test -p "test_matcher_frozen.py" -v
"""

import unittest

from src.matcher import frozen
from src.matcher.match import MatchConfig


class MatcherFrozenTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(frozen.VERSION, "matcher_v1.0")
        self.assertEqual(frozen.CONFIG, MatchConfig("ratio", "n3", True, False, "none", 0.0, 70.0))
        self.assertEqual(frozen.RULES_MIN_FREQ, 5)
        self.assertEqual(len(frozen.RULES_SHA256), 64)

    def test_rule_hash_check(self):
        with self.assertRaises(RuntimeError):
            frozen.frozen_rules([("Toy Second Addition", "TOY 2ND ADD")] * 5)

    def test_digest_is_order_independent(self):
        a = {"canon": {"X": "Y", "A": "B"}, "merges": {}, "used": {}}
        b = {"used": {}, "merges": {}, "canon": {"A": "B", "X": "Y"}}
        self.assertEqual(frozen.rules_digest(a), frozen.rules_digest(b))


if __name__ == "__main__":
    unittest.main()
