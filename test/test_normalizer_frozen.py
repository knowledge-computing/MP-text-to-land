"""
Guard for the frozen rule_v2 constants (src/normalizer/frozen.py). Fails on any undeclared change.

    mapprejudice/bin/python -m unittest discover -s test -p "test_normalizer_frozen.py" -v
"""

import os
import unittest

from src.normalizer import frozen

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FrozenConfigTests(unittest.TestCase):
    def test_live_constants_match_frozen_record(self):
        self.assertEqual(frozen.current(), frozen.FROZEN)

    def test_plss_table_defaults(self):
        with open(os.path.join(REPO_ROOT, "dev", "build_plss_table.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn(f'"--min-frac", type=float, default={frozen.FROZEN["plss_table"]["min_frac"]}', src)
        self.assertIn(f'"--mn-rdir", default="{frozen.FROZEN["plss_table"]["mn_rdir"]}"', src)

if __name__ == "__main__":
    unittest.main()
