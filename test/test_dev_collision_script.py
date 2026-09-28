"""
Tests for dev/check_county_id_collisions.py on synthetic toy data only.

The script must run with the Python standard library alone (no python-dotenv, geopandas,
spaCy, ...) and must print aggregate counts only. Run from the repository root:

    python -m unittest discover -s test -p "test_dev_collision_script.py" -v
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "dev", "check_county_id_collisions.py")

spec = importlib.util.spec_from_file_location("check_county_id_collisions", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

# Synthetic tree covering each collision case (see test_county_collisions.py).
TOY_PAGES = [
    "mn-toy-county/toy-book-c/mnonly_001", "wi-toy-county/toy-book-d/wionly_001",    # control
    "mn-toy-county/toy-book-x/sharedpage", "wi-toy-county/toy-book-x/sharedpage",    # same id, no "_"
    "mn-toy-county/toy-book-y/page_001", "wi-toy-county/toy-book-y/page_001",        # same id, with "_"
    "mn-toy-county/toy-book-m/scan_001", "wi-toy-county/toy-book-w/scan_002",        # same prefix
    "mn-toy-county/toy-folder-1/multi_001", "mn-toy-county/toy-folder-2/multi_002",  # one county, 2 folders
]
EXPECTED = {"files": 10, "county_folders": 2, "files_under_root": 0, "groups": 7,
            "mixed_county_groups": 2, "multi_folder_groups": 1,
            "colliding_ids": 2, "colliding_ids_no_underscore": 1}

# Imports the script must not need.
BLOCKED = ["dotenv", "geopandas", "pandas", "numpy", "shapely", "pyproj", "pyogrio",
           "spacy", "rapidfuzz", "tqdm", "torch", "src"]
RUNNER = """
import importlib.abc, runpy, sys
blocked = set(sys.argv[1].split(","))
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in blocked:
            raise ModuleNotFoundError(f"blocked for test: {name}")
sys.meta_path.insert(0, Block())
script, sys.argv = sys.argv[2], [sys.argv[2]] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
"""


class CollisionScriptTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.root = os.path.join(tmp.name, "ocr", "txt")
        for rel in TOY_PAGES:
            path = os.path.join(self.root, rel + ".txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("toy text")

    def run_script_stdlib_only(self, *args):
        return subprocess.run([sys.executable, "-c", RUNNER, ",".join(BLOCKED), SCRIPT, *args],
                              capture_output=True, text=True, cwd=self.tmp, env={**os.environ, "OCRTXT_PATH": ""})

    def test_counts_on_toy_tree(self):
        self.assertEqual(script.count_collisions(self.root), EXPECTED)

    def test_runs_without_third_party_packages(self):
        proc = self.run_script_stdlib_only("--root", self.root)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OCR txt files: 10 | county folders: 2", proc.stdout)
        self.assertIn("groups mixing >1 county (Tool 3 merges across counties): 2", proc.stdout)
        self.assertIn("county-relative ids present in >1 county: 2", proc.stdout)

    def test_output_contains_counts_only(self):
        out = self.run_script_stdlib_only("--root", self.root).stdout
        self.assertNotIn(self.root, out)
        for rel in TOY_PAGES:
            for part in rel.split("/"):
                self.assertNotIn(part, out)

    def test_missing_root_fails_without_printing_a_path(self):
        missing = os.path.join(self.tmp, "does-not-exist")
        proc = self.run_script_stdlib_only("--root", missing)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn(missing, proc.stdout + proc.stderr)

    def test_root_resolution_order(self):
        env_file = os.path.join(self.tmp, ".env")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write('# comment\nGEOJSON_PATH="/toy/geojson/"\nexport OCRTXT_PATH="/toy/from-dotenv/"\n')
        self.assertEqual(script.resolve_root("/toy/cli/", {"OCRTXT_PATH": "/toy/env/"}, env_file), "/toy/cli/")
        self.assertEqual(script.resolve_root(None, {"OCRTXT_PATH": "/toy/env/"}, env_file), "/toy/env/")
        self.assertEqual(script.resolve_root(None, {}, env_file), "/toy/from-dotenv/")
        self.assertIsNone(script.resolve_root(None, {}, os.path.join(self.tmp, "no.env")))

    def test_read_env_file_formats(self):
        env_file = os.path.join(self.tmp, ".env")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("\n# full-line comment\nA=plain\nB='single quoted # not a comment'\n"
                    'C="double quoted"\nD=value # trailing comment\nexport E=exported\nnot a pair\n')
        self.assertEqual(script.read_env_file(env_file),
                         {"A": "plain", "B": "single quoted # not a comment", "C": "double quoted",
                          "D": "value", "E": "exported"})


if __name__ == "__main__":
    unittest.main()
