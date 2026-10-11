"""
Tests for dev/private_paths.py with synthetic environment files and folders only.

    python -m unittest discover -s test -p "test_private_paths.py" -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dev"))
import private_paths as pp  # noqa: E402

VARS = ("MP_DATA_ROOT", "MP_RUN_DIR", "MP_OCR_ROOT", "MP_PLAT_CONFIG")


class PrivatePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "toy_root"
        (self.root / "toy_run").mkdir(parents=True)
        (self.root / "toy_plats").mkdir()
        self.env_file = Path(self.tmp.name) / "toy.env"
        self.clean = mock.patch.dict(os.environ, {k: "" for k in VARS})
        self.clean.start()

    def tearDown(self):
        self.clean.stop()
        self.tmp.cleanup()

    def test_missing_variable_is_a_clear_error(self):
        self.env_file.write_text("# nothing\n")
        with self.assertRaises(pp.PrivatePathError) as cm:
            pp.get("MP_DATA_ROOT", self.env_file)
        self.assertIn("MP_DATA_ROOT is not set", str(cm.exception))

    def test_env_file_and_relative_resolution(self):
        self.env_file.write_text(f'export MP_DATA_ROOT="{self.root}"\nMP_RUN_DIR=toy_run\n')
        self.assertEqual(pp.path("MP_RUN_DIR", self.env_file), self.root / "toy_run")
        with mock.patch.dict(os.environ, {"MP_RUN_DIR": str(self.root / "missing")}):
            with self.assertRaises(pp.PrivatePathError) as cm:
                pp.path("MP_RUN_DIR", self.env_file)
            self.assertIn("does not exist", str(cm.exception))

    def test_environment_wins_over_env_file(self):
        self.env_file.write_text("MP_DATA_ROOT=/nowhere\n")
        with mock.patch.dict(os.environ, {"MP_DATA_ROOT": str(self.root)}):
            self.assertEqual(pp.path("MP_DATA_ROOT", self.env_file), self.root)

    def test_lazy_path_resolves_only_when_used(self):
        lazy = pp.run_dir("output", "x.geojson")            # no error at creation
        self.assertIn("MP_RUN_DIR", repr(lazy))
        with mock.patch.object(pp, "ENV_FILE", self.env_file):
            self.env_file.write_text("")
            with self.assertRaises(pp.PrivatePathError):
                os.fspath(lazy)
        with mock.patch.dict(os.environ, {"MP_DATA_ROOT": str(self.root), "MP_RUN_DIR": "toy_run"}):
            self.assertEqual(Path(lazy), self.root / "toy_run" / "output" / "x.geojson")
            self.assertEqual(Path(lazy / "y"), self.root / "toy_run" / "output" / "x.geojson" / "y")

    def test_plat_config(self):
        cfg = {"root": "toy_plats", "layers": {"Anoka": {"path": "a.shp", "name_field": "N", "alias_fields": []}},
               "rule_pairs": {}}
        cfg_path = Path(self.tmp.name) / "plat.json"
        cfg_path.write_text(json.dumps(cfg))
        self.env_file.write_text(f"MP_DATA_ROOT={self.root}\nMP_PLAT_CONFIG={cfg_path}\n")
        out = pp.plat_config(self.env_file)
        self.assertEqual(out["root"], self.root / "toy_plats")
        cfg["layers"]["Anoka"].pop("name_field")
        cfg_path.write_text(json.dumps(cfg))
        with self.assertRaises(pp.PrivatePathError):
            pp.plat_config(self.env_file)

    def test_example_config_has_the_expected_shape(self):
        ex = json.loads((Path(REPO_ROOT) / "plat_config.example.json").read_text())
        self.assertEqual(set(ex["layers"]), {"anoka", "dakota", "milwaukee", "olmsted", "sherburne"})
        self.assertTrue(all({"path", "name_field", "alias_fields"} <= set(v) for v in ex["layers"].values()))


if __name__ == "__main__":
    unittest.main()
