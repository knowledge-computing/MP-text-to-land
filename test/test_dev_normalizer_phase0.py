"""
Tests for the Phase 0 dev scripts on synthetic toy data only (fake PLSS grid, fake counties, fake deeds):
dev/build_plss_table.py, dev/derive_gold_trs.py, dev/tally_overrides.py. Skipped without geopandas.

    mapprejudice/bin/python -m unittest discover -s test -p "test_dev_normalizer_phase0.py" -v
"""

import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, box
except ImportError:  # pragma: no cover
    gpd = None


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "dev", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = fn(*args)
    return code, out.getvalue()


def write(gdf, path):
    gdf.to_file(path, driver="GeoJSON")
    return path


@unittest.skipIf(gpd is None, "geopandas not installed")
class BuildPlssTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        crs = "EPSG:5070"
        self.counties = write(gpd.GeoDataFrame(
            {"STATEFP": ["27", "27", "27", "55"], "NAME": ["Anoka", "Washington", "Hennepin", "Milwaukee"]},
            geometry=[box(0, 0, 20, 10), box(20, 0, 40, 10), box(0, 20, 40, 40), box(200, 200, 220, 220)],
            crs=crs), os.path.join(d, "counties.geojson"))
        self.mn = write(gpd.GeoDataFrame({
            "COUN": [2, 2, 82, 82, 27],
            "TOWN": [31, 31, 31, 30, 29],
            "RANG": [22, 21, 20, 20, 5],
            "RDIR": [1, 1, 1, 1, 2],
            "SECT": [1, 2, 3, 40, 1],
            "TRS_DESCRI": ["T31N R22W S1 Anoka County", "Township 31 North Range 21 West, Anoka",
                           "T31N R20W S3 Washington", "T30N R20W S40 Washington", "T29N R5E S1 Hennepin"],
        }, geometry=[box(0, 0, 10, 10),       # Anoka only
                     box(15, 0, 25, 10),      # straddles Anoka / Washington 50:50
                     box(30, 0, 40, 10),      # Washington only
                     box(39.95, 0, 49.95, 10),  # 0.5% in Washington: sliver, dropped
                     box(5, 25, 10, 30)],     # Hennepin: not a study county
            crs=crs), os.path.join(d, "mn.geojson"))
        self.wi = write(gpd.GeoDataFrame(
            {"TWP": [7, 7], "RNG": [21, 22], "SEC": [5, 6], "DIR_ALPHA": ["E", "E"]},
            geometry=[box(200, 200, 210, 210), box(500, 500, 510, 510)], crs=crs), os.path.join(d, "wi.geojson"))
        self.script = load("build_plss_table")

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *extra):
        d = self.tmp.name
        code, out = quiet(self.script.main, ["--mn", self.mn, "--wi", self.wi, "--counties", self.counties,
                                             "--out-dir", d, "--summary", os.path.join(d, "s.json"), *extra])
        self.assertEqual(code, 0)
        with open(os.path.join(d, "plss_valid_trs.json")) as f:
            return json.loads(out), json.load(f)

    def test_counties_rdir_and_valid_table(self):
        summary, table = self.run_script("--mn-rdir", "infer")
        self.assertEqual(summary["mn_rdir_map"]["map"], {"1": "W", "2": "E"})
        self.assertEqual(summary["mn_rdir_crosstab_statewide"]["1"], {"W": 4})
        mn = summary["states"]["MN"]
        self.assertEqual(mn["rows_in_study_counties"], 3)
        self.assertEqual(mn["sections_in_more_than_one_county"], 1)
        self.assertEqual(mn["multi_county_pairs"], {"anoka;washington": 1})
        self.assertEqual(mn["sliver_overlaps_below_min_frac"], 1)
        self.assertEqual(summary["states"]["WI"]["rows_in_study_counties"], 1)
        valid = table["valid"]
        self.assertEqual(valid["MN"]["anoka"], [[31, "N", 21, "W", 2], [31, "N", 22, "W", 1]])
        self.assertEqual(valid["MN"]["washington"], [[31, "N", 20, "W", 3], [31, "N", 21, "W", 2]])
        self.assertEqual(valid["WI"]["milwaukee"], [[7, "N", 21, "E", 5]])
        self.assertNotIn("hennepin", valid["MN"])
        self.assertEqual(summary["counties"]["MN:anoka"]["townships_T_lt_100"], 2)
        self.assertEqual(summary["counties"]["MN:anoka"]["townships_T_ge_100"], 0)
        self.assertEqual(table["valid_state"]["WI"], [[7, "N", 21, "E", 5], [7, "N", 22, "E", 6]])
        self.assertEqual(len(table["valid_state"]["MN"]), 4)  # S=40 excluded
        self.assertEqual(summary["mn_primary_county_vs_COUN_code"]["washington|82"], 1)
        sections = gpd.read_file(os.path.join(self.tmp.name, "plss_study_sections.gpkg"))
        self.assertEqual(len(sections), 4)
        self.assertEqual(set(sections["Tdir"]), {"N"})

    def test_rdir_override_and_min_frac(self):
        summary, table = self.run_script("--mn-rdir", "1=E", "--min-frac", "0.001")
        self.assertEqual(summary["mn_rdir_map"]["source"], "given (--mn-rdir)")
        self.assertEqual(table["valid"]["MN"]["anoka"][0][3], "E")
        self.assertEqual(summary["states"]["MN"]["sliver_overlaps_below_min_frac"], 0)
        self.assertNotIn([30, "N", 20, "E", 40], table["valid"]["MN"]["washington"])
        self.assertEqual(summary["counties"]["MN:washington"]["S_outside_1_36_values_excluded"], [40])

    def test_default_rdir_map_is_code0_west(self):
        summary, table = self.run_script()
        self.assertEqual(summary["mn_rdir_map"], {"source": "given (--mn-rdir)", "map": {"0": "W"}})
        self.assertEqual(table["valid"]["MN"]["anoka"][0][3], "")  # toy rows use code 1: unmapped


def toy_sections(path):
    """Two MN sections side by side (31N 22W, S1 and S2) and one WI section."""
    gpd.GeoDataFrame({
        "state": ["MN", "MN", "WI"], "counties": ["anoka", "anoka", "milwaukee"],
        "T": [31, 31, 7], "Tdir": ["N", "N", "N"], "R": [22, 22, 21], "Rdir": ["W", "W", "E"], "S": [1, 2, 5],
    }, geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10), box(200, 200, 210, 210)], crs="EPSG:5070").to_file(
        path, layer="sections", driver="GPKG")
    return path


@unittest.skipIf(gpd is None, "geopandas not installed")
class DeriveGoldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.sections = toy_sections(os.path.join(d, "sections.gpkg"))
        mp = lambda *boxes: MultiPolygon(list(boxes))
        self.test_split = write(gpd.GeoDataFrame({
            "image_ids": ["['a1']", "['b1', 'b2']", "['c1']", "['d1']", "['e1']", "['e1']", "['f1']", "['g1']"],
            "cnty_name": ["Anoka County", "Anoka County", "Anoka County", "Anoka County", "Anoka County",
                          "Anoka County", "Milwaukee County", "Hennepin County"],
        }, geometry=[
            mp(box(1, 1, 3, 3)),                    # a: inside S1
            mp(box(7, 1, 17, 2)),                   # b: 30% S1 / 70% S2 -> gold S2, straddles
            mp(box(100, 100, 101, 101)),            # c: no section
            mp(box(1, 1, 2, 2), box(4, 4, 5, 5)),   # d: two parts -> not single
            mp(box(1, 5, 2, 6)), mp(box(11, 5, 12, 6)),  # e: one deed, two rows -> not single
            mp(box(201, 201, 202, 202)),            # f: WI section
            mp(box(1, 1, 2, 2)),                    # g: not a study county
        ], crs="EPSG:5070"), os.path.join(d, "test.geojson"))
        self.train_split = write(gpd.GeoDataFrame({"image_ids": ["['h1']"], "cnty_name": ["Anoka"]},
                                                  geometry=[mp(box(12, 1, 13, 2))], crs="EPSG:5070"),
                                 os.path.join(d, "train.geojson"))
        self.script = load("derive_gold_trs")

    def tearDown(self):
        self.tmp.cleanup()

    def test_gold_rows_and_summary(self):
        d = self.tmp.name
        out_csv = os.path.join(d, "gold.csv")
        code, out = quiet(self.script.main, ["--split", f"test={self.test_split}", "--split", f"train={self.train_split}",
                                             "--sections", self.sections, "--out", out_csv,
                                             "--summary", os.path.join(d, "s.json")])
        self.assertEqual(code, 0)
        summary = json.loads(out)
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))
        test = [r for r in rows if r["split"] == "test"]
        a, b, c, dd, e1, e2, f_, g = test
        self.assertEqual((a["status"], a["S"], a["ratio"], a["single"]), ("ok", "1", "1.0", "True"))
        self.assertEqual((b["S"], b["n_keys_ge_1pct"], b["ratio"], b["second_ratio"]), ("2", "2", "0.7", "0.3"))
        self.assertEqual(b["deed_key"], "b1|b2")
        self.assertEqual(c["status"], "no_match")
        self.assertEqual((dd["n_parts"], dd["single"]), ("2", "False"))
        self.assertEqual((e1["rows_per_deed"], e1["single"], e2["single"]), ("2", "False", "False"))
        self.assertEqual((f_["state"], f_["S"], f_["Rdir"], f_["in_county"]), ("WI", "5", "E", "True"))
        self.assertEqual(g["status"], "no_state")
        s = summary["splits"]["test"]
        self.assertEqual(s["rows"], 8)
        self.assertEqual(s["single_polygon_rows"], 5)
        self.assertEqual(s["status_single_rows"], {"ok": 3, "no_match": 1, "no_state": 1})
        ok = summary["single_ok"]["test"]
        self.assertEqual(ok["rows"], 3)
        self.assertEqual(ok["straddling_rows_gt1_section"], 1)
        self.assertEqual(ok["section_overlap_ratio_hist"], {"[0.5,0.9)": 1, "[0.99,1.0]": 2})
        self.assertEqual(ok["Rdir"], {"W": 2, "E": 1})
        self.assertEqual(summary["splits"]["train"]["single_polygon_rows"], 1)
        deeds = summary["deed_sets"]
        self.assertEqual(deeds["train"]["deeds"], 1)
        self.assertEqual(deeds["train"]["gold_set_size"], {"1": 1})
        self.assertNotIn("test", deeds)  # default deed splits: train,val

    def test_deed_sets_union_rows_and_parts(self):
        d = self.tmp.name
        deed_csv = os.path.join(d, "deeds.csv")
        code, out = quiet(self.script.main, ["--split", f"test={self.test_split}", "--sections", self.sections,
                                             "--out", os.path.join(d, "g.csv"), "--summary", os.path.join(d, "s.json"),
                                             "--deed-splits", "test", "--deed-out", deed_csv])
        self.assertEqual(code, 0)
        with open(deed_csv) as f:
            deeds = {r["deed_key"]: r for r in csv.DictReader(f)}
        # d: two parts both in S1 -> one key; e: two rows in S1 and S2 -> two keys; b: one part -> max overlap S2
        self.assertEqual(json.loads(deeds["d1"]["gold_set"]), [[31, "N", 22, "W", 1]])
        self.assertEqual(json.loads(deeds["e1"]["gold_set"]), [[31, "N", 22, "W", 1], [31, "N", 22, "W", 2]])
        self.assertEqual((deeds["e1"]["n_rows"], deeds["e1"]["n_parts"]), ("2", "2"))
        self.assertEqual(json.loads(deeds["b1|b2"]["gold_set"]), [[31, "N", 22, "W", 2]])
        self.assertEqual(deeds["c1"]["status"], "no_match")
        summary = json.loads(out)["deed_sets"]["test"]
        self.assertEqual(summary["gold_set_size"], {"1": 4, "2": 1})
        self.assertEqual(summary["multi_row_deeds"], 1)
        self.assertNotIn("e1", out)
        # privacy: the printed summary carries no deed keys
        self.assertNotIn("b1", out)
        self.assertNotIn("a1", out)


@unittest.skipIf(gpd is None, "geopandas not installed")
class TallyOverridesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        geoms = [box(i, 0, i + 1, 1) for i in range(4)]
        common = {"image_ids": ["['a']", "['b']", "['c']", "['d']"], "cnty_name": ["Anoka County"] * 4}
        self.batch = write(gpd.GeoDataFrame({**common,
            "TRIPLE_p": [str([(1, 31, 22)]), str([(9, 31, 22)]), str([]), str([(1, 31, 22)])],
            "RANGE_p": [str(["Range 22 West"])] * 4}, geometry=geoms, crs="EPSG:5070"), os.path.join(d, "b.geojson"))
        self.saved = write(gpd.GeoDataFrame({**common,
            "TRIPLE_p": [str([(1, 31, 22, "W")]), str([(2, 31, 22, "W")]), str([(3, 31, 22, "W")]),
                         str([(1, 31, 22, "W"), (2, 31, 22, "W")])]}, geometry=geoms, crs="EPSG:5070"),
            os.path.join(d, "s.geojson"))
        self.gold = os.path.join(d, "gold.csv")
        with open(self.gold, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "single", "status", "county", "deed_key", "S", "T", "R", "Rdir"])
            for key, s in (("a", 1), ("b", 2), ("c", 3), ("d", 2)):
                w.writerow(["test", "True", "ok", "anoka", key, s, 31, 22, "W"])
        self.script = load("tally_overrides")

    def tearDown(self):
        self.tmp.cleanup()

    def test_tally(self):
        code, out = quiet(self.script.main, ["--batch", self.batch, "--saved", self.saved, "--gold", self.gold,
                                             "--summary", os.path.join(self.tmp.name, "t.json")])
        self.assertEqual(code, 0)
        r = json.loads(out)
        self.assertEqual((r["matched"], r["changed_deeds"]), (4, 3))
        self.assertEqual(r["change_types"], {"S_changed": 1, "empty_to_nonempty": 1, "triple_count_changed": 2})
        g = r["gold_changed_deeds"]
        self.assertEqual(g["with_gold"], 3)
        self.assertEqual(g["before"].get("top1_S", 0), 0)
        self.assertEqual(g["after"]["top1_S"], 2)  # b and c fixed; d's first triple still S1
        self.assertEqual(g["after"]["any_S_T_R"], 3)
        self.assertNotIn("'a'", out)


if __name__ == "__main__":
    unittest.main()
