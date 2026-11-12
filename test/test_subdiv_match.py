"""
Synthetic tests for src/subdiv_match (normalizer port + extensions, block/lot parsing, gazetteer,
leave-one-out, matchers, reranker). Needs the thesis-hf venv (scikit-learn, rapidfuzz):

    ~/MapPrejudice/venvs/thesis-hf/bin/python -m unittest test.test_subdiv_match -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from subdiv_match.blocklot import parse_block, parse_lots  # noqa: E402
from subdiv_match.build import build_gazetteer  # noqa: E402
from subdiv_match.data import deed_key, norm_doc, parse_ids  # noqa: E402
from subdiv_match.deeds import leak_free_train  # noqa: E402
from subdiv_match.gazetteer import LAYER  # noqa: E402
from subdiv_match.matchers import FEATURE_SETS, MatchIndex, collect, fit_reranker, rerank  # noqa: E402
from subdiv_match.normalize import plat_numbers, standardize, standardize_addition_mp  # noqa: E402
from subdiv_match.span_extract import TOKEN, merge_entities, token_chunks  # noqa: E402
from subdiv_match import runguard  # noqa: E402
from subdiv_match.stats import discordant, holm, mcnemar_exact  # noqa: E402


class NormalizeTest(unittest.TestCase):
    def test_port_matches_processor_behaviour(self):
        self.assertEqual(standardize_addition_mp("Jane's Second Addition to the City of St. Paul"), "janes 2nd")
        self.assertEqual(standardize_addition_mp("JANES 2ND ADDITION"), "janes 2nd")
        self.assertEqual(standardize_addition_mp("Oak Ridge Addition No. 2"), "oak ridge 2")
        self.assertEqual(standardize_addition_mp("Toy Park Subd."), "toy park")
        self.assertEqual(standardize_addition_mp(None), "")

    def test_extensions(self):
        cases = {
            "Twenty-First Addition to Toyville": "21st to toyville",
            "Toy Park Number Two": "toy park 2",
            "Registered Land Survey No. Twelve": "rls 12",
            "R.L.S. No. 12": "rls 12",
            "aud 6": "auditors 6",
            "AUD SUB NO 6": "auditors 6",
            "Re-Arrangement of Block 3, Elm Park": "rearrangement of block 3 elm park",
            "Re-Plat of Toy Park": "replat of toy park",
            "Toy Park Re-Sub": "toy park resubdivision",
            "C0LUMBIA PARK": "columbia park",
            "Toy Lake Shores Addi- tion": "toy lake shores",
            "Toy Park Additon": "toy park",
            "Toy Hghts 2nd Add": "toy heights 2nd",
            "Toyland, according to the recorded plat thereof": "toyland",
            "Toy Park, Anoka County, Minnesota": "toy park",
            "Washington County Toy Plat": "washington county toy plat",
            "The Toys": "toys",
            "Toy 21st Addition": "toy 21st",
        }
        for raw, want in cases.items():
            self.assertEqual(standardize(raw), want, raw)

    def test_plat_numbers_are_kept_apart(self):
        self.assertNotEqual(standardize("Oak Ridge #1"), standardize("Oak Ridge #2"))
        self.assertEqual(plat_numbers(standardize("Oak Ridge Second Addition")), {2})


class BlockLotTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_block("Block 03"), "3")
        self.assertEqual(parse_block("a"), "A")
        self.assertEqual(parse_lots("Lots 3 thru 5 and 9"), {"3", "4", "5", "9"})
        self.assertEqual(parse_lots("Lot 4A"), {"4A"})
        self.assertEqual(parse_lots(None), set())


class DataTest(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(parse_ids("('a/1', 'a/2')"), ["a/1", "a/2"])
        self.assertEqual(parse_ids("a/1,a/2"), ["a/1", "a/2"])
        self.assertEqual(deed_key({"cnty_name": "Toy County", "image_ids": "a,b"}), ("toy", frozenset({"a", "b"})))
        self.assertEqual(norm_doc(" 00123.0 "), "123")

    def test_leak_free_train(self):
        tr = {("x", frozenset({"1"})): {"county": "x", "doc_nums": ["10"]},
              ("x", frozenset({"2"})): {"county": "x", "doc_nums": ["20"]},
              ("x", frozenset({"3"})): {"county": "x", "doc_nums": ["30"]}}
        va = {("x", frozenset({"1"})): {"county": "x", "doc_nums": ["99"]},
              ("x", frozenset({"9"})): {"county": "x", "doc_nums": ["20"]}}
        kept, dropped = leak_free_train(tr, [va])
        self.assertEqual(dropped, 2)
        self.assertEqual(list(kept), [("x", frozenset({"3"}))])


def toy_records():
    name = lambda c, n, src="mngeo", **kw: {"type": "name", "county": c, "name": n, "source": src,  # noqa: E731
                                           "contributor": LAYER, "aliases": kw.get("aliases", []),
                                           "n_parcels": kw.get("n", 10), "blocks": kw.get("blocks", []),
                                           "lots": kw.get("lots", []), "pairs": kw.get("pairs", [])}
    cov = lambda c, a, m, doc, src="public": {"type": "cov", "county": c, "add_cov": a, "add_mod": m,  # noqa: E731
                                              "source": src, "contributor": (c, doc), "block": "1", "lots": ["2"]}
    return [
        name("toy", "Oak Ridge Addition", blocks=["1"], lots=["1", "2"], pairs=[("1", "1"), ("1", "2")]),
        name("toy", "Oak Ridge 2nd Addition", blocks=["1"], lots=["5"], pairs=[("1", "5")]),
        name("toy", "Elm Park", src="partner", aliases=["ELM PK"]),
        name("toy", "Birch Hills"),
        name("other", "Oak Ridge Addition"),
        cov("toy", "Birch Hls", "Birch Hills", "d1"),
        cov("toy", "Maple Grove Acres", "Maple Grove", "d2", src="train"),
    ]


class GazetteerMatcherTest(unittest.TestCase):
    def setUp(self):
        self.gaz = build_gazetteer(toy_records(), standardize)
        self.ix = MatchIndex(self.gaz)
        self.key = lambda e: (self.gaz.entries[e].county, self.gaz.entries[e].key)  # noqa: E731

    def test_entries_and_surfaces(self):
        keys = {(e.county, e.key) for e in self.gaz.entries}
        self.assertIn(("toy", "oak ridge"), keys)
        self.assertIn(("toy", "oak ridge 2nd"), keys)
        self.assertIn(("toy", "maple grove"), keys)
        self.assertEqual(len(self.gaz.gold_entries("toy", ["ELM PK"])), 1)        # alias field resolves
        self.assertEqual(self.gaz.gold_entries("toy", ["Birch Hls"]), set())     # cov surfaces never gold

    def test_subset_by_source(self):
        g = build_gazetteer(toy_records(), standardize, sources={"partner"})
        self.assertEqual({e.key for e in g.entries}, {"elm park"})

    def test_leave_one_out(self):
        d = {"county": "toy", "queries": ["Maple Grove Acres"], "exclude": set()}
        self.assertEqual([self.key(e) for e in self.ix.exact(d)], [("toy", "maple grove")])
        d["exclude"] = {("toy", "d2")}
        self.assertEqual(self.ix.exact(d), [])
        self.assertEqual(self.gaz.gold_entries("toy", ["Maple Grove"], {("toy", "d2")}), set())

    def test_matchers_rank_the_right_plat(self):
        d = {"county": "toy", "queries": ["Oak Ridge Second Addition"], "exclude": set()}
        for m in (self.ix.exact, self.ix.token_set, self.ix.tfidf):
            self.assertEqual(self.key(m(d)[0]), ("toy", "oak ridge 2nd"), m.__name__)
        d = {"county": "toy", "queries": ["Birch Hls"], "exclude": set()}
        self.assertEqual(self.key(self.ix.exact(d)[0]), ("toy", "birch hills"))
        self.assertEqual(self.ix.exact({"county": "toy", "queries": [], "exclude": set()}), [])

    def test_own_county_only_for_a_to_c(self):
        d = {"county": "other", "queries": ["Oak Ridge 2nd"], "exclude": set()}
        self.assertTrue(all(self.gaz.entries[e].county == "other" for e in self.ix.tfidf(d)))

    def test_features_and_reranker(self):
        d = {"county": "toy", "queries": ["Oak Ridge"], "blocks": {"1"}, "lots": {"5"}, "exclude": set()}
        cands, tf = self.ix.candidates(d, 5)
        rows = {self.key(e): r for e, r in zip(cands, self.ix.features(d, cands, tf))}
        self.assertEqual(rows[("toy", "oak ridge 2nd")]["pair_in"], 1.0)
        self.assertEqual(rows[("toy", "oak ridge")]["pair_in"], 0.0)
        self.assertEqual(rows[("toy", "oak ridge 2nd")]["num_entry_only"], 1.0)
        self.assertEqual(rows[("other", "oak ridge")]["county_match"], 0.0)
        deeds = [d, {"county": "toy", "queries": ["Oak Ridge"], "blocks": {"1"}, "lots": {"1"}, "exclude": set()},
                 {"county": "toy", "queries": ["Elm Pk"], "blocks": set(), "lots": set(), "exclude": set()}]
        gold = [self.gaz.gold_entries("toy", g) for g in (["Oak Ridge 2nd"], ["Oak Ridge"], ["Elm Park"])]
        col = collect(self.ix, deeds, 5)
        cols = FEATURE_SETS["rerank"]
        m = fit_reranker(col, gold, cols)
        top = rerank(m, cols, *col[0], k=1)
        self.assertEqual(self.key(top[0]), ("toy", "oak ridge 2nd"))


class Step7HelpersTest(unittest.TestCase):
    def test_mclio_extract(self):
        from subdiv_match.mclio import extract
        self.assertEqual(extract("TOY HEIGHTS NORTH ADD 1 NE 1/4 SEC 3 5 22 LOT 9 BLK 4 64 X 141"),
                         ("TOY HEIGHTS NORTH ADD 1", "4", {"9"}))
        self.assertEqual(extract("TOY SUBD OF LTS 47-55 INCL IN SW 1/4 16-7-22 V2P125 BLOCK 51 LOT 9")[0],
                         "TOY SUBD OF LTS 47-55 INCL")
        self.assertIsNone(extract("COM 100 FT N OF SE COR SEC 3 TH W 50 FT")[0])
        self.assertIsNone(extract("TOYS CONDOMINIUM NW 1 8 21 BLDG 5 UNIT 22")[0])
        self.assertIsNone(extract("TOY PARK LOT 2", condo_name="TOY CONDO")[0])
        self.assertIsNone(extract(None)[0])

    def test_metes_bounds(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))
        from subdiv_11_metes_bounds import indicators
        f = indicators("Beginning at the SW corner of Section 12; thence N 89° 30' E 330 feet to the point of beginning")
        self.assertTrue(f["bearing_distance"] and f["metes_bounds"] and f["section_corner"])
        self.assertFalse(indicators("Lot 7, Block 3, Toy Heights Addition")["metes_bounds"])

    def test_llm_windows(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))
        from subdiv_09_llm_extract import windows
        w = windows("x " * 500 + "Lot 7, Block 3, Toy Heights Addition" + " y" * 500, 40, 3000)
        self.assertIn("Toy Heights Addition", w)
        self.assertLess(len(w), 300)
        self.assertEqual(len(windows("no keyword " * 500, 40, 100)), 100)

    def test_error_categories(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))
        from subdiv_10_eval import categorize
        g = build_gazetteer(toy_records(), standardize)
        e = lambda k: g._eid[("toy", k)]  # noqa: E731
        d = lambda q, gold: {"queries": q, "gold": gold}  # noqa: E731
        text = "lot 1 block 1 oak ridge addition and more words"
        gold = {e("oak ridge")}
        self.assertEqual(categorize(d(["Oak Ridge"], ["Oak Ridge"]), e("oak ridge"), gold, g, text, standardize), "correct")
        self.assertEqual(categorize(d([], ["Oak Ridge"]), None, gold, g, text, standardize), "no_query")
        self.assertEqual(categorize(d(["X"], ["Nowhere"]), None, set(), g, text, standardize), "gold_not_in_gazetteer")
        self.assertEqual(categorize(d(["Oak Ridge 2nd"], ["Oak Ridge"]), e("oak ridge 2nd"), gold, g, text, standardize),
                         "wrong_number")
        self.assertEqual(categorize(d(["Birch Hills"], ["Oak Ridge"]), e("birch hills"), gold, g, text, standardize),
                         "wrong_span")
        self.assertEqual(categorize(d(["Oak Ridgc"], ["Oak Ridge"]), e("birch hills"), gold, g,
                                    "lot 1 oak ridgc addition", standardize), "ocr_garbled")
        self.assertEqual(categorize(d(["Elm Park"], ["Oak Ridge"]), e("elm park"), gold, g, "nothing relevant", standardize),
                         "other_gold_not_in_text")


class CompareV10RuleTest(unittest.TestCase):
    """dev/subdiv_13_compare_v10.py decision rule on toy outcomes (no data)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))
        import subdiv_13_compare_v10 as cmp
        cls.cmp = cmp

    def rows(self, all_f, all_v, un_f, un_v, counties):
        r = [self.cmp.scope_row("all", all_f, all_v), self.cmp.scope_row("val_unseen_subd120", un_f, un_v)]
        return r + [self.cmp.scope_row(f"county:{c}", f, v) for c, (f, v) in counties.items()]

    def result(self, crit):
        return {c["criterion"]: c["result"] for c in crit}

    def test_replace_when_all_criteria_hold(self):
        f, v = [1] * 90 + [0] * 10, [1] * 70 + [0] * 30          # +20 pts, 20 discordant all for final
        crit, replace = self.cmp.decide(self.rows(f, v, [1, 0], [1, 0], {"x": (f[:40], v[:40]), "y": ([1] * 10, [0] * 10)}))
        self.assertTrue(replace)
        self.assertEqual(self.result(crit), {"a": "PASS", "b": "PASS", "c": "PASS",
                                             "decision": "REPLACE with subdiv_match_final_v1"})

    def test_each_criterion_can_fail(self):
        f, v = [1] * 90 + [0] * 10, [1] * 70 + [0] * 30
        small = [1] * 52 + [0] * 48, [1] * 50 + [0] * 50        # +2 pts: (a) fails
        crit, replace = self.cmp.decide(self.rows(*small, [1], [1], {}))
        self.assertFalse(replace)
        self.assertEqual(self.result(crit)["a"], "FAIL")
        crit, _ = self.cmp.decide(self.rows(f, v, [0] * 10, [1] * 10, {}))   # unseen -100 pts: (b) fails
        self.assertEqual(self.result(crit)["b"], "FAIL")
        county = ([0] * 31 + [1] * 9, [1] * 40)                  # -77.5 pts in a county with 40 eligible
        crit, _ = self.cmp.decide(self.rows(f, v, [1], [1], {"x": county}))
        self.assertEqual(self.result(crit)["c"], "FAIL")
        crit, _ = self.cmp.decide(self.rows(f, v, [1], [1], {"x": (county[0][:29], county[1][:29])}))  # < 30: ignored
        self.assertEqual(self.result(crit)["c"], "PASS")

    def test_boundaries_are_inclusive(self):
        f, v = [1] * 25 + [0] * 75, [1] * 20 + [0] * 80          # exactly +5.0 pts, 5 / 0 discordant: p = 0.0625
        crit, _ = self.cmp.decide(self.rows(f, v, [1], [1], {}))
        self.assertEqual(self.result(crit)["a"], "FAIL")          # gain ok, p not < 0.05
        f, v = [1] * 30 + [0] * 170, [1] * 20 + [0] * 180        # exactly +5.0 pts, 10 / 0: p = 0.002
        un = [1] * 15 + [0] * 5, [1] * 16 + [0] * 4               # exactly -5.0 pts
        crit, _ = self.cmp.decide(self.rows(f, v, *un, {"x": ([1] * 27 + [0] * 3, [1] * 30)}))   # exactly -10.0 pts
        self.assertEqual(self.result(crit), {"a": "PASS", "b": "PASS", "c": "PASS",
                                             "decision": "REPLACE with subdiv_match_final_v1"})


class SpanExtractTest(unittest.TestCase):
    """Chunking / merging for GLiNER (dev/subdiv_15_gliner_extract.py), on synthetic text."""

    @staticmethod
    def fake_model(text, chunks, name):
        """Entities a perfect model would return per chunk: every whole occurrence of name inside the chunk, plus a
        truncated copy when the chunk cuts it."""
        out = []
        for a, b, _, _ in chunks:
            ents, i = [], text.find(name)
            while i >= 0:
                j = i + len(name)
                if a <= i < b:
                    ents.append({"start": i - a, "end": min(j, b) - a})
                elif i < a < j:
                    ents.append({"start": 0, "end": j - a})
                i = text.find(name, i + 1)
            out.append(ents)
        return out

    def test_chunks_cover_text_and_respect_limit(self):
        text = " ".join(f"w{i}" for i in range(1000))
        ch = token_chunks(text, 100, 20)
        self.assertEqual(ch[0][0], 0)
        self.assertEqual(ch[-1][1], len(text))
        for a, b, oa, ob in ch:
            self.assertLessEqual(len(TOKEN.findall(text[a:b])), 100)
            self.assertTrue(a <= oa < ob <= len(text))
        owned = [(oa, ob) for _, _, oa, ob in ch]
        self.assertEqual(owned[0][0], 0)
        self.assertEqual(owned[-1][1], len(text))
        for (_, e1), (s2, _) in zip(owned, owned[1:]):
            self.assertEqual(e1, s2)                    # owned ranges tile the text

    def test_short_and_empty(self):
        self.assertEqual(token_chunks("", 10, 2), [])
        self.assertEqual(len(token_chunks("Lot 1, Block 2, Toy Park", 384, 64)), 1)
        with self.assertRaises(ValueError):
            token_chunks("a b", 10, 10)

    def test_merge_keeps_whole_name_once_across_chunk_edges(self):
        name = "Meadowbrook Gardens Second Addition"
        filler = " ".join(f"x{i}" for i in range(57))
        for shift in range(0, 12):                      # slide the name across a chunk edge
            text = " ".join(["y"] * shift) + " " + filler + f" Lot 1, {name}, and " + filler
            ch = token_chunks(text, 64, 16)
            self.assertGreater(len(ch), 1)
            self.assertEqual(merge_entities(text, ch, self.fake_model(text, ch, name)), [name], shift)

    def test_merge_order_dedupe_and_cleanup(self):
        text = "Lot 1, Toy Park; and Lot 2, Elm Grove Replat, and Lot 3, TOY  PARK,"
        ch = token_chunks(text, 384, 64)
        ents = [[{"start": text.find("Elm"), "end": text.find(", and Lot 3")},
                 {"start": text.find("Toy"), "end": text.find("; and") + 1},
                 {"start": text.find("TOY"), "end": len(text)}]]
        self.assertEqual(merge_entities(text, ch, ents), ["Toy Park", "Elm Grove Replat"])


class StatsTest(unittest.TestCase):
    """McNemar / Holm of docs/preregistration_test.md section 6, on hand-computed toy numbers."""

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(9, 0), 2 / 512)           # 2 * C(9,0) / 2^9
        self.assertAlmostEqual(mcnemar_exact(1, 5), 2 * 7 / 64)        # 2 * (C(6,0) + C(6,1)) / 2^6
        self.assertEqual(mcnemar_exact(5, 1), mcnemar_exact(1, 5))     # two-sided, symmetric
        self.assertEqual(mcnemar_exact(4, 4), 1.0)                     # capped at 1
        self.assertAlmostEqual(mcnemar_exact(0, 4), 0.125)
        self.assertLess(mcnemar_exact(0, 2000), 1e-300)                # large n stays exact (integer arithmetic)
        with self.assertRaises(ValueError):
            mcnemar_exact(-1, 2)

    def test_holm(self):
        adj = holm([0.01, 0.04, 0.03, 0.005])
        # sorted 0.005, 0.01, 0.03, 0.04 -> 4*0.005, 3*0.01, max(0.03, 2*0.03), max(0.06, 0.04)
        for got, want in zip(adj, [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(holm([0.9, 0.5]), [1.0, 1.0])
        self.assertEqual(holm([]), [])

    def test_discordant(self):
        self.assertEqual(discordant([1, 1, 0, 0, 1], [1, 0, 1, 0, 0]), (2, 1))
        with self.assertRaises(ValueError):
            discordant([1], [1, 0])


class RunGuardTest(unittest.TestCase):
    """Run-once guard and the single retry (pre-registration sections 4.1 and 9), on synthetic records."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "c.jsonl"
        self.keys = {("a", frozenset({"1"})), ("a", frozenset({"2"})), ("b", frozenset({"3"}))}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, recs):
        import json
        self.cache.write_text("".join(json.dumps(r) + "\n" for r in recs))

    def recs(self, error_at=None):
        out = []
        for i, (c, ids) in enumerate(sorted(self.keys, key=lambda k: (k[0], sorted(k[1])))):
            r = {"county": c, "image_ids": sorted(ids), "names": ["x"]}
            if i == error_at:
                r = {**r, "names": [], "error": "TimeoutError"}
            out.append(r)
        return out

    def test_status_and_guard(self):
        self.assertEqual(runguard.cache_status(self.cache, self.keys)["missing"], 3)
        self.write(self.recs()[:2])
        st = runguard.guard_pass(self.cache, self.keys)              # incomplete: resume allowed
        self.assertEqual((st["lines"], st["missing"]), (2, 1))
        self.write(self.recs())
        with self.assertRaises(SystemExit):                          # complete: second pass refused
            runguard.guard_pass(self.cache, self.keys)
        self.write(self.recs() + self.recs()[:1])
        self.assertEqual(runguard.cache_status(self.cache, self.keys)["duplicates"], 1)
        with self.assertRaises(SystemExit):
            runguard.guard_pass(self.cache, self.keys)
        self.cache.write_text(self.cache.read_text() + "{not json")
        with self.assertRaisesRegex(ValueError, "line 4"):
            runguard.read_records(self.cache)

    def test_retry_once(self):
        self.write(self.recs(error_at=1)[:2])
        with self.assertRaises(SystemExit):                          # pass not finished
            runguard.retry_errors(self.cache, self.keys, lambda r: r)
        self.write(self.recs(error_at=1))
        self.cache.chmod(0o600)
        calls = []
        counts = runguard.retry_errors(self.cache, self.keys,
                                       lambda r: calls.append(1) or {"county": r["county"], "image_ids": r["image_ids"], "names": ["y"]})
        self.assertEqual((counts["errors_before"], counts["retried"], counts["errors_after"], len(calls)), (1, 1, 0, 1))
        recs = runguard.read_records(self.cache)
        self.assertEqual(sum("retry_of_error" in r for r in recs), 1)
        self.assertEqual(len(recs), 3)
        self.assertEqual(self.cache.stat().st_mode & 0o777, 0o600)   # file mode kept
        self.assertTrue(runguard.retry_marker(self.cache).exists())
        with self.assertRaises(SystemExit):                          # once only
            runguard.retry_errors(self.cache, self.keys, lambda r: r)

    def test_append_log(self):
        log = Path(self.tmp.name) / "log.csv"
        runguard.append_log(log, {"phase": "pass", "ok": 2})
        runguard.append_log(log, {"phase": "retry", "retried": 1})
        import csv
        rows = list(csv.DictReader(log.open()))
        self.assertEqual([r["phase"] for r in rows], ["pass", "retry"])
        self.assertEqual(rows[1]["retried"], "1")


if __name__ == "__main__":
    unittest.main()
