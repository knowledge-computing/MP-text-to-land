"""
Matcher ablation on VAL. EXPLORATORY: val only, not pre-registered, never run on test (configs/matcher_ablation_val.yaml).

(a) The matcher ladder A -> A' -> B -> C on every val query source:
      A   paper_b1 (casefold exact match; dev/subdiv_14_extract_x_matcher.ExactIndex)
      A'  exact match after n1 (upper case, punctuation -> space)            -> case/punctuation normalization
      B   fuzz.ratio on n1, cutoff 70, no learned rules                       -> fuzziness
      C   matcher_v1.0 (fuzz.ratio on n3 = n1 + rules learned from plat layers) -> the learned rules
    plus B_raw (fuzz.ratio on raw, case-sensitive strings) for try08 only. All use the partner plat layers, the deed's
    own county, primary names and matcher_v1.0's tie rule. Sources = the six extractors' val outputs
    (configs/extract_x_matcher.yaml) and the oracle query (the deed's add_mod values).
(b) Oracle ceiling (add_mod -> A / A' / B / C) and an error decomposition of each failed eligible deed (first
    category that applies, order fixed in the config): no extraction; no gold; gold not resolvable (the oracle
    fails; gold not in the gazetteer / in it); right name (fuzz.ratio on n1 >= 90 between some extracted string and
    some add_mod value) but no top-1 or wrong plat; wrong name, no top-1 or wrong plat.

Metrics are matcher_v1.0's own (dev/eval_matcher_val.evaluate on all val deeds, weak labels from dev/weak_labels.py,
eligibility = county with a partner plat layer and single-part geometry; centroid-in over (b) all eligible deeds is
primary). Per-deed outcomes are recomputed alongside and must sum to evaluate()'s counts.

Outputs are written ONLY if (1) A and C reproduce every compared value of results/extract_x_matcher/val_cells.csv
for all six extractors, (2) try08 + B / C reproduce results/matcher/val_grid.json (ratio|n1|county|primary,
ratio|n3|county|primary), and (3) the per-deed outcomes sum to evaluate()'s counts. Otherwise it exits with code 2.

Writes results/matcher_ablation/val_{cells,errors,paired}.csv and val_summary.json. Prints and writes counts, rates
and check results only; exceptions as type + numeric deed index. Imports frozen files read-only and edits none.

    mapprejudice/bin/python dev/matcher_ablation_val.py          # the real val run
    mapprejudice/bin/python dev/matcher_ablation_val.py --toy    # synthetic data only (temp folder)
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))
import yaml  # noqa: E402
from eval_matcher_val import evaluate, preds_of  # noqa: E402  (frozen, read-only)
from subdiv_14_extract_x_matcher import ExactIndex, llm_names  # noqa: E402  (frozen, read-only)
from weak_labels import weak_labels  # noqa: E402  (frozen, read-only)

from src.matcher import MatchConfig, NameNormalizer, PlatIndex, frozen, geom_metrics, learn_rules, n1  # noqa: E402
from src.matcher.match import Match  # noqa: E402
from src.subdiv_match.stats import mcnemar_exact  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "matcher_ablation_val.yaml"
STATUS = "exploratory (val only, not pre-registered)"
LADDER = ("A", "A_prime", "B", "C")
OUTCOME_KEYS = ("top1", "plat_correct", "centroid_in", "contains", "intersects")
FAIL_ORDER = ["no_extraction", "gold_no_gold", "gold_not_in_gazetteer", "gold_in_gazetteer_oracle_fails",
              "right_name_no_top1", "right_name_wrong_plat", "wrong_name_no_top1", "wrong_name_wrong_plat"]
CORRECT = ["correct_gold_in_gazetteer", "correct_gold_not_in_gazetteer", "correct_no_gold"]
NO_TRS = frozenset()   # no section constraint in any matcher here
VAL_CELLS_MATCHER = {"paper_b1": "A", "matcher_v1.0": "C"}
GRID_KEY = {"B": "ratio|n1|county|primary", "C": "ratio|n3|county|primary"}
GRID_FIELD = {"b_all": "eligible_b", "a_with_pred": "with_extraction_a"}


# ---------------------------------------------------------------- matchers
class ExactN1Index(PlatIndex):
    """A': the first predicted string whose n1 form equals the n1 form of a partner plat primary name in the deed's
    county (first plat with that form in layer order); identity (county, n1(name)) as for every matcher here."""

    def top1(self, preds, county, cfg=None, triples=frozenset()):
        if not hasattr(self, "_exact_n1"):
            self._exact_n1 = {}
            for i, p in enumerate(self.plats):
                key = n1(p["name"]) if isinstance(p["name"], str) else ""
                if key:
                    self._exact_n1.setdefault(p["county"], {}).setdefault(key, i)
        for q in preds:
            i = self._exact_n1.get(county, {}).get(n1(q))
            if i is not None:
                ident = self.ident[i]
                return Match(ident, list(self.by_ident[ident]), 100.0, 100.0)
        return None


def fuzzy_config(m):
    return MatchConfig(scorer=m["scorer"], norm=m["norm"], own_county=bool(m["own_county"]),
                       aliases=bool(m["aliases"]), section="none", k=0.0, cutoff=float(m["cutoff"]))


def build_matchers(cfg, plats, norm3):
    m = cfg["matchers"]
    if (m["A"]["form"], m["A_prime"]["form"], m["C"]["kind"]) != ("casefold", "n1", "frozen"):
        sys.exit("refused: matchers A / A' / C in the config differ from the code")
    return {"A": (ExactIndex(plats, norm3), frozen.CONFIG),          # cfg ignored by top1; used by evaluate's
            "A_prime": (ExactN1Index(plats, norm3), frozen.CONFIG),  # paper-criterion column only
            "B": (PlatIndex(plats, norm3), fuzzy_config(m["B"])),
            "C": (PlatIndex(plats, norm3), frozen.CONFIG),
            "B_raw": (PlatIndex(plats, norm3), fuzzy_config(cfg["extra_rows"]["B_raw"]))}


# ---------------------------------------------------------------- per-deed logic (pure)
def name_agree(queries, gold, threshold):
    from rapidfuzz import fuzz
    return any(fuzz.ratio(n1(q), n1(g)) >= threshold for q in queries for g in gold)


def categorize(outcome, queries, gold, oracle_ok, gold_in_gaz, agree):
    """One eligible deed -> its category (CORRECT for hits, FAIL_ORDER for misses; first that applies)."""
    if outcome["centroid_in"]:
        return "correct_no_gold" if not gold else CORRECT[0] if gold_in_gaz else CORRECT[1]
    if not queries:
        return "no_extraction"
    if not gold:
        return "gold_no_gold"
    if not oracle_ok:
        return "gold_in_gazetteer_oracle_fails" if gold_in_gaz else "gold_not_in_gazetteer"
    kind = "right_name" if agree else "wrong_name"
    return f"{kind}_wrong_plat" if outcome["top1"] else f"{kind}_no_top1"


def gazetteer_names(plats):
    out = defaultdict(set)
    for p in plats:
        if isinstance(p["name"], str) and n1(p["name"]):
            out[p["county"]].add(n1(p["name"]))
    return out


def deed_outcomes(deeds, labels, queries, index, mcfg, errors, tag):
    """-> per deed: None (not eligible) or {has_query, top1, plat_correct, centroid_in, contains, intersects}."""
    from shapely.ops import unary_union
    out = []
    for i, (d, lab, q) in enumerate(zip(deeds, labels, queries)):
        if lab["status"] != "eligible":
            out.append(None)
            continue
        rec = {"has_query": bool(q), **{k: False for k in OUTCOME_KEYS}}
        try:
            m = index.top1(q, d["county"], mcfg, NO_TRS) if q else None
            if m is not None:
                truth = index.ident[lab["plat"]] if lab["plat"] is not None else None
                gm = geom_metrics(unary_union([index.plats[j]["geometry"] for j in m.plats]), d["geometry"])
                rec.update(top1=True, plat_correct=m.ident == truth, centroid_in=bool(gm["centroid_in"]),
                           contains=bool(gm["contains"]), intersects=bool(gm["intersects"]))
        except Exception as e:  # noqa: BLE001 - type + index only
            errors[f"{tag}:outcome:{type(e).__name__}#{i}"] += 1
        out.append(rec)
    return out


def aggregate(deeds, outcomes):
    per = defaultdict(Counter)
    for d, o in zip(deeds, outcomes):
        if o is None:
            continue
        for g in (d["county"], "ALL"):
            per[g]["eligible_b"] += 1
            per[g]["with_extraction_a"] += o["has_query"]
            for k in OUTCOME_KEYS:
                per[g][k] += o[k]
    return per


def county_order(keys):
    return sorted(keys, key=lambda k: (k == "ALL", k))


# ---------------------------------------------------------------- the run
def run(cfg, plats, norm3, deeds, gold, sources, errors):
    """sources: {name: per-deed query lists} (extractors only); gold: per-deed add_mod lists."""
    if cfg["errors"]["order"] != FAIL_ORDER or cfg["errors"]["correct_split"] != CORRECT:
        sys.exit("refused: the error categories in the config differ from the code")
    labels = weak_labels(deeds, plats, errors)
    matchers = build_matchers(cfg, plats, norm3)
    oracle = cfg["oracle"]["source_name"]
    queries = {**sources, oracle: [preds_of({"SUBDIVISION_p": list(g)}) for g in gold]}
    gaz = gazetteer_names(plats)
    in_gaz = [bool(g) and any(n1(x) in gaz.get(d["county"], set()) for x in g) for d, g in zip(deeds, gold)]
    no_trs = [NO_TRS] * len(deeds)
    cells, outcomes, consistency = [], {}, []
    for src, qs in queries.items():
        ds = [{**d, "props": {**d["props"], "SUBDIVISION_p": q}} for d, q in zip(deeds, qs)]
        extra = ("B_raw",) if src in cfg["extra_rows"]["B_raw"]["sources"] else ()
        for m in LADDER + extra:
            index, mcfg = matchers[m]
            res = evaluate(ds, labels, no_trs, index, mcfg, errors)          # matcher_v1.0's metric code
            outs = deed_outcomes(deeds, labels, qs, index, mcfg, errors, f"{src}|{m}")
            agg = aggregate(deeds, outs)
            for county in county_order(set(res) | set(agg)):
                r = res.get(county, {"counts": {}})
                c = r["counts"]
                for f in ("b_all", "a_with_pred") + OUTCOME_KEYS:
                    if c.get(f, 0) != agg[county][GRID_FIELD.get(f, f)]:
                        consistency.append(f"{src}|{m}|{county}|{f}")
                cells.append({"status": "exploratory", "source": src, "matcher": m, "county": county,
                              "eligible_b": c.get("b_all", 0), "with_extraction_a": c.get("a_with_pred", 0),
                              **{k: c.get(k, 0) for k in OUTCOME_KEYS},
                              "centroid_in_pct_b": r.get("centroid_in_pct_b"),
                              "centroid_in_pct_a": r.get("centroid_in_pct_a"),
                              "plat_acc_pct_b": r.get("plat_acc_b"), "plat_acc_pct_a": r.get("plat_acc_a"),
                              "contains_pct_b": r.get("contains_pct_b"), "intersects_pct_b": r.get("intersects_pct_b")})
            outcomes[(src, m)] = outs

    thr = float(cfg["errors"]["name_agreement"]["threshold"])
    err_rows = []
    for src in sources:
        agree = [name_agree(q, g, thr) if q and g else False for q, g in zip(queries[src], gold)]
        for m in LADDER:
            per = defaultdict(Counter)
            for i, (d, o) in enumerate(zip(deeds, outcomes[(src, m)])):
                if o is None:
                    continue
                cat = categorize(o, queries[src][i], gold[i], outcomes[(oracle, m)][i]["centroid_in"], in_gaz[i],
                                 agree[i])
                for g in (d["county"], "ALL"):
                    per[g]["eligible_b"] += 1
                    per[g][cat] += 1
            for county in county_order(per):
                c = per[county]
                err_rows.append({"status": "exploratory", "source": src, "matcher": m, "county": county,
                                 "eligible_b": c["eligible_b"], "correct": sum(c[k] for k in CORRECT),
                                 "failed": sum(c[k] for k in FAIL_ORDER), **{k: c[k] for k in FAIL_ORDER + CORRECT}})

    paired = []
    for src in queries:
        for x, y in cfg["paired"]["pairs"]:
            ox = [o["centroid_in"] for o in outcomes[(src, x)] if o is not None]
            oy = [o["centroid_in"] for o in outcomes[(src, y)] if o is not None]
            b = sum(p and not q for p, q in zip(ox, oy))
            c = sum(q and not p for p, q in zip(ox, oy))
            paired.append({"status": "exploratory", "source": src, "x": x, "y": y, "eligible_b": len(ox),
                           "centroid_in_x": sum(ox), "centroid_in_y": sum(oy), "only_x": b, "only_y": c,
                           "p_exact_mcnemar_info_only": mcnemar_exact(b, c)})

    gold_by = defaultdict(Counter)
    for i, (d, lab) in enumerate(zip(deeds, labels)):
        for g in (d["county"], "ALL"):
            el = lab["status"] == "eligible"
            gold_by[g]["deeds"] += 1
            gold_by[g]["deeds_with_gold"] += bool(gold[i])
            gold_by[g]["eligible"] += el
            gold_by[g]["eligible_with_gold"] += el and bool(gold[i])
            gold_by[g]["eligible_gold_in_gazetteer"] += el and in_gaz[i]
    return {"cells": cells, "errors": err_rows, "paired": paired, "consistency_mismatches": consistency,
            "gold": {g: dict(gold_by[g]) for g in county_order(gold_by)}}


# ---------------------------------------------------------------- checks
def _num(v):
    return None if v in (None, "") else float(v)


def check_reproduction(cells, val_cells_rows, grid, rep, extractors):
    """-> (number of compared values, list of mismatch labels 'source|matcher|county|field'; no data values)."""
    ours = {(r["source"], r["matcher"], r["county"]): r for r in cells}
    mismatches, compared, seen = [], 0, set()
    for row in val_cells_rows:
        if row["extractor"] not in extractors or row["matcher"] not in VAL_CELLS_MATCHER:
            continue
        key = (row["extractor"], VAL_CELLS_MATCHER[row["matcher"]], row["county"])
        seen.add(key[:2])
        o = ours.get(key)
        if o is None:
            mismatches.append("|".join(key) + "|missing")
            continue
        for f in rep["int_fields"]:
            compared += 1
            if int(row[f]) != int(o[f]):
                mismatches.append("|".join(key) + f"|{f}")
        for f in rep["pct_fields"]:
            compared += 1
            if _num(row[f]) != _num(o[f]):
                mismatches.append("|".join(key) + f"|{f}")
    for ex in extractors:
        for m in VAL_CELLS_MATCHER.values():
            if (ex, m) not in seen:
                mismatches.append(f"{ex}|{m}|not_in_val_cells")
            if not any(k[:2] == (ex, m) for k in ours):
                mismatches.append(f"{ex}|{m}|not_computed")
    ner = [ex for ex in extractors if ex == "try08_ner"]
    for m, gkey in GRID_KEY.items():
        for county, r in (grid.get("grid", {}).get(gkey) or {}).items():
            for src in ner:
                o = ours.get((src, m, county))
                if o is None:
                    mismatches.append(f"{src}|{m}|{county}|missing_vs_grid")
                    continue
                for f in rep["grid_fields"]:
                    compared += 1
                    if r["counts"].get(f, 0) != o[GRID_FIELD.get(f, f)]:
                        mismatches.append(f"{src}|{m}|{county}|{f}_vs_grid")
    if not ner:
        mismatches.append("try08_ner|not_in_sources")
    return compared, mismatches


# ---------------------------------------------------------------- output
def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_outputs(out, result, summary):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "val_cells.csv", result["cells"])
    write_csv(out / "val_errors.csv", result["errors"])
    write_csv(out / "val_paired.csv", result["paired"])
    (out / "val_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def print_report(result):
    cells = {(r["source"], r["matcher"], r["county"]): r for r in result["cells"]}
    srcs = list(dict.fromkeys(r["source"] for r in result["cells"]))
    print(f"[{STATUS}] centroid-in (b), ALL eligible deeds: count / pct")
    print(f"{'source':<22}" + "".join(f"{m:>16}" for m in LADDER + ("B_raw",)))
    for s in srcs:
        row = [cells.get((s, m, "ALL")) for m in LADDER + ("B_raw",)]
        print(f"{s:<22}" + "".join(f"{(str(r['centroid_in']) + ' / ' + str(r['centroid_in_pct_b'])) if r else '-':>16}"
                                   for r in row))
    print("error decomposition, ALL (counts):")
    keys = ["correct"] + FAIL_ORDER
    print(f"{'source|matcher':<26}" + "".join(f"{k[:14]:>15}" for k in keys))
    for r in result["errors"]:
        if r["county"] == "ALL":
            print(f"{r['source'] + '|' + r['matcher']:<26}" + "".join(f"{r[k]:>15}" for k in keys))


def summary_of(cfg_bytes, ex_bytes, result, checks, errors, meta, t0):
    import rapidfuzz
    import shapely
    return {"version": yaml.safe_load(cfg_bytes)["version"], "status": STATUS,
            "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
            "extractor_config_sha256": hashlib.sha256(ex_bytes).hexdigest(),
            "matcher_C": frozen.VERSION, "frozen_rules_sha256": frozen.RULES_SHA256, **meta,
            "gold": result["gold"], "checks": checks, "errors": dict(errors),
            "versions": {"python": platform.python_version(), "rapidfuzz": rapidfuzz.__version__,
                         "shapely": shapely.__version__},
            "seconds": round(time.time() - t0, 1)}


def finish(cfg_bytes, ex_bytes, result, val_cells_rows, grid, extractors, errors, meta, out, t0):
    cfg = yaml.safe_load(cfg_bytes)
    n, mism = check_reproduction(result["cells"], val_cells_rows, grid, cfg["reproduction"], extractors)
    checks = {"reproduction_compared": n, "reproduction_mismatches": len(mism),
              "consistency_mismatches": len(result["consistency_mismatches"])}
    print("checks:", checks, "| errors:", dict(errors))
    if mism or result["consistency_mismatches"]:
        for lab in (mism + result["consistency_mismatches"])[:20]:
            print("  mismatch:", lab)
        print("REFUSED: checks failed; nothing written")
        return 2
    print_report(result)
    write_outputs(out, result, summary_of(cfg_bytes, ex_bytes, result, checks, errors, meta, t0))
    print("written:", out)
    return 0


# ---------------------------------------------------------------- real data
def load_real(ex_cfg, errors):
    import plat_layers as pl
    from src.subdiv_match.data import deed_key
    from src.subdiv_match.deeds import load_split_deeds
    from weak_labels import load_deeds
    warnings.filterwarnings("ignore", message="Error parsing datetimes")   # the warning quotes a field value
    plats, pairs = pl.load_plats()
    norm3 = frozen.frozen_normalizer(pairs)                    # raises if the frozen name rules changed
    rf = pl.pp.run_files()
    for p in (rf["pred"]["val"], rf["splits"]["val"]):
        if "test" in Path(str(p)).name.lower():
            sys.exit("refused: the test split is frozen")
    deeds = load_deeds(rf["pred"]["val"])                      # refuses a test file
    gold_by_key = load_split_deeds(rf["splits"]["val"])
    keys = [deed_key(d["props"]) for d in deeds]
    gold = [list((gold_by_key.get(k) or {}).get("add_mod") or []) for k in keys]
    sources, missing = {}, {}
    for name, ex in ex_cfg["extractors"].items():
        if ex["kind"] == "ner":
            sources[name] = [preds_of(d["props"]) for d in deeds]
            continue
        path = REPO_ROOT / ex["cache"]
        if not path.exists():
            sys.exit(f"refused: no val cache for {name}")
        cache, n_err = llm_names(path)
        errors[f"{name}:cached_errors"] += n_err
        missing[name] = sum(k not in cache for k in keys)
        sources[name] = [preds_of({"SUBDIVISION_p": list(cache[k][0])}) if k in cache else [] for k in keys]
    meta = {"val_deeds": len(deeds), "gold_join_missing_keys": sum(k not in gold_by_key for k in keys),
            "missing_in_cache": missing}
    return plats, norm3, deeds, gold, sources, meta


# ---------------------------------------------------------------- synthetic toy data
def toy_inputs():
    from shapely.geometry import MultiPolygon, box
    plats = [{"county": "anoka", "name": nm, "aliases": [], "geometry": box(x, 0, x + 100, 100), "sections": set()}
             for nm, x in (("OAK RIDGE", 0), ("OAK RIDGE 2ND", 100), ("ELM PARK", 200), ("MAPLE HEIGHTS", 300))]
    plats.append({"county": "dakota", "name": "PINE HILL", "aliases": [], "geometry": box(5000, 0, 5100, 100),
                  "sections": set()})
    rows = [  # county, try08 query, llm query, gold, deed box x0, parts
        ("anoka", ["Oak Ridge"], ["OAK RIDGE"], ["OAK RIDGE"], 10, 1),           # correct everywhere
        ("anoka", ["Oak Ridge."], ["oak ridge"], ["OAK RIDGE"], 20, 1),          # punctuation: A fails, A' ok
        ("anoka", ["Oak Rdge"], [], ["OAK RIDGE"], 30, 1),                       # typo: B ok; llm no extraction
        ("anoka", ["Oak Ridge Second"], ["Oak Ridge 2nd"], ["OAK RIDGE 2ND"], 110, 1),   # ordinal: C rule
        ("anoka", [], ["Elm Park"], ["ELM PARK"], 210, 1),                       # try08 no extraction
        ("anoka", ["Birch Woods"], ["Birch Woods"], ["BIRCH WOODS"], 310, 1),    # gold not in gazetteer
        ("anoka", ["Elm Park"], ["Maple Heights"], ["OAK RIDGE"], 40, 1),        # wrong name, wrong plat
        ("anoka", ["Zzzz Qqqq"], ["Xxxx"], [], 50, 1),                           # no gold
        ("anoka", ["Oak Ridge"], ["Oak Ridge"], ["OAK RIDGE"], 60, 2),           # multi-part: not eligible
        ("washington", ["Oak Ridge"], ["Oak Ridge"], ["OAK RIDGE"], 70, 1),      # no plat layer
        ("dakota", ["Pine Hill"], ["pine hill"], ["PINE HILL"], 5010, 1),        # correct
    ]
    deeds, gold, ner, llm = [], [], [], []
    for county, q_ner, q_llm, g, x, parts in rows:
        geom = box(x, 10, x + 5, 15) if parts == 1 else MultiPolygon([box(x, 10, x + 2, 12), box(x + 3, 13, x + 4, 14)])
        deeds.append({"county": county, "props": {"SUBDIVISION_p": q_ner}, "geometry": geom, "n_parts": parts})
        gold.append(g)
        ner.append(preds_of({"SUBDIVISION_p": q_ner}))
        llm.append(preds_of({"SUBDIVISION_p": q_llm}))
    norm3 = NameNormalizer(learn_rules([], 5))             # ordinal classes only (no layer pairs in the toy)
    return plats, norm3, deeds, gold, {"try08_ner": ner, "toy_llm": llm}


def toy_expected(plats, norm3, deeds, sources):
    """The reproduction targets, built the way subdiv_14 (val_cells) and eval_matcher_val (val_grid) build them."""
    labels = weak_labels(deeds, plats, Counter())
    no_trs = [NO_TRS] * len(deeds)
    idx = {"paper_b1": ExactIndex(plats, norm3), "matcher_v1.0": PlatIndex(plats, norm3)}
    rows = []
    for src, qs in sources.items():
        ds = [{**d, "props": {**d["props"], "SUBDIVISION_p": q}} for d, q in zip(deeds, qs)]
        for m in ("paper_b1", "matcher_v1.0"):
            for county, r in evaluate(ds, labels, no_trs, idx[m], frozen.CONFIG, Counter()).items():
                c = r["counts"]
                rows.append({"extractor": src, "matcher": m, "county": county, "eligible_b": c.get("b_all", 0),
                             "with_extraction_a": c.get("a_with_pred", 0), "top1": c.get("top1", 0),
                             "centroid_in": c.get("centroid_in", 0), "centroid_in_pct_b": r["centroid_in_pct_b"],
                             "centroid_in_pct_a": r["centroid_in_pct_a"], "plat_acc_pct_b": r["plat_acc_b"],
                             "plat_acc_pct_a": r["plat_acc_a"], "contains_pct_b": r["contains_pct_b"],
                             "intersects_pct_b": r["intersects_pct_b"]})
    ds = [{**d, "props": {**d["props"], "SUBDIVISION_p": q}} for d, q in zip(deeds, sources["try08_ner"])]
    b_cfg = MatchConfig("ratio", "n1", True, False)
    grid = {"grid": {GRID_KEY["B"]: evaluate(ds, labels, no_trs, PlatIndex(plats, norm3), b_cfg, Counter()),
                     GRID_KEY["C"]: evaluate(ds, labels, no_trs, PlatIndex(plats, norm3), frozen.CONFIG, Counter())}}
    # val_cells.csv holds strings; round-trip the toy rows the same way
    return [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows], grid


def run_toy(out, tamper=False):
    t0, errors = time.time(), Counter()
    cfg_bytes = CONFIG.read_bytes()
    plats, norm3, deeds, gold, sources = toy_inputs()
    val_cells_rows, grid = toy_expected(plats, norm3, deeds, sources)
    if tamper:
        val_cells_rows[0]["centroid_in"] = str(int(val_cells_rows[0]["centroid_in"]) + 1)
    result = run(yaml.safe_load(cfg_bytes), plats, norm3, deeds, gold, sources, errors)
    return finish(cfg_bytes, b"toy", result, val_cells_rows, grid, list(sources), errors,
                  {"toy": True, "val_deeds": len(deeds)}, out, t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--toy", action="store_true", help="synthetic data only; writes to a temp folder")
    ap.add_argument("--out", help="output folder (default: the config's out)")
    args = ap.parse_args(argv)
    if args.toy:
        out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="matcher_ablation_toy_"))
        return run_toy(out)
    t0, errors = time.time(), Counter()
    cfg_bytes = CONFIG.read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    ex_path = REPO_ROOT / cfg["extractor_config"]
    ex_bytes = ex_path.read_bytes()
    ex_cfg = yaml.safe_load(ex_bytes)
    plats, norm3, deeds, gold, sources, meta = load_real(ex_cfg, errors)
    result = run(cfg, plats, norm3, deeds, gold, sources, errors)
    with open(REPO_ROOT / cfg["reproduction"]["val_cells"], newline="") as fh:
        val_cells_rows = list(csv.DictReader(fh))
    grid = json.loads((REPO_ROOT / cfg["reproduction"]["val_grid"]).read_text())
    out = Path(args.out) if args.out else REPO_ROOT / cfg["out"]
    return finish(cfg_bytes, ex_bytes, result, val_cells_rows, grid, list(ex_cfg["extractors"]), errors, meta, out, t0)


if __name__ == "__main__":
    sys.exit(main())
