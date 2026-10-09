"""
Phase 2 evaluation of PLSS normalizers on the test split (single run with the frozen rule_v2.0).

Normalizers, all fed the same try08 NER spans (testdata_all_predicted.geojson):
  rule_v2[county] / rule_v2[state] / rule_v2[none]  src/normalizer/rule_based.py (frozen constants)
  regex_v1                                          src/normalizer/baselines.py (dev/recompute_table3.py logic)
  gpt4o_auto                                        TRIPLE_p of the GPT-4o batch files (before the 81 overrides)

Deeds: single-polygon test deeds with a gold section (dev/derive_gold_trs.py), joined by deed key.
Denominators: all such deeds, and those with any PLSS span (the headline). Reported overall and per county:
  * coverage, triples per covered deed, k histogram, abstention reasons;
  * top-1 and any-of-k accuracy at 1x1 (section polygon contains the deed) and 6x6 (township union
    contains it), scored on the unchanged merged_plss_5070.geojson, DIR used when present;
  * field accuracy of the top-1 reading vs gold (T, R, S, Rdir, exact T+R+S);
  * the oracle ceiling (gold T, R and S all among the span candidates of any span) and each normalizer's
    accuracy as a share of it, plus exact accuracy on oracle-recoverable deeds only;
  * error taxonomy of top-1 misses (deeds with spans): abstain:<reason>, gold_not_in_spans (NER-limited),
    gold_not_among_valid (recoverable values, but no valid combination produced it), wrong_pick (gold was a
    valid reading but not top-1), wrong_value (baselines: recoverable but wrong);
  * the paper's original cascade on each normalizer's triples (paper_scoring), which must reproduce
    1,174 / 1,183 for regex_v1 and 1,215 / 1,225 for gpt4o_auto.

Privacy: reads image_ids, cnty_name, the span columns, TRIPLE_p / RANGE_p and geometry; prints and writes
COUNTS ONLY. Errors inside data loops report the exception type and a numeric index only.

    mapprejudice/bin/python dev/eval_normalizer.py
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.normalizer import TRS, Result, spans_from_fields  # noqa: E402
from src.normalizer.baselines import RegexV1Normalizer  # noqa: E402
from src.normalizer.candidates import candidate_values, read_spans  # noqa: E402
from src.normalizer.constraints import Constraints  # noqa: E402
from src.normalizer.frozen import VERSION  # noqa: E402
from src.normalizer.rule_based import RuleBasedNormalizer  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "dev" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt3 = _load("recompute_table3")
cso = _load("check_split_overlap")
DEFAULTS = {
    "gold": REPO_ROOT / "output" / "derived" / "gold_trs_622.csv",
    "valid": REPO_ROOT / "output" / "derived" / "plss_valid_trs.json",
    "summary": REPO_ROOT / "results" / "normalizer" / "phase2_eval.json",
}
COUNTY_STATE = {"anoka": "MN", "dakota": "MN", "olmsted": "MN", "sherburne": "MN", "washington": "MN", "milwaukee": "WI"}


def load_gold(path):
    import csv
    gold = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and r["single"] == "True" and r["status"] == "ok":
                gold[(r["county"], r["deed_key"])] = TRS(int(r["T"]), r["Tdir"], int(r["R"]), r["Rdir"], int(r["S"]))
    return gold


def deed_key(props):
    ids = cso.parse_image_ids(props.get("image_ids"))
    return (cso.county_key(props.get("cnty_name")), "|".join(sorted(set(ids)))) if ids else None


def gpt_results(paths, errors):
    """{deed key: Result} from the GPT-4o batch files (TRIPLE_p before the overrides)."""
    import pandas as pd
    gdf = pd.concat([rt3.read(p) for p in paths], ignore_index=True)
    out = {}
    for i, props in enumerate(gdf.drop(columns="geometry", errors="ignore").to_dict("records")):
        try:
            key = deed_key(props)
            triples = rt3.as_triples(props.get("TRIPLE_p"), rt3.range_direction(props.get("RANGE_p")))
            ranked = [(TRS(t, "", r, d, s), 1.0 - 0.01 * j) for j, (s, t, r, d) in enumerate(triples)]
            out[key] = Result(ranked, None if ranked else "no_triple")
        except Exception as e:  # type and index only
            errors[f"gpt#{i}:{type(e).__name__}"] += 1
    return out


class Scorer:
    """1x1 / 6x6 containment on the unchanged merged PLSS layer (same lookups as recompute_table3.py)."""

    def __init__(self, plss):
        self.idx = rt3.plss_index(plss)
        self.towns = rt3.township_polygons(self.idx)

    def hits(self, trs, deed):
        if trs is None or trs.T is None or trs.R is None:
            return False, False
        d = trs.Rdir
        one = trs.S is not None and rt3.hit(
            self.idx["sec_dir"].get((trs.S, trs.T, trs.R, d), []) if d else self.idx["sec"].get((trs.S, trs.T, trs.R), []),
            deed, "contains")
        town = self.towns[0].get((trs.T, trs.R, d)) if d else self.towns[1].get((trs.T, trs.R))
        return bool(one), bool(town is not None and town.contains(deed))


def oracle_ok(spans, g):
    readings = read_spans(spans)
    return all(getattr(g, k) in candidate_values(readings, k, False) for k in ("T", "R", "S"))


def classify(res, g, oracle, is_rule):
    """Top-1 miss category for a deed with spans (None when the top-1 T+R+S is exact)."""
    top = res.top
    if top is None:
        return f"abstain:{res.reason}"
    if (top.T, top.R, top.S) == (g.T, g.R, g.S):
        return None
    if not oracle:
        return "gold_not_in_spans"
    if not is_rule:
        return "wrong_value"
    valid = {(t.T, t.R, t.S) for t in res.info.get("valid_readings", [])}
    return "wrong_pick" if (g.T, g.R, g.S) in valid else "gold_not_among_valid"


def evaluate(deeds, results, scorer, errors):
    """deeds: list of dicts(key, county, geom, gold, has_span, oracle). results: {name: {key: Result}}."""
    per = {name: defaultdict(lambda: {"all": Counter(), "with_span": Counter()}) for name in results}
    paper = {name: [] for name in results}
    for i, d in enumerate(deeds):
        for name, res_by_key in results.items():
            try:
                res = res_by_key.get(d["key"]) or Result([], "no_triple")
                g, top = d["gold"], res.top
                c = Counter(deeds=1, oracle_TRS=d["oracle"])
                k = len(res.ranked)
                c["covered"] += top is not None
                c["triples"] += k
                c[f"k_{min(k, 3)}"] += 1
                if top is None:
                    c[f"reason:{res.reason}"] += 1
                one, six = scorer.hits(top, d["geom"])
                c["top1_1x1"] += one
                c["top1_6x6"] += six
                any_hits = [scorer.hits(t, d["geom"]) for t in res.top_k()]
                c["anyk_1x1"] += any(h[0] for h in any_hits)
                c["anyk_6x6"] += any(h[1] for h in any_hits)
                if top is not None:
                    for f in ("T", "R", "S", "Rdir"):
                        c[f"field_{f}"] += getattr(top, f) == getattr(g, f)
                    exact = (top.T, top.R, top.S) == (g.T, g.R, g.S)
                    c["field_exact_TRS"] += exact
                    c["exact_and_oracle"] += exact and d["oracle"]
                if d["has_span"]:
                    cat = classify(res, g, d["oracle"], name.startswith("rule"))
                    if cat:
                        c[f"miss:{cat}"] += 1
                for county in (d["county"], "ALL"):
                    per[name][county]["all"].update(c)
                    if d["has_span"]:
                        per[name][county]["with_span"].update(c)
                paper[name].append((d["geom"], [t.paper_triple() for t in res.top_k()]))
            except Exception as e:  # type and index only
                errors[f"{name}#{i}:{type(e).__name__}"] += 1
    return per, paper


def finish(c):
    n = c.get("deeds", 0) or 1
    out = dict(sorted(c.items()))
    pct = {m: round(100 * c.get(m, 0) / n, 2) for m in ("covered", "top1_1x1", "top1_6x6", "anyk_1x1", "anyk_6x6",
                                                       "field_exact_TRS", "oracle_TRS")}
    out["pct"] = pct
    out["triples_per_covered"] = round(c.get("triples", 0) / (c.get("covered") or 1), 3)
    orc = c.get("oracle_TRS", 0)
    out["share_of_oracle"] = {
        m: round(c.get(m, 0) / orc, 4) if orc else None for m in ("top1_1x1", "field_exact_TRS")}
    out["exact_given_oracle"] = round(c.get("exact_and_oracle", 0) / orc, 4) if orc else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--raw", default=str(rt3.DEFAULTS["raw"]))
    ap.add_argument("--batch", action="append", default=None, help="GPT-4o batch GeoJSONs (default: try08 1-4)")
    ap.add_argument("--plss", default=str(rt3.DEFAULTS["plss"]))
    ap.add_argument("--gold", default=str(DEFAULTS["gold"]))
    ap.add_argument("--valid", default=str(DEFAULTS["valid"]))
    ap.add_argument("--summary", default=str(DEFAULTS["summary"]))
    args = ap.parse_args(argv)

    errors = Counter()
    gold = load_gold(args.gold)
    constraints = Constraints.from_json(args.valid)
    raw = rt3.read(args.raw)
    deeds, seen = [], set()
    for i, (props, geom) in enumerate(zip(raw.drop(columns="geometry").to_dict("records"), raw.geometry)):
        try:
            key = deed_key(props)
            if key not in gold or key in seen:
                errors["raw_row_not_in_gold_or_duplicate"] += 1
                continue
            seen.add(key)
            spans = spans_from_fields(props)
            deeds.append({"key": key, "county": key[0], "geom": geom, "gold": gold[key], "spans": spans,
                          "has_span": bool(spans), "oracle": oracle_ok(spans, gold[key])})
        except Exception as e:
            errors[f"raw#{i}:{type(e).__name__}"] += 1

    norms = [RuleBasedNormalizer(constraints, level) for level in ("county", "state", "none")] + [RegexV1Normalizer()]
    results = {}
    for norm in norms:
        out = {}
        for i, d in enumerate(deeds):
            try:
                out[d["key"]] = norm.normalize(d["spans"], d["county"], COUNTY_STATE.get(d["county"]))
            except Exception as e:
                errors[f"{norm.name}#{i}:{type(e).__name__}"] += 1
        results[norm.name] = out
    results["gpt4o_auto"] = gpt_results(args.batch or rt3.DEFAULTS["batches"], errors)

    scorer = Scorer(rt3.read(args.plss))
    per, paper = evaluate(deeds, results, scorer, errors)
    summary = {
        "inputs": {"rule_version": VERSION, "raw_rows": len(raw), "gold_test_deeds": len(gold), "joined_deeds": len(deeds),
                   "deeds_with_span": sum(d["has_span"] for d in deeds), "gold_not_in_raw": len(set(gold) - seen)},
        "results": {name: {county: {den: finish(c) for den, c in blocks.items()} for county, blocks in sorted(by.items())}
                    for name, by in per.items()},
        "paper_scoring": {},
    }
    for name, pairs in paper.items():
        p = dict(rt3.score_plss(pairs, scorer.idx, scorer.towns, errors))
        if p.get("with_triple"):
            for m in ("orig_1x1", "orig_6x6", "fixed_1x1_contains", "fixed_6x6_contains"):
                p[m + "_pct_of_with_triple"] = round(100 * p.get(m, 0) / p["with_triple"], 2)
        summary["paper_scoring"][name] = p
    summary["errors"] = dict(Counter(k.split(":")[-1] if "#" in k else k for k in errors.elements()))

    text = json.dumps(summary, indent=2, sort_keys=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text + "\n", encoding="utf-8")
    print(f"{'normalizer':18s} {'county':10s} {'n':>5s} cover  1x1   6x6   anyk1  anyk6  exact  oracle share  k/deed")
    for name, by in summary["results"].items():
        for county, blocks in by.items():
            b = blocks["with_span"]
            if not b.get("deeds"):
                continue
            pc = b["pct"]
            share = b["share_of_oracle"]["field_exact_TRS"]
            print(f"{name:18s} {county:10s} {b['deeds']:5d} {pc['covered']:5.1f} {pc['top1_1x1']:5.1f} {pc['top1_6x6']:5.1f} "
                  f"{pc['anyk_1x1']:6.1f} {pc['anyk_6x6']:6.1f} {pc['field_exact_TRS']:6.1f} {pc['oracle_TRS']:6.1f} "
                  f"{share if share is not None else float('nan'):5.3f} {b['triples_per_covered']:6.2f}")
    print(f"(with-span denominator; all-deed denominator and paper_scoring in {Path(args.summary).name})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
