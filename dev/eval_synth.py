"""
Per-slot accuracy of PLSS normalizers on the synthetic corpora (synthetic data only; safe to print).

Corpora (src/normalizer/synth.py, shared base phrases): clean, ocr, span, ocr+span. Normalizers:
rule_v2 at county / state / none constraint levels, and regex v1. For each example the prediction is
the top-1 reading; slot accuracy compares it with the gold TRS that matches it best (examples with two
gold readings accept either). Abstaining counts as wrong; coverage is reported separately.

Seeds: --set dev (seed 0) was used while debugging the rules; --set heldout (seed 20261008, generated
after the rules were frozen) gives the numbers to report.

    mapprejudice/bin/python dev/eval_synth.py --toy --set heldout                # fake PLSS table
    mapprejudice/bin/python dev/eval_synth.py --valid-json output/derived/plss_valid_trs.json --set heldout
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.normalizer import synth  # noqa: E402
from src.normalizer.baselines import RegexV1Normalizer  # noqa: E402
from src.normalizer.candidates import candidate_values, read_spans  # noqa: E402
from src.normalizer.constraints import Constraints  # noqa: E402
from src.normalizer.rule_based import RuleBasedNormalizer  # noqa: E402

SLOTS = ("T", "R", "S", "Rdir")
SEEDS = {"dev": 0, "heldout": 20261008}


def best_gold(pred, gold):
    return max(gold, key=lambda g: sum(getattr(pred, k) == getattr(g, k) for k in ("T", "R", "S")))


def oracle(examples):
    """Share of examples whose gold T / R / S (and all three) appear among the candidates of their spans."""
    c = Counter(n=len(examples))
    for e in examples:
        readings = read_spans(e.spans)
        vals = {k: candidate_values(readings, k) for k in ("T", "R", "S")}
        hit = {k: any((getattr(g, k) in vals[k]) if getattr(g, k) is not None else True for g in e.gold) for k in vals}
        for k, v in hit.items():
            c[f"oracle_{k}"] += v
        c["oracle_TRS"] += any(all((getattr(g, k) is None or getattr(g, k) in vals[k]) for k in vals) for g in e.gold)
    return {k: round(v / (c["n"] or 1), 4) for k, v in c.items() if k != "n"}


def score(examples, norm):
    c, reasons, k_hist = Counter(n=len(examples)), Counter(), Counter()
    for e in examples:
        res = norm.normalize(e.spans, e.county, e.state)
        top = res.top
        k_hist[min(len(res.ranked), 3)] += 1
        if top is None:
            reasons[res.reason] += 1
            continue
        c["covered"] += 1
        g = best_gold(top, e.gold)
        for k in SLOTS:
            c[f"slot_{k}"] += getattr(top, k) == getattr(g, k)
        key = lambda t: (t.T, t.R, t.S, t.Rdir)
        c["exact_top1"] += key(top) in {key(x) for x in e.gold}
        c["exact_top1_ignore_dir"] += (top.T, top.R, top.S) in {(x.T, x.R, x.S) for x in e.gold}
        c["township_top1"] += (top.T, top.R) in {(x.T, x.R) for x in e.gold}
        c["exact_any_of_k"] += bool({key(t) for t in res.top_k()} & {key(x) for x in e.gold})
        c["triples_total"] += len(res.ranked)
    n = c["n"] or 1
    out = {k: round(c[k] / n, 4) for k in ("covered", "exact_top1", "exact_top1_ignore_dir", "township_top1",
                                         "exact_any_of_k", *[f"slot_{s}" for s in SLOTS])}
    out["triples_per_covered"] = round(c["triples_total"] / (c["covered"] or 1), 3)
    out["k_hist"] = {str(k): v for k, v in sorted(k_hist.items())}
    out["reasons"] = dict(reasons)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--toy", action="store_true")
    src.add_argument("--valid-json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--set", choices=sorted(SEEDS), default="heldout", help="dev = debugging seed; heldout = report")
    ap.add_argument("--seed", type=int, default=None, help="override the --set seed")
    ap.add_argument("--ocr-p", type=float, default=0.05)
    ap.add_argument("--span-p", type=float, default=0.2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    args.seed = SEEDS[args.set] if args.seed is None else args.seed
    if args.toy:
        table = synth.toy_valid()
        constraints = Constraints.from_trs(table)
    else:
        table = synth.load_valid_json(args.valid_json)
        constraints = Constraints.from_json(args.valid_json)
    sets = synth.corpora(table, args.n, args.seed, args.ocr_p, args.span_p)
    norms = [RuleBasedNormalizer(constraints, level) for level in ("county", "state", "none")] + [RegexV1Normalizer()]
    from src.normalizer.frozen import VERSION
    result = {"inputs": {"table": "toy" if args.toy else Path(args.valid_json).name, "n": args.n, "seed": args.seed,
                         "set": args.set if args.seed == SEEDS[args.set] else "custom", "rule_version": VERSION,
                         "ocr_p": args.ocr_p, "span_p": args.span_p,
                         "layouts": dict(Counter(e.layout for e in sets["clean"]))},
              "results": {name: {norm.name: score(ex, norm) for norm in norms} for name, ex in sets.items()},
              "oracle": {name: oracle(ex) for name, ex in sets.items()}}
    for name, per in result["results"].items():
        o = result["oracle"][name]
        print(f"\n[{name}] oracle (gold among span candidates): T {o['oracle_T']:.3f}  R {o['oracle_R']:.3f}  "
              f"S {o['oracle_S']:.3f}  T+R+S {o['oracle_TRS']:.3f}")
        print(f"[{name}]  {'normalizer':18s} cover  exact  ex-dir  twp    T      R      S      Rdir   any-k  k/deed")
        for nn, m in per.items():
            print(f"{'':10s}{nn:18s} " + " ".join(f"{m[k]:.3f}" for k in (
                "covered", "exact_top1", "exact_top1_ignore_dir", "township_top1", "slot_T", "slot_R", "slot_S",
                "slot_Rdir", "exact_any_of_k")) + f"  {m['triples_per_covered']:.2f}")
    tag = args.tag or f"synth_{'toy' if args.toy else 'plss'}_{result['inputs']['set']}"
    out = REPO_ROOT / "results" / "normalizer" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
