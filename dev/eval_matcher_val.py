"""
S5: select the subdivision matcher configuration on VAL (the test split is frozen; CLAUDE.md).

For each configuration (scorer x ablation step x aliases x PLSS-section mode) and each deed, the matcher picks
one top-1 plat (src.matcher). Reported per county and overall, aggregate numbers only:

  plat accuracy   top-1 plat identity == geometric weak label (dev/weak_labels.py), with denominators
                  (a) eligible deeds with a SUBDIVISION_p prediction and (b) all eligible deeds
  original        the paper's subdivision-level criterion (2025 load_subd_experiment.py; 16.18% on test):
                  deeds with >= 1 plat name at fuzz.ratio >= 70 for any predicted string / deeds with a
                  prediction; reported for every config's pool, plus the share whose >= 70 candidates
                  include one intersecting the deed (dev/recompute_table3.py's extra column)
  geometry        predicted plat polygon (union of all polygons of that identity) vs deed polygon:
                  contains (whole deed inside; the 2025 PLSS code's criterion), intersects (area > 0; the
                  paper's "or partially overlap"), centroid_in; rates over (a) and (b) (no top-1 = fail);
                  distributions over deeds with a top-1: covered fraction, IoU, centroid distance (m)

PLSS readings for section modes come from the local normalizer rule_v2.0 (dev/normalize_predictions.py).
Eligible = county has a plat layer and the deed geometry is single-part (as in test).

Privacy: prints and writes counts, rates and distributions only; errors as exception type counts.

    python dev/eval_matcher_val.py --pred <try08_val_predicted.geojson> [--summary results/matcher/val_grid.json]
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))
import plat_layers as pl  # noqa: E402
from normalize_predictions import readings  # noqa: E402
from weak_labels import FRAC_BINS, dist, load_deeds, summarize, weak_labels  # noqa: E402

from src.matcher import MatchConfig, NameNormalizer, PlatIndex, geom_metrics, learn_rules  # noqa: E402
from src.matcher.match import SCORERS  # noqa: E402

EMPTY = {"", "[]", "none", "nan", "null"}
DIST_BINS = [("0", 1e-9), ("<100m", 100), ("<500m", 500), ("<1km", 1000), ("<5km", 5000), (">=5km", float("inf"))]
SCORER_NAMES = ("ratio", "WRatio", "token_set_ratio")


def preds_of(props):
    import ast
    v = props.get("SUBDIVISION_p")
    if hasattr(v, "tolist") and not isinstance(v, str):
        v = v.tolist()
    if isinstance(v, str):
        if v.strip().lower() in EMPTY:
            return []
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    return [x for x in v if isinstance(x, str) and x.strip().lower() not in EMPTY]


def configs():
    out = []
    for sc in SCORER_NAMES:
        out += [MatchConfig(sc, "raw", False), MatchConfig(sc, "n1", False), MatchConfig(sc, "n1", True),
                MatchConfig(sc, "n3", True), MatchConfig(sc, "n3", True, aliases=True)]
        for al in (False, True):
            out.append(MatchConfig(sc, "n3", True, aliases=al, section="hard"))
            out += [MatchConfig(sc, "n3", True, aliases=al, section="soft", k=k) for k in (3, 5, 10)]
    return out


def original_columns(index, preds, county, cfg, deed_geom):
    """>= 1 candidate at cutoff (paper criterion) and whether any such candidate intersects the deed."""
    from rapidfuzz import process
    keys, members = index.pool(county, cfg)
    fn = SCORERS[cfg.scorer]
    hit = False
    for p in preds:
        for k, _, _ in process.extract(index._form(p, cfg.norm), keys, scorer=fn, score_cutoff=cfg.cutoff,
                                       limit=None, processor=None):
            hit = True
            if any(index.plats[i]["geometry"].intersects(deed_geom) for i in members[k]):
                return True, True
    return hit, False


def evaluate(deeds, labels, trs, index, cfg, errors):
    from shapely.ops import unary_union
    per = defaultdict(Counter)
    dists = defaultdict(lambda: defaultdict(list))
    for i, d in enumerate(deeds):
        lab = labels[i]
        if lab["status"] != "eligible":
            continue
        try:
            preds = preds_of(d["props"])
            for g in (d["county"], "ALL"):
                per[g]["b_all"] += 1
                per[g]["a_with_pred"] += bool(preds)
            if not preds:
                continue
            orig, orig_int = original_columns(index, preds, d["county"], cfg, d["geometry"])
            m = index.top1(preds, d["county"], cfg, trs[i])
            truth = index.ident[lab["plat"]] if lab["plat"] is not None else None
            gm = None
            if m is not None:
                gm = geom_metrics(unary_union([index.plats[j]["geometry"] for j in m.plats]), d["geometry"])
            for g in (d["county"], "ALL"):
                c = per[g]
                c["original_any_candidate"] += orig
                c["original_candidate_intersects"] += orig_int
                c["top1"] += m is not None
                c["plat_correct"] += m is not None and m.ident == truth
                if gm:
                    for key in ("contains", "intersects", "centroid_in"):
                        c[key] += gm[key]
                    for key in ("covered", "iou", "centroid_m"):
                        dists[g][key].append(gm[key])
        except Exception as e:  # noqa: BLE001
            errors[f"{cfg.name()}:{type(e).__name__}"] += 1
    out = {}
    for g, c in per.items():
        a, b = c["a_with_pred"], c["b_all"]
        pct = lambda n, den: round(100 * n / den, 2) if den else None  # noqa: E731
        out[g] = {"counts": dict(c),
                  "plat_acc_a": pct(c["plat_correct"], a), "plat_acc_b": pct(c["plat_correct"], b),
                  "original_pct_a": pct(c["original_any_candidate"], a),
                  "original_intersects_pct_a": pct(c["original_candidate_intersects"], a),
                  **{f"{k}_pct_a": pct(c[k], a) for k in ("contains", "intersects", "centroid_in")},
                  **{f"{k}_pct_b": pct(c[k], b) for k in ("contains", "intersects", "centroid_in")},
                  "covered": dist(dists[g]["covered"], FRAC_BINS), "iou": dist(dists[g]["iou"], FRAC_BINS),
                  "centroid_m": dist(dists[g]["centroid_m"], DIST_BINS)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pred", required=True, help="val predictions from dev/infer_try08.py (test is refused)")
    ap.add_argument("--summary", default=str(REPO_ROOT / "results" / "matcher" / "val_grid.json"))
    ap.add_argument("--min-rule-freq", type=int, default=5)
    args = ap.parse_args(argv)
    t0, errors, stats = time.time(), Counter(), Counter()
    deeds = load_deeds(args.pred)
    plats, pairs = pl.load_plats()
    rules = learn_rules(pairs, args.min_rule_freq)
    index = PlatIndex(plats, NameNormalizer(rules))
    labels = weak_labels(deeds, plats, errors)
    trs = readings(deeds, stats=stats)
    grid = {cfg.name(): evaluate(deeds, labels, trs, index, cfg, errors) for cfg in configs()}
    result = {"split": "val", "deeds": len(deeds), "weak_labels": summarize(deeds, labels), "readings": dict(stats),
              "rules": {"n_classes_tokens": len(rules["canon"]), "n_merges": len(rules["merges"]),
                        "n_learned_pairs": len(rules["used"])},
              "grid": grid, "errors": dict(errors), "seconds": round(time.time() - t0, 1)}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(result, indent=1) + "\n")
    print(f"{'config':<44}{'acc(a)':>8}{'acc(b)':>8}{'orig(a)':>8}{'cont(a)':>8}{'int(a)':>8}{'cin(a)':>8}{'IoU p50':>8}{'dist p50':>9}")
    for name, res in grid.items():
        r = res.get("ALL", {})
        q = lambda k, i=3: (r.get(k, {}).get("min,p10,p25,p50,p75,p90,max") or [None] * 7)[i]  # noqa: E731
        print(f"{name:<44}{r.get('plat_acc_a')!s:>8}{r.get('plat_acc_b')!s:>8}{r.get('original_pct_a')!s:>8}"
              f"{r.get('contains_pct_a')!s:>8}{r.get('intersects_pct_a')!s:>8}{r.get('centroid_in_pct_a')!s:>8}"
              f"{q('iou')!s:>8}{q('centroid_m')!s:>9}")
    print("errors:", dict(errors), "| seconds:", result["seconds"], "| summary:", args.summary)


if __name__ == "__main__":
    main()
