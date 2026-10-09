"""
ONE-TIME test evaluation of the frozen matcher_v1.0 (approved by Me 2026-10-09; CLAUDE.md "Matcher v1.0
selection rule"). Metrics are exactly the pre-registered list:

  primary   share of eligible test deeds whose centroid lies in the predicted plat, denominator (b) all
            eligible deeds (no top-1 = miss)
  also      the same with denominator (a) eligible deeds with a SUBDIVISION_p prediction; plat accuracy vs
            the geometric weak label; intersects rate; contains rate
  paper     the paper's original subdivision criterion: >= 1 plat name with fuzz.ratio >= 70 for any raw
            predicted string, candidates from all five counties (load_subd_experiment.py), over (a) and over
            all test deeds with a prediction (the paper's 2,162-deed denominator)

Input: try08 test NER predictions (testdata_all_predicted.geojson; reproduced exactly by dev/infer_try08.py).
Eligible = county has a plat layer and single-part deed geometry. Reported per county and overall.

Run-once guard: refuses to run if the summary file already exists. Prints and writes aggregate numbers only.

    python dev/eval_matcher_test.py [--summary results/matcher/test_v1.json]
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
from eval_matcher_val import evaluate, original_columns, preds_of  # noqa: E402
from weak_labels import summarize, weak_labels  # noqa: E402

from src.matcher import MatchConfig, PlatIndex  # noqa: E402
from src.matcher import frozen  # noqa: E402

TEST_PRED = pl.MP / "<RUN_DIR>" / "output" / "try08" / "prediction" / "testdata_all_predicted.geojson"
SUMMARY = REPO_ROOT / "results" / "matcher" / "test_v1.json"
PAPER_CFG = MatchConfig(scorer="ratio", norm="raw", own_county=False, cutoff=70.0)


def load_test_deeds(path, crs=pl.CRS):
    """Like weak_labels.load_deeds, but for the test predictions (one row per deed, no n_parts column)."""
    import geopandas as gpd
    g = gpd.read_file(path).to_crs(crs)
    g["geometry"] = g.geometry.make_valid()
    rows = g.drop(columns="geometry").to_dict("records")
    return [{"county": str(r.get("cnty_name") or "").strip().lower(), "props": r, "geometry": geom,
             "n_parts": (len(geom.geoms) if hasattr(geom, "geoms") else 1) if geom is not None and not geom.is_empty else 0}
            for r, geom in zip(rows, g.geometry)]


def paper_original(deeds, labels, index, errors):
    """Paper criterion per county: counts over eligible deeds with a prediction (a) and over all deeds with one."""
    per = defaultdict(Counter)
    for i, d in enumerate(deeds):
        try:
            preds = preds_of(d["props"])
            if not preds:
                continue
            hit, hit_int = original_columns(index, preds, d["county"], PAPER_CFG, d["geometry"])
            eligible = labels[i]["status"] == "eligible"
            for g in (d["county"], "ALL"):
                c = per[g]
                c["all_with_pred"] += 1
                c["all_hit"] += hit
                c["all_hit_intersects"] += hit_int
                c["a_with_pred"] += eligible
                c["a_hit"] += eligible and hit
                c["a_hit_intersects"] += eligible and hit_int
        except Exception as e:  # noqa: BLE001
            errors[f"paper:{type(e).__name__}"] += 1
    pct = lambda n, den: round(100 * n / den, 2) if den else None  # noqa: E731
    return {g: {"counts": dict(c), "pct_all": pct(c["all_hit"], c["all_with_pred"]),
                "pct_all_intersects": pct(c["all_hit_intersects"], c["all_with_pred"]),
                "pct_a": pct(c["a_hit"], c["a_with_pred"]), "pct_a_intersects": pct(c["a_hit_intersects"], c["a_with_pred"])}
            for g, c in per.items()}


def run(deeds, plats, index, errors):
    labels = weak_labels(deeds, plats, errors)
    no_trs = [frozenset()] * len(deeds)  # matcher_v1.0 has no section constraint
    metrics = evaluate(deeds, labels, no_trs, index, frozen.CONFIG, errors)
    return {"weak_labels": summarize(deeds, labels), "matcher": metrics,
            "paper_original": paper_original(deeds, labels, index, errors)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pred", default=str(TEST_PRED))
    ap.add_argument("--summary", default=str(SUMMARY))
    args = ap.parse_args(argv)
    if Path(args.summary).exists():
        sys.exit(f"refused: {args.summary} exists; test is run once per frozen version (CLAUDE.md)")
    t0, errors = time.time(), Counter()
    plats, pairs = pl.load_plats()
    index = PlatIndex(plats, frozen.frozen_normalizer(pairs))  # raises if the name rules changed
    deeds = load_test_deeds(args.pred)
    result = {"matcher": frozen.VERSION, "config": frozen.CONFIG.name(), "split": "test", "deeds": len(deeds),
              **run(deeds, plats, index, errors), "errors": dict(errors), "seconds": round(time.time() - t0, 1)}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(result, indent=1) + "\n")
    print(f"{frozen.VERSION} on test | deeds {len(deeds)} | errors {dict(errors)} | {result['seconds']} s")
    print(f"{'county':<11}{'b':>5}{'a':>5}{'cin(b)':>8}{'cin(a)':>8}{'acc(b)':>8}{'acc(a)':>8}{'int(a)':>8}{'int(b)':>8}"
          f"{'cont(a)':>8}{'cont(b)':>8}{'paper(all)':>11}{'paper(a)':>9}")
    for g in sorted(result["matcher"], key=lambda x: (x == "ALL", x)):
        m, p = result["matcher"][g], result["paper_original"].get(g, {})
        c = m["counts"]
        print(f"{g:<11}{c['b_all']:>5}{c.get('a_with_pred', 0):>5}{m['centroid_in_pct_b']!s:>8}{m['centroid_in_pct_a']!s:>8}"
              f"{m['plat_acc_b']!s:>8}{m['plat_acc_a']!s:>8}{m['intersects_pct_a']!s:>8}{m['intersects_pct_b']!s:>8}"
              f"{m['contains_pct_a']!s:>8}{m['contains_pct_b']!s:>8}{p.get('pct_all')!s:>11}{p.get('pct_a')!s:>9}")


if __name__ == "__main__":
    main()
