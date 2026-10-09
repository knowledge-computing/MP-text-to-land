"""
Select matcher_v1.0 by the pre-registered rule in CLAUDE.md ("Matcher v1.0 selection rule"):

  selection set = train ∪ val try08 predictions, deduplicated by deed key (county, set of image_ids),
                  eligible = county has a plat layer and single-part deed geometry (as in test)
  A = ratio | n3 | own county | primary, no section constraint
  B = A + soft PLSS-section bonus k = 5 (readings from normalizer rule_v2.0)
  decision: exact two-sided McNemar on the per-deed outcome "deed centroid inside the predicted plat"
            (no top-1 = 0) over all eligible deeds; B only if p < 0.05 and B wins more discordant deeds

Also reports, for A and B, the full aggregate metric table of dev/eval_matcher_val.evaluate and McNemar on
plat accuracy (information only). Prints and writes aggregate numbers only.

    python dev/select_matcher.py --pred <val predicted> --pred <train predicted> [--summary results/matcher/selection_v1.json]
"""
import argparse
import ast
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))
import plat_layers as pl  # noqa: E402
from eval_matcher_val import evaluate, preds_of  # noqa: E402
from normalize_predictions import readings  # noqa: E402
from weak_labels import load_deeds, summarize, weak_labels  # noqa: E402

from src.matcher import MatchConfig, NameNormalizer, PlatIndex, geom_metrics, learn_rules  # noqa: E402

CANDIDATES = {"A": MatchConfig("ratio", "n3", True), "B": MatchConfig("ratio", "n3", True, section="soft", k=5)}
ALPHA = 0.05


def deed_key(props):
    ids = props.get("image_ids")
    if isinstance(ids, str):
        try:
            ids = ast.literal_eval(ids)
        except (ValueError, SyntaxError):
            ids = [ids]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = ids if isinstance(ids, (list, tuple)) else ([] if ids is None else [ids])
    return (str(props.get("cnty_name") or "").strip().lower(), frozenset(map(str, ids)))


def dedupe(deed_lists, stats):
    seen, out = {}, []
    for deeds in deed_lists:
        for d in deeds:
            k = deed_key(d["props"])
            if k in seen:
                stats["duplicate_deeds_dropped"] += 1
                stats["duplicate_same_subdivision_pred"] += preds_of(d["props"]) == preds_of(seen[k]["props"])
                continue
            seen[k] = d
            out.append(d)
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: b = A wins (A=1, B=0), c = B wins. Returns p."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def per_deed(deeds, labels, trs, index, cfg, errors):
    """{deed index: (centroid_in, plat_correct)} for eligible deeds."""
    from shapely.ops import unary_union
    out = {}
    for i, d in enumerate(deeds):
        if labels[i]["status"] != "eligible":
            continue
        cin = ok = 0
        try:
            preds = preds_of(d["props"])
            m = index.top1(preds, d["county"], cfg, trs[i]) if preds else None
            if m is not None:
                cin = int(geom_metrics(unary_union([index.plats[j]["geometry"] for j in m.plats]), d["geometry"])["centroid_in"])
                ok = int(labels[i]["plat"] is not None and m.ident == index.ident[labels[i]["plat"]])
        except Exception as e:  # noqa: BLE001
            errors[f"{cfg.name()}:{type(e).__name__}"] += 1
        out[i] = (cin, ok)
    return out


def compare(a, b, slot):
    wins_a = sum(1 for i in a if a[i][slot] == 1 and b[i][slot] == 0)
    wins_b = sum(1 for i in a if a[i][slot] == 0 and b[i][slot] == 1)
    return {"n": len(a), "A_success": sum(v[slot] for v in a.values()), "B_success": sum(v[slot] for v in b.values()),
            "A_only": wins_a, "B_only": wins_b, "p_exact_two_sided": round(mcnemar_exact(wins_a, wins_b), 6)}


def decide(primary):
    return "B" if primary["p_exact_two_sided"] < ALPHA and primary["B_only"] > primary["A_only"] else "A"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pred", action="append", required=True, help="train / val predictions (test is refused)")
    ap.add_argument("--summary", default=str(REPO_ROOT / "results" / "matcher" / "selection_v1.json"))
    args = ap.parse_args(argv)
    t0, errors, stats, rstats = time.time(), Counter(), Counter(), Counter()
    deeds = dedupe([load_deeds(p) for p in args.pred], stats)
    stats["deeds_after_dedup"] = len(deeds)
    plats, pairs = pl.load_plats()
    index = PlatIndex(plats, NameNormalizer(learn_rules(pairs, 5)))
    labels = weak_labels(deeds, plats, errors)
    trs = readings(deeds, stats=rstats)
    outcomes = {k: per_deed(deeds, labels, trs, index, cfg, errors) for k, cfg in CANDIDATES.items()}
    primary = compare(outcomes["A"], outcomes["B"], 0)
    result = {
        "inputs": [Path(p).name for p in args.pred], "counts": dict(stats), "readings": dict(rstats),
        "weak_labels": summarize(deeds, labels),
        "candidates": {k: c.name() for k, c in CANDIDATES.items()},
        "mcnemar_primary_centroid_in": primary,
        "mcnemar_plat_accuracy_info_only": compare(outcomes["A"], outcomes["B"], 1),
        "decision": decide(primary),
        "metrics": {k: evaluate(deeds, labels, trs, index, cfg, errors) for k, cfg in CANDIDATES.items()},
        "errors": dict(errors), "seconds": round(time.time() - t0, 1),
    }
    result["selected_config"] = result["candidates"][result["decision"]]
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("counts", "readings", "candidates", "mcnemar_primary_centroid_in",
                                             "mcnemar_plat_accuracy_info_only", "decision", "selected_config", "errors")}, indent=1))


if __name__ == "__main__":
    main()
