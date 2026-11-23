"""
DESCRIPTIVE RECOMPUTATION of GeoHCC '25 Table 3 (PLSS) under other spatial predicates, from the saved published
outputs (all_output.geojson = try08 + GPT-4o + 81 overrides). Not a new result and not a method change: the
published numbers used `contains` only (2025 load_plss_construct_triple.py; dev/recompute_table3.py reproduces
them). The paper's text says deeds "fall within or partially overlap" the PLSS cell; this script reports what the
same 1,470 deeds would score if overlap counted.

Cascades (dev/recompute_table3.py's logic; that file is imported read-only and not changed):
  original   1x1: section (S,T,R,DIR) polygon; if no hit and the deed's DIR is empty, section (S,T,R) polygon.
             6x6: only for deeds still without a hit AND with an empty DIR, any SECTION polygon of the (T,R).
  corrected  every deed gets a 1x1 test (DIR used when present) and a 6x6 test against the dissolved township.
Predicates: contains (published), intersects (shapely: shares any point, boundary contact included; as the
"fixed_*_intersects" columns of results/provenance/gpt_trial_saved.json), overlap_area (intersection area > 0).

Checks: original/contains must give 1,258 / 1,270 and corrected/contains 1,258 / 1,291 of 1,470 (the committed
provenance values), else the script exits 2 without writing. Prints and writes COUNTS only; exceptions as type +
numeric index.

    mapprejudice/bin/python dev/recompute_table3_predicates.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import recompute_table3 as rt  # noqa: E402  (read-only reuse)

PREDICATES = ("contains", "intersects", "overlap_area")
EXPECTED = {("original", "contains"): (1258, 1270), ("corrected", "contains"): (1258, 1291)}
STATUS = "descriptive recomputation from saved published outputs (no method change)"


def test(geoms, deed, predicate):
    for g in geoms:
        if g is None:
            continue
        if predicate == "contains" and g.contains(deed):
            return True
        if predicate == "intersects" and g.intersects(deed):
            return True
        if predicate == "overlap_area" and g.intersects(deed) and g.intersection(deed).area > 0:
            return True
    return False


def score(deeds, idx, towns, errors):
    town_dir, town_any = towns
    c = Counter(deeds_total=len(deeds))
    for i, (deed, triples) in enumerate(deeds):
        try:
            if not triples:
                continue
            c["with_triple"] += 1
            d = triples[0][3]
            for p in PREDICATES:
                s1 = any(test(idx["sec_dir"].get(t, []), deed, p) for t in triples)
                s2 = not s1 and d == "" and any(test(idx["sec"].get(t[:3], []), deed, p) for t in triples)
                s3 = not s1 and not s2 and d == "" and any(test(idx["tr"].get(t[1:3], []), deed, p) for t in triples)
                c[f"original_1x1_{p}"] += s1 or s2
                c[f"original_6x6_{p}"] += s1 or s2 or s3
                one = any(test(idx["sec_dir"].get(t, []) if t[3] else idx["sec"].get(t[:3], []), deed, p) for t in triples)
                six = any(test([town_dir.get(t[1:4]) if t[3] else town_any.get(t[1:3])], deed, p) for t in triples)
                c[f"corrected_1x1_{p}"] += one
                c[f"corrected_6x6_{p}"] += six
                c[f"corrected_6x6_or_1x1_{p}"] += one or six
            c["failed_step1_with_dir_never_retried"] += not any(
                test(idx["sec_dir"].get(t, []), deed, "contains") for t in triples) and d != ""
        except Exception as e:  # noqa: BLE001 - type + index only
            errors[f"plss#{i}:{type(e).__name__}"] += 1
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pred", default=rt.DEFAULTS["pred"])
    ap.add_argument("--plss", default=rt.DEFAULTS["plss"])
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "provenance" / "gpt_trial_saved_predicates.json"))
    args = ap.parse_args(argv)
    errors = Counter()
    gdf = rt.read(args.pred)
    triples = [rt.as_triples(v) for v in gdf["TRIPLE_p"]]
    idx = rt.plss_index(rt.read(args.plss))
    c = score(list(zip(gdf.geometry, triples)), idx, rt.township_polygons(idx), errors)
    n = c["with_triple"]
    table = {f"{cas}_{res}_{p}": {"hits": c[f"{cas}_{res}_{p}"], "pct_of_with_triple": round(100 * c[f"{cas}_{res}_{p}"] / n, 2)}
             for cas in ("original", "corrected") for res in ("1x1", "6x6") for p in PREDICATES}
    mism = [f"{cas}/{p}" for (cas, p), (a, b) in EXPECTED.items()
            if (c[f"{cas}_1x1_{p}"], c[f"{cas}_6x6_{p}"]) != (a, b)]
    result = {"status": STATUS, "deeds_total": c["deeds_total"], "with_triple": n,
              "failed_step1_with_dir_never_retried": c["failed_step1_with_dir_never_retried"],
              "table3": table, "corrected_6x6_or_1x1": {p: c[f"corrected_6x6_or_1x1_{p}"] for p in PREDICATES},
              "check_vs_published": {"mismatches": mism}, "errors": dict(errors)}
    print(json.dumps(result, indent=1))
    if mism:
        print("REFUSED: contains does not reproduce the published counts; nothing written")
        return 2
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", Path(args.out).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
