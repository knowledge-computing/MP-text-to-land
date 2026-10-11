"""
Aggregate-only recomputation of the GeoHCC '25 georeferencing tables from saved predictions.

Reproduces, from a saved test-prediction GeoJSON:
  * Table 3 (PLSS): the original cascade in the 2025 run folder's ($MP_RUN_DIR)
    load_plss_construct_triple.py (1x1 = section polygon .contains(deed); retry without
    Range direction only for deeds whose DIR is empty; 6x6 fallback only for those), plus
    corrected variants: every deed gets both a 1x1 and a 6x6 test, with the 6x6 test against
    the dissolved Township polygon, and each test runs with both `contains` and `intersects`.
  * Table 2 (RPSS): evaluate_predictions() from load_subd_ner_performance.py
    (deed-level, any prediction with rapidfuzz token_set_ratio >= 70, empty-truth deeds skipped).
  * Optional subdivision figure (--subd): share of deeds with >=1 plat name at fuzz.ratio >= 70
    (load_subd_experiment.py), and the share whose matched plat polygon also intersects the deed.

Triple sources (--triples):
  saved        TRIPLE_p as stored (e.g. all_output.geojson = GPT-4o trial + 81 manual overrides)
  batches      TRIPLE_p from the GPT-4o batch files before the overrides (--batch files)
  regex        recomputed from the raw SECTION_p/TOWNSHIP_p/RANGE_p strings with the local
               regex normalizer below (NEW code, not the paper-time normalizer)

Privacy: prints and writes COUNTS ONLY (no ids, strings, numbers read from deeds). Errors
inside data loops report the exception type and a numeric index only. Needs geopandas,
shapely and rapidfuzz (the mapprejudice/ venv has them).

    mapprejudice/bin/python dev/recompute_table3.py --tag gpt_trial_saved
"""

import argparse
import ast
import json
import numbers
import re
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import private_paths as pp  # noqa: E402

TRY08 = pp.run_dir("output", "try08")  # resolved only when a default is used
DEFAULTS = {
    "pred": TRY08 / "prediction" / "all_output.geojson",
    "raw": TRY08 / "prediction" / "testdata_all_predicted.geojson",
    "plss": TRY08 / "merged_plss_5070.geojson",
    "subd": TRY08 / "merged_subd_5070.geojson",
    "batches": [TRY08 / "prediction" / f"all_with_openainumber_batch{i}.geojson" for i in range(1, 5)],
}
CRS = "EPSG:5070"

# ---------------------------------------------------------------- regex normalizer (new)
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
TENS["fourty"] = 40


def words_to_int(words):
    """Parse a run of number words ('one hundred and six') to an int; None if no number word."""
    total, seen = 0, False
    for w in words:
        if w in UNITS:
            total += UNITS[w]; seen = True
        elif w in TENS:
            total += TENS[w]; seen = True
        elif w == "hundred" and seen:
            total = (total or 1) * 100
        elif w == "and" and seen:
            continue
        elif seen:
            break
    return total if seen else None


def normalize_number(text):
    """One PLSS mention -> int. Number words win over digits (the trial prompt's rule); else first digits."""
    if not isinstance(text, str):
        return None
    low = re.sub(r"[-‐–]", " ", text.lower())
    words = re.findall(r"[a-z]+", low)
    for i, w in enumerate(words):
        if w in UNITS or w in TENS:
            value = words_to_int(words[i:])
            if value:
                return value
    digits = re.findall(r"\d+", low)
    return int(digits[0]) if digits else None


def range_direction(range_value):
    """Same regex as check_and_modify_openai_number.py:192-196, applied to str(value)."""
    x = str(range_value)
    if re.search(r"West|west|WEST|\sW\b|\sw\b|\sW\.|\sw\.", x):
        return "W"
    if re.search(r"East|east|EAST|\sE\b|\se\b|\sE\.|\se\.", x):
        return "E"
    return ""


# ---------------------------------------------------------------- parsing helpers
def as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):  # numpy arrays from pyogrio list columns
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def as_triples(value, direction=None):
    """TRIPLE_p cell -> list of (sec, twp, rng, dir) tuples (ints or None)."""
    out = []
    for t in as_list(value):
        t = list(t) if isinstance(t, (list, tuple)) else []
        if len(t) == 3:
            t = t + [direction or ""]
        if len(t) != 4:
            continue
        nums = [int(x) if isinstance(x, numbers.Real) and not isinstance(x, bool) and x == x else None for x in t[:3]]
        out.append((*nums, t[3] if isinstance(t[3], str) else ""))
    return out


def regex_triples(props):
    lists = [[n for n in (normalize_number(s) for s in as_list(props.get(k))) if n is not None]
             for k in ("SECTION_p", "TOWNSHIP_p", "RANGE_p")]
    d = range_direction(props.get("RANGE_p"))
    return [(s, t, r, d) for s, t, r in zip_longest(*lists)]


# ---------------------------------------------------------------- PLSS scoring
def plss_index(plss):
    idx = {k: defaultdict(list) for k in ("sec_dir", "sec", "tr", "tr_dir")}
    for sec, twp, rng, d, geom in zip(plss["SEC"], plss["TWP"], plss["RNG"], plss["DIR"], plss.geometry):
        idx["sec_dir"][(sec, twp, rng, d)].append(geom)
        idx["sec"][(sec, twp, rng)].append(geom)
        idx["tr"][(twp, rng)].append(geom)
        idx["tr_dir"][(twp, rng, d)].append(geom)
    return idx


def township_polygons(idx):
    from shapely.ops import unary_union
    return {k: unary_union(v) for k, v in idx["tr_dir"].items()}, {k: unary_union(v) for k, v in idx["tr"].items()}


def hit(geoms, deed, predicate):
    return any(getattr(g, predicate)(deed) if predicate == "contains" else g.intersects(deed) for g in geoms)


def score_plss(deeds, idx, towns, errors):
    """deeds: list of (geometry, triples). Returns counts for the original cascade and fixed variants."""
    town_dir, town_any = towns
    c = Counter(deeds_total=len(deeds))
    for i, (deed, triples) in enumerate(deeds):
        try:
            if not triples:
                continue
            c["with_triple"] += 1
            d = triples[0][3]
            # original cascade (contains only)
            s1 = any(hit(idx["sec_dir"].get(t, []), deed, "contains") for t in triples)
            s2 = (not s1) and d == "" and any(hit(idx["sec"].get(t[:3], []), deed, "contains") for t in triples)
            s3 = (not s1) and (not s2) and d == "" and any(hit(idx["tr"].get(t[1:3], []), deed, "contains") for t in triples)
            c["orig_step1_dir"] += s1
            c["orig_step2_nodir"] += s2
            c["orig_step3_tuple"] += s3
            c["orig_1x1"] += s1 or s2
            c["orig_6x6"] += s1 or s2 or s3
            c["orig_failed_with_dir_never_retried"] += (not s1) and d != ""
            # fixed: every deed gets 1x1 and 6x6; DIR used when present, ignored when empty
            for pred in ("contains", "intersects"):
                one = any(hit(idx["sec_dir"].get(t, []) if t[3] else idx["sec"].get(t[:3], []), deed, pred) for t in triples)
                six_geoms = [town_dir.get(t[1:4]) if t[3] else town_any.get(t[1:3]) for t in triples]
                six = any(hit([g], deed, pred) for g in six_geoms if g is not None)
                c[f"fixed_1x1_{pred}"] += one
                c[f"fixed_6x6_{pred}"] += six
        except Exception as e:  # type and index only
            errors[f"plss#{i}:{type(e).__name__}"] += 1
    return c


# ---------------------------------------------------------------- Table 2 (RPSS)
def table2(rows):
    from rapidfuzz import fuzz
    fields = {"Subdivision": ("SUBDIVISION_p", "add_cov"), "City": ("CITY_p", "city"),
              "County": ("COUNTY_p", "cnty_name"), "State": ("STATE_p", "state")}
    out = {}
    for name, (pcol, tcol) in fields.items():
        tp = fp = fn = 0
        for props in rows:
            truth = props.get(tcol)
            if name == "State":
                truth = {"WI": "Wisconsin", "MN": "Minnesota"}.get(truth, truth)
            if not (isinstance(truth, str) and truth.strip()):
                continue
            preds = [p for p in as_list(props.get(pcol)) if isinstance(p, str)]
            if not preds:
                fn += 1
            elif any(fuzz.token_set_ratio(p.lower(), truth.lower()) >= 70 for p in preds):
                tp += 1
            else:
                fp += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {"TP": tp, "FP": fp, "FN": fn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    return out


# ---------------------------------------------------------------- subdivision figure
def subdivision(rows, geoms, plats, errors):
    from rapidfuzz import fuzz, process
    names = [n if isinstance(n, str) else "" for n in plats["SubdName"]]
    c = Counter()
    for i, (props, deed) in enumerate(zip(rows, geoms)):
        try:
            preds = [p for p in as_list(props.get("SUBDIVISION_p")) if isinstance(p, str)]
            if not preds:
                continue
            c["deeds_with_subdivision_prediction"] += 1
            matched = set()
            for p in preds:
                for _, _, j in process.extract(p, names, scorer=fuzz.ratio, score_cutoff=70, limit=None):
                    matched.add(j)
            c["deeds_with_>=1_plat_candidate"] += bool(matched)
            c["deeds_candidate_intersects_deed"] += any(g is not None and g.intersects(deed) for g in (plats.geometry.iloc[j] for j in matched))
        except Exception as e:
            errors[f"subd#{i}:{type(e).__name__}"] += 1
    return c


# ---------------------------------------------------------------- main
def read(path):
    import geopandas as gpd
    gdf = gpd.read_file(path)
    return gdf.to_crs(CRS) if gdf.crs and str(gdf.crs) != CRS else gdf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--triples", choices=["saved", "batches", "regex"], default="saved")
    ap.add_argument("--pred", default=DEFAULTS["pred"], help="prediction GeoJSON (saved triples / Table 2)")
    ap.add_argument("--raw", default=DEFAULTS["raw"], help="raw NER GeoJSON (for --triples regex)")
    ap.add_argument("--batch", action="append", default=None, help="GPT batch files (for --triples batches)")
    ap.add_argument("--plss", default=DEFAULTS["plss"])
    ap.add_argument("--subd", nargs="?", const=DEFAULTS["subd"], default=None, help="also compute the subdivision figure")
    ap.add_argument("--tag", default=None, help="output name under results/provenance/")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "provenance"))
    args = ap.parse_args(argv)

    import pandas as pd
    errors = Counter()
    if args.triples == "batches":
        parts = [read(p) for p in (args.batch or DEFAULTS["batches"])]
        gdf = pd.concat(parts, ignore_index=True)
        triples = [as_triples(v, range_direction(r)) for v, r in zip(gdf["TRIPLE_p"], gdf["RANGE_p"])]
    elif args.triples == "regex":
        gdf = read(args.raw)
        triples = [regex_triples(row) for row in gdf.drop(columns="geometry").to_dict("records")]
    else:
        gdf = read(args.pred)
        triples = [as_triples(v) for v in gdf["TRIPLE_p"]]
    plss = read(args.plss)
    idx = plss_index(plss)
    result = {
        "inputs": {"triples": args.triples, "deeds": len(gdf), "plss_features": len(plss)},
        "table3": dict(score_plss(list(zip(gdf.geometry, triples)), idx, township_polygons(idx), errors)),
    }
    t3 = result["table3"]
    if t3.get("with_triple"):
        for k in ("orig_1x1", "orig_6x6", "fixed_1x1_contains", "fixed_6x6_contains", "fixed_1x1_intersects", "fixed_6x6_intersects"):
            t3[k + "_pct_of_with_triple"] = round(100 * t3.get(k, 0) / t3["with_triple"], 2)
        t3["coverage_pct"] = round(100 * t3["with_triple"] / t3["deeds_total"], 2)
    rows = gdf.drop(columns="geometry").to_dict("records")
    result["table2"] = table2(rows)
    if args.subd:
        s = subdivision(rows, list(gdf.geometry), read(args.subd), errors)
        for k in ("deeds_with_>=1_plat_candidate", "deeds_candidate_intersects_deed"):
            if s.get("deeds_with_subdivision_prediction"):
                s[k + "_pct_of_with_prediction"] = round(100 * s.get(k, 0) / s["deeds_with_subdivision_prediction"], 2)
                s[k + "_pct_of_all_deeds"] = round(100 * s.get(k, 0) / len(gdf), 2)
        result["subdivision"] = dict(s)
    result["errors"] = dict(errors)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    tag = args.tag or f"table3_{args.triples}"
    out = Path(args.outdir) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
