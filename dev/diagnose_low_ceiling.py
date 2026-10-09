"""
Why is the oracle ceiling low in Anoka / Dakota / Sherburne? Counts only, for with-span test deeds.

Per county (default: anoka, dakota, sherburne, plus olmsted and milwaukee as references):
  a. span-parsed (T, R, S) readings: all candidate combinations from the spans (up to 10 values per span),
     how many are valid in the deed's county, and deeds with >= 1 valid section / township reading;
  b. for deeds with a valid reading, where it lies relative to the gold section, for the rule_v2[county]
     top-1 reading and for the nearest valid reading: same_section / adjacent (touching) /
     same_township / other_township_le_10km / gt_10km; plus how often the reading's township number is
     from the other meridian's numbering than the gold's (T < 100 vs T >= 100);
  c. S / T / R spans containing plat-like keywords (addition, plat, subdivision, lot, block, outlot) vs PLSS
     keywords (township, range, meridian, quarter), per slot, per keyword, and deeds with any; plus S
     spans with a number above 36 (lot numbers?). Note: "lot" also matches "Government Lot" (a PLSS term).

Inputs: raw NER GeoJSON, gold CSV, valid-TRS JSON and study sections GPKG (dev/build_plss_table.py).
Privacy: prints and writes COUNTS ONLY; no text, numbers or ids. Errors report type and index only.

    mapprejudice/bin/python dev/diagnose_low_ceiling.py
"""

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.normalizer import spans_from_fields  # noqa: E402
from src.normalizer.candidates import read_spans  # noqa: E402
from src.normalizer.constraints import Constraints  # noqa: E402
from src.normalizer.rule_based import RuleBasedNormalizer  # noqa: E402
from src.normalizer.validate import build  # noqa: E402


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
    "sections": REPO_ROOT / "output" / "derived" / "plss_study_sections.gpkg",
    "summary": REPO_ROOT / "results" / "normalizer" / "diagnose_low_ceiling.json",
}
COUNTIES = ("anoka", "dakota", "sherburne", "olmsted", "milwaukee")
COUNTY_STATE = {"anoka": "MN", "dakota": "MN", "olmsted": "MN", "sherburne": "MN", "washington": "MN", "milwaukee": "WI"}
PLAT_KW = ("addition", "plat", "subdivision", "lot", "block", "outlot")
PLSS_KW = ("township", "range", "meridian", "quarter")
KW = {k: re.compile(rf"\b{k}s?\b", re.IGNORECASE) for k in PLAT_KW + PLSS_KW}
MAX_VALUES = 10
ADJACENT_M, NEAR_M = 1.0, 10_000.0


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and r["single"] == "True" and r["status"] == "ok":
                gold[(r["county"], r["deed_key"])] = (int(r["T"]), int(r["R"]), r["Rdir"], int(r["S"]))
    return gold


def section_geoms(path):
    """{(state, T, R, Rdir, S): geometry} (polygons sharing a key are unioned)."""
    import geopandas as gpd
    from shapely.ops import unary_union
    gdf = gpd.read_file(path)
    parts = defaultdict(list)
    for st, t, r, rd, s, g in zip(gdf["state"], gdf["T"], gdf["R"], gdf["Rdir"], gdf["S"], gdf.geometry):
        parts[(st, int(t), int(r), rd, int(s))].append(g)
    return {k: unary_union(v) if len(v) > 1 else v[0] for k, v in parts.items()}


def where(reading, gold, state, geoms):
    """Category of a valid (T, R, Rdir, S) reading relative to the gold section; None if geometry missing."""
    if reading == gold:
        return "same_section"
    a, b = geoms.get((state, *reading[:3], reading[3])), geoms.get((state, *gold))
    if a is None or b is None:
        return None
    dist = a.distance(b)
    if dist < ADJACENT_M:
        return "adjacent"
    if reading[:3] == gold[:3]:
        return "same_township"
    return "other_township_le_10km" if dist <= NEAR_M else "gt_10km"


def keyword_counts(spans):
    c = Counter()
    deed_plat = deed_plss = False
    for s in spans:
        plat = [k for k in PLAT_KW if KW[k].search(s.text)]
        plss = [k for k in PLSS_KW if KW[k].search(s.text)]
        c[f"{s.label}_spans"] += 1
        c[f"{s.label}_with_plat_kw"] += bool(plat)
        c[f"{s.label}_with_plss_kw"] += bool(plss)
        c[f"{s.label}_with_both"] += bool(plat and plss)
        c[f"{s.label}_with_neither"] += not plat and not plss
        for k in plat + plss:
            c[f"kw_{k}"] += 1
        deed_plat |= bool(plat)
        deed_plss |= bool(plss)
    c["deeds_with_plat_kw"] += deed_plat
    c["deeds_with_plss_kw"] += deed_plss
    return c


def diagnose(spans, county, gold, constraints, rule, geoms):
    state = COUNTY_STATE[county]
    scope = constraints.scope(county, state, "county")
    readings = read_spans(spans)
    cands, info = build(readings, scope, MAX_VALUES)
    c = Counter(deeds=1)
    c.update(keyword_counts(spans))
    c["S_spans_with_value_gt_36"] += sum(1 for r in readings if r.label == "S" and any(v.value > 36 for v in r.values))
    sections = {(t.trs.T, t.trs.R, t.trs.Rdir, t.trs.S) for t in cands if t.trs.S is not None}
    valid = {(t.trs.T, t.trs.R, t.trs.Rdir, t.trs.S) for t in cands if t.valid == "section"}
    townships = {(t.trs.T, t.trs.R, t.trs.Rdir) for t in cands if t.valid == "township"}
    c["deeds_with_T_and_R_candidates"] += bool(info.get("pool_T") and info.get("pool_R"))
    c["readings_in_range"] += len(sections)
    c["readings_valid_in_county"] += len(valid)
    c["deeds_with_valid_section_reading"] += bool(valid)
    c["deeds_with_valid_township_reading"] += bool(townships)
    c[f"valid_readings_per_deed_{min(len(valid), 3)}"] += 1
    c["deeds_gold_township_among_valid_townships"] += (gold[0], gold[1], gold[2]) in townships
    c["gold_T_lt_100"] += gold[0] < 100
    if valid:
        res = rule.normalize(spans, county, state)
        top = res.top
        if top is not None and top.S is not None:
            cat = where((top.T, top.R, top.Rdir, top.S), gold, state, geoms)
            c[f"top1_{cat or 'no_geometry'}"] += 1
            c["top1_other_meridian_numbering"] += (top.T < 100) != (gold[0] < 100)
        cats = [where(v, gold, state, geoms) for v in valid]
        order = ("same_section", "adjacent", "same_township", "other_township_le_10km", "gt_10km")
        best = next((o for o in order if o in cats), "no_geometry")
        c[f"nearest_{best}"] += 1
        c["valid_readings_other_meridian_numbering"] += sum((v[0] < 100) != (gold[0] < 100) for v in valid)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--raw", default=str(rt3.DEFAULTS["raw"]))
    ap.add_argument("--gold", default=str(DEFAULTS["gold"]))
    ap.add_argument("--valid", default=str(DEFAULTS["valid"]))
    ap.add_argument("--sections", default=str(DEFAULTS["sections"]))
    ap.add_argument("--counties", default=",".join(COUNTIES))
    ap.add_argument("--summary", default=str(DEFAULTS["summary"]))
    args = ap.parse_args(argv)

    counties = tuple(args.counties.split(","))
    gold = load_gold(args.gold)
    constraints = Constraints.from_json(args.valid)
    rule = RuleBasedNormalizer(constraints, "county")
    geoms = section_geoms(args.sections)
    raw = rt3.read(args.raw)
    per, errors, seen = defaultdict(Counter), Counter(), set()
    for i, props in enumerate(raw.drop(columns="geometry").to_dict("records")):
        try:
            ids = cso.parse_image_ids(props.get("image_ids"))
            key = (cso.county_key(props.get("cnty_name")), "|".join(sorted(set(ids)))) if ids else None
            if key not in gold or key in seen or key[0] not in counties:
                continue
            seen.add(key)
            spans = spans_from_fields(props)
            if spans:
                per[key[0]].update(diagnose(spans, key[0], gold[key], constraints, rule, geoms))
        except Exception as e:  # type and index only
            errors[f"row#{i}:{type(e).__name__}"] += 1
    result = {"inputs": {"counties": list(counties), "max_values_per_span": MAX_VALUES,
                         "adjacent_m": ADJACENT_M, "near_m": NEAR_M},
              "counties": {c: dict(sorted(per[c].items())) for c in counties},
              "errors": dict(Counter(k.split(":")[-1] for k in errors.elements()))}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
