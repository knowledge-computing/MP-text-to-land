"""
Derive gold (T, Tdir, R, Rdir, S) for every deed row from its volunteer polygon (Phase 0a).

For each feature of the train / val / test split GeoJSONs, the volunteer polygon is overlaid with the
study-area PLSS sections (output/derived/plss_study_sections.gpkg from dev/build_plss_table.py) of the
deed's state. Gold = the (T, Tdir, R, Rdir, S) key with the largest overlap area (polygons sharing a key
are summed); the gold township (6x6) is chosen the same way over township keys. Each row is tagged with
its split and whether it is a single-polygon deed (one polygon part, and its (county, image_ids) deed
key occurs once in that split): only those are used for scoring.

Deed-level gold sets (--deed-splits, default train,val; labels for a future ranker / ByT5): for every
deed key, the union over its rows and polygon parts of each part's max-overlap section (and township).
Train/val labels use set membership; the test split keeps the single-polygon rows above.

Privacy: reads only image_ids, cnty_name and geometry. Prints and writes to results/ COUNTS ONLY.
The per-row CSV (deed keys + gold numbers) goes to the git-ignored output/derived/ and is never printed.
Errors inside data loops report the exception type and a numeric index only.

    mapprejudice/bin/python dev/derive_gold_trs.py
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_split_overlap", REPO_ROOT / "dev" / "check_split_overlap.py")
cso = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cso)

DEFAULTS = {
    "sections": REPO_ROOT / "output" / "derived" / "plss_study_sections.gpkg",
    "out": REPO_ROOT / "output" / "derived" / "gold_trs_622.csv",
    "deed_out": REPO_ROOT / "output" / "derived" / "gold_trs_deedsets_622.csv",
    "summary": REPO_ROOT / "results" / "normalizer" / "phase0_gold_summary.json",
}
CRS = "EPSG:5070"
COUNTY_STATE = {"anoka": "MN", "dakota": "MN", "olmsted": "MN", "sherburne": "MN", "washington": "MN", "milwaukee": "WI"}
MIN_KEY_SHARE = 0.01  # a key "touches" the deed when it covers >= 1% of the deed area
RATIO_BINS = ((0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0001))
COLUMNS = ["split", "row", "county", "state", "deed_key", "n_parts", "rows_per_deed", "single", "status",
           "T", "Tdir", "R", "Rdir", "S", "ratio", "second_ratio", "n_keys_ge_1pct",
           "twp_T", "twp_Tdir", "twp_R", "twp_Rdir", "twp_ratio", "in_county"]


def read(path, columns=None):
    import geopandas as gpd
    gdf = gpd.read_file(path, columns=columns)
    return gdf.to_crs(CRS) if gdf.crs and str(gdf.crs) != CRS else gdf


def ratio_bin(r):
    for lo, hi in RATIO_BINS:
        if lo <= r < hi:
            return f"[{lo},{min(hi, 1.0)}{']' if hi > 1 else ')'}"
    return "other"


def gold_for(geom, sections, sindex):
    """Return dict with gold section/township keys and overlap stats; None if no section intersects."""
    import shapely
    idx = sindex.query(geom, predicate="intersects")
    if len(idx) == 0:
        return None
    sub = sections.iloc[idx]
    areas = shapely.area(shapely.intersection(sub.geometry.values, geom))
    sec, twp, cnty = defaultdict(float), defaultdict(float), defaultdict(set)
    for t, td, r, rd, s, cs, a in zip(sub["T"], sub["Tdir"], sub["R"], sub["Rdir"], sub["S"], sub["counties"], areas):
        key = (int(t), td, int(r), rd, int(s))
        sec[key] += a
        twp[key[:4]] += a
        cnty[key].update(cs.split(";"))
    total = geom.area
    if total <= 0 or max(sec.values()) <= 0:
        return None
    ranked = sorted(sec.items(), key=lambda kv: -kv[1])
    best, best_a = ranked[0]
    tbest, tbest_a = max(twp.items(), key=lambda kv: kv[1])
    return {
        "key": best, "ratio": float(best_a / total),
        "second_ratio": float(ranked[1][1] / total) if len(ranked) > 1 else 0.0,
        "n_keys": int(sum(a / total >= MIN_KEY_SHARE for _, a in ranked)),
        "twp": tbest, "twp_ratio": float(tbest_a / total), "counties": cnty[best],
    }


DEED_COLUMNS = ["split", "county", "state", "deed_key", "n_rows", "n_parts", "parts_matched", "status",
                "gold_set", "twp_set"]


def part_keys(geom, sections, sindex):
    """[(section key, township key) or None] for each polygon part."""
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for p in parts:
        g = gold_for(p, sections, sindex) if not p.is_empty and p.area > 0 else None
        out.append((g["key"], g["twp"]) if g else None)
    return out


def derive(splits, sections, errors, deed_splits=()):
    """One output row dict per split feature. Rows of deed_splits also get '_parts' (per-part gold keys)."""
    import shapely
    by_state = {st: sections[sections["state"] == st].reset_index(drop=True) for st in sections["state"].unique()}
    sindex = {st: g.sindex for st, g in by_state.items()}
    rows = []
    for split, path in splits.items():
        gdf = read(path, ["image_ids", "cnty_name"])
        keys = []
        for i, (ids_value, cname) in enumerate(zip(gdf["image_ids"], gdf["cnty_name"])):
            ids = cso.parse_image_ids(ids_value)
            keys.append((cso.county_key(cname), "|".join(sorted(set(ids)))) if ids else None)
        mult = Counter(k for k in keys if k)
        for i, (geom, key) in enumerate(zip(gdf.geometry, keys)):
            row = dict.fromkeys(COLUMNS, "")
            row.update(split=split, row=i)
            try:
                county = key[0] if key else cso.county_key(gdf["cnty_name"].iloc[i])
                state = COUNTY_STATE.get(county, "")
                row.update(county=county, state=state, deed_key=key[1] if key else "",
                           rows_per_deed=mult[key] if key else 0)
                if geom is None or geom.is_empty:
                    row.update(status="empty_geom", single=False)
                    rows.append(row)
                    continue
                if not geom.is_valid:
                    geom = shapely.make_valid(geom)
                    errors["repaired_invalid_geometry"] += 1
                n_parts = len(geom.geoms) if hasattr(geom, "geoms") else 1
                single = n_parts == 1 and row["rows_per_deed"] == 1
                row.update(n_parts=n_parts, single=single)
                if state not in by_state:
                    row["status"] = "no_state"
                    rows.append(row)
                    continue
                g = gold_for(geom, by_state[state], sindex[state])
                if g is None:
                    row["status"] = "no_match"
                else:
                    t, td, r, rd, s = g["key"]
                    row.update(status="ok", T=t, Tdir=td, R=r, Rdir=rd, S=s, ratio=round(g["ratio"], 4),
                               second_ratio=round(g["second_ratio"], 4), n_keys_ge_1pct=g["n_keys"],
                               twp_T=g["twp"][0], twp_Tdir=g["twp"][1], twp_R=g["twp"][2], twp_Rdir=g["twp"][3],
                               twp_ratio=round(g["twp_ratio"], 4), in_county=county in g["counties"])
                if split in deed_splits and key:
                    row["_parts"] = part_keys(geom, by_state[state], sindex[state])
            except Exception as e:  # type and index only
                errors[f"{split}#{i}:{type(e).__name__}"] += 1
                row["status"] = "error"
            rows.append(row)
    return rows


def deed_sets(rows, deed_splits):
    """Aggregate per-part gold keys into one gold set per (split, deed key)."""
    groups = defaultdict(list)
    for r in rows:
        if r["split"] in deed_splits and r["deed_key"]:
            groups[(r["split"], r["county"], r["deed_key"])].append(r)
    deeds = []
    for (split, county, key), rs in groups.items():
        parts = [p for r in rs for p in r.get("_parts", [])]
        matched = [p for p in parts if p]
        secs = sorted({p[0] for p in matched})
        twps = sorted({p[1] for p in matched})
        status = "no_geometry" if not parts else "ok" if len(matched) == len(parts) else "partial" if matched else "no_match"
        deeds.append({"split": split, "county": county, "state": rs[0]["state"], "deed_key": key, "n_rows": len(rs),
                      "n_parts": len(parts), "parts_matched": len(matched), "status": status,
                      "gold_set": json.dumps([list(k) for k in secs]), "twp_set": json.dumps([list(k) for k in twps]),
                      "_n": len(secs), "_ntwp": len(twps)})
    return deeds


def summarize_deeds(deeds):
    out = {}
    for split in dict.fromkeys(d["split"] for d in deeds):
        ds = [d for d in deeds if d["split"] == split]
        with_gold = [d for d in ds if d["_n"]]
        out[split] = {
            "deeds": len(ds),
            "deeds_with_gold_set": len(with_gold),
            "status": dict(Counter(d["status"] for d in ds)),
            "multi_row_deeds": sum(d["n_rows"] > 1 for d in ds),
            "gold_set_size": dict(sorted(Counter(str(min(d["_n"], 3)) for d in with_gold).items())),
            "township_set_size": dict(sorted(Counter(str(min(d["_ntwp"], 3)) for d in with_gold).items())),
            "deeds_with_gold_by_county": dict(sorted(Counter(d["county"] for d in with_gold).items())),
        }
    keys = defaultdict(set)
    for d in deeds:
        keys[d["split"]].add((d["county"], d["deed_key"]))
    names = sorted(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}~{b}_identical_deed_keys"] = len(keys[a] & keys[b])
    return out  # set-size key "3" means "3 or more"


def summarize(rows):
    out = {"splits": {}, "single_ok": {}}
    for split in dict.fromkeys(r["split"] for r in rows):
        rs = [r for r in rows if r["split"] == split]
        single = [r for r in rs if r["single"] is True]
        ok = [r for r in single if r["status"] == "ok"]
        out["splits"][split] = {
            "rows": len(rs),
            "single_polygon_rows": len(single),
            "multi_part_rows": sum(1 for r in rs if isinstance(r["n_parts"], int) and r["n_parts"] > 1),
            "rows_of_multi_row_deeds": sum(1 for r in rs if isinstance(r["rows_per_deed"], int) and r["rows_per_deed"] > 1),
            "status_all_rows": dict(Counter(r["status"] for r in rs)),
            "status_single_rows": dict(Counter(r["status"] for r in single)),
            "single_rows_by_county": dict(sorted(Counter(r["county"] for r in single).items())),
        }
        out["single_ok"][split] = {
            "rows": len(ok),
            "section_overlap_ratio_hist": dict(sorted(Counter(ratio_bin(r["ratio"]) for r in ok).items())),
            "township_overlap_ratio_hist": dict(sorted(Counter(ratio_bin(r["twp_ratio"]) for r in ok).items())),
            "sections_touched_ge_1pct": dict(sorted(Counter(min(r["n_keys_ge_1pct"], 3) for r in ok).items())),
            "straddling_rows_gt1_section": sum(1 for r in ok if r["n_keys_ge_1pct"] > 1),
            "S_outside_1_36": sum(1 for r in ok if not 1 <= r["S"] <= 36),
            "Rdir": dict(Counter(r["Rdir"] or "unknown" for r in ok)),
            "Tdir": dict(Counter(r["Tdir"] or "unknown" for r in ok)),
            "gold_section_not_assigned_to_deed_county": sum(1 for r in ok if not r["in_county"]),
            "section_key_differs_from_township_key": sum(
                1 for r in ok if (r["T"], r["Tdir"], r["R"], r["Rdir"]) != (r["twp_T"], r["twp_Tdir"], r["twp_R"], r["twp_Rdir"])),
        }
    return out  # in sections_touched_ge_1pct, key 3 means "3 or more"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--split", action="append", default=[], metavar="NAME=PATH",
                    help="split name and GeoJSON path; repeatable (default: July 2025 6_2_2 files)")
    ap.add_argument("--sections", default=str(DEFAULTS["sections"]))
    ap.add_argument("--out", default=str(DEFAULTS["out"]), help="per-row CSV (git-ignored location)")
    ap.add_argument("--summary", default=str(DEFAULTS["summary"]))
    ap.add_argument("--deed-splits", default="train,val", help="splits that get deed-level gold sets ('' = none)")
    ap.add_argument("--deed-out", default=str(DEFAULTS["deed_out"]), help="deed-level CSV (git-ignored location)")
    args = ap.parse_args(argv)
    deed_splits = tuple(filter(None, args.deed_splits.split(",")))
    splits = dict(s.split("=", 1) for s in args.split) if args.split else {k: str(v) for k, v in cso.DEFAULT_SPLITS.items()}
    missing = [n for n, p in splits.items() if not Path(p).exists()]
    if missing or not Path(args.sections).exists():
        print(f"missing inputs: {', '.join(missing) or 'sections'}", file=sys.stderr)
        return 2

    errors = Counter()
    sections = read(args.sections)
    rows = derive(splits, sections, errors, deed_splits)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    result = summarize(rows)
    if deed_splits:
        deeds = deed_sets(rows, deed_splits)
        with open(args.deed_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DEED_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(deeds)
        result["deed_sets"] = summarize_deeds(deeds)
    result["inputs"] = {"splits": list(splits), "section_rows": len(sections), "min_key_share": MIN_KEY_SHARE}
    result["errors"] = dict(Counter(k.split(":")[-1] if "#" in k else k for k in errors.elements()))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out.name} ({len(rows)} rows, not printed) and {Path(args.summary).name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
