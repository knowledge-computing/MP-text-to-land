"""
Build the study-area PLSS section table from public layers (Phase 0, normalizer plan).

Inputs (public data, no deed data):
  * MN_PLSS.shp (COUN, TOWN, RANG, RDIR, SECT, TRS_DESCRI) and WI_PLSS.shp (TWP, RNG, SEC, DIR_ALPHA)
    from the 2025 run folder's PLSS/ directory ($MP_RUN_DIR/PLSS);
  * Census cartographic-boundary counties (cb_2024_us_county_500k.zip: STATEFP, NAME).

A section belongs to a study county when (section ∩ county) / section area >= --min-frac, so a section
on a county line can belong to two counties. Township direction is "N" for every row (neither layer has
a township-direction field; every study county lies north of its baseline). MN range direction comes
from the RDIR code via --mn-rdir (default '0=W': every MN study section has code 0 and every MN study
county is Range West of the 4th or 5th P.M.; codes 1/2 occur only outside the study area and stay
unmapped). `--mn-rdir infer` instead reads the letter after the range number in TRS_DESCRI.

Townships in Dakota, Washington and Sherburne are numbered from two meridians (e.g. T26-28 and
T112-115), so T/R min-max is not a usable constraint: constraints are the exact (T,R,S) sets per county.
Sections numbered outside 1-36 stay in the .gpkg (for gold) but are excluded from the valid sets.

Outputs (git-ignored, under output/derived/):
  plss_study_sections.gpkg   one row per section polygon: state, counties, primary_county,
                             primary_frac, T, Tdir, R, Rdir, S, src_row, geometry (EPSG:5070)
  plss_valid_trs.json        {"meta": ..., "valid": {state: {county: [[T, Tdir, R, Rdir, S], ...]}},
                              "valid_state": {state: [[T, Tdir, R, Rdir, S], ...]}}  (whole state layer,
                              for the no-county ablation; Rdir "" where the code is unmapped)
and an aggregate summary (counts only) printed and written to results/normalizer/.

    mapprejudice/bin/python dev/build_plss_table.py
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import private_paths as pp  # noqa: E402

PLSS_DIR = pp.run_dir("PLSS")  # MP_RUN_DIR, resolved only when a default is used
DEFAULTS = {
    "mn": PLSS_DIR / "MN_PLSS.shp",
    "wi": PLSS_DIR / "WI_PLSS.shp",
    "counties": REPO_ROOT.parent / "public_layers" / "census_cb2024" / "cb_2024_us_county_500k.zip",
    "out_dir": REPO_ROOT / "output" / "derived",
    "summary": REPO_ROOT / "results" / "normalizer" / "phase0_plss_table.json",
}
CRS = "EPSG:5070"
STATEFP = {"27": "MN", "55": "WI"}
STUDY = {"MN": ("anoka", "dakota", "olmsted", "sherburne", "washington"), "WI": ("milwaukee",)}
RANGE_LETTER = re.compile(r"\bR(?:ange|ng|ge)?\.?\s*0*(\d{1,3})\s*([EW])(?:est|ast)?\b", re.IGNORECASE)


def read(path, columns=None):
    import geopandas as gpd
    p = str(path)
    if p.endswith(".zip") and not p.startswith("zip://"):
        p = "zip://" + p
    gdf = gpd.read_file(p, columns=columns)
    return gdf.to_crs(CRS) if gdf.crs and str(gdf.crs) != CRS else gdf


def load_counties(path):
    gdf = read(path, ["STATEFP", "NAME"])
    gdf["state"] = gdf["STATEFP"].map(STATEFP)
    gdf["county"] = gdf["NAME"].str.lower()
    keep = [(s in STUDY and c in STUDY[s]) for s, c in zip(gdf["state"], gdf["county"])]
    return gdf[keep][["state", "county", "geometry"]].reset_index(drop=True)


def parse_rdir_map(text):
    """'1=W,2=E,0=' -> {1: 'W', 2: 'E', 0: ''}"""
    out = {}
    for part in filter(None, (text or "").split(",")):
        code, _, letter = part.partition("=")
        out[int(code)] = letter.strip().upper()
    return out


def infer_rdir_map(codes, descr, ranges):
    """Majority letter per RDIR code, from the 'R<number><E|W>' text in TRS_DESCRI. Counts only."""
    tab = defaultdict(Counter)
    for code, text, rng in zip(codes, descr, ranges):
        m = RANGE_LETTER.search(text) if isinstance(text, str) else None
        if not m:
            tab[int(code)]["no_letter_found"] += 1
            continue
        agrees = int(m.group(1)) == int(rng)
        tab[int(code)][m.group(2).upper() + ("" if agrees else "_number_mismatch")] += 1
    mapping = {}
    for code, c in tab.items():
        letters = {k: v for k, v in c.items() if k in ("E", "W")}
        total = sum(letters.values())
        best = max(letters, key=letters.get) if letters else ""
        mapping[code] = best if total and letters[best] / total >= 0.95 else ""
    return mapping, {str(k): dict(v) for k, v in sorted(tab.items())}


def load_mn(path, rdir_map_text, summary):
    gdf = read(path, ["COUN", "TOWN", "RANG", "RDIR", "SECT", "TRS_DESCRI"])
    if rdir_map_text and rdir_map_text != "infer":
        mapping, source = parse_rdir_map(rdir_map_text), "given (--mn-rdir)"
    else:
        mapping, crosstab = infer_rdir_map(gdf["RDIR"], gdf["TRS_DESCRI"], gdf["RANG"])
        summary["mn_rdir_crosstab_statewide"] = crosstab
        source = "inferred from TRS_DESCRI"
    summary["mn_rdir_map"] = {"source": source, "map": {str(k): v for k, v in sorted(mapping.items())}}
    out = gdf.rename(columns={"TOWN": "T", "RANG": "R", "SECT": "S"})
    out["Rdir"] = [mapping.get(int(c), "") for c in gdf["RDIR"]]
    out["RDIR_code"] = gdf["RDIR"].astype(int)
    out["state"] = "MN"
    out["src_row"] = range(len(out))
    return out


def load_wi(path):
    gdf = read(path, ["TWP", "RNG", "SEC", "DIR_ALPHA"])
    out = gdf.rename(columns={"TWP": "T", "RNG": "R", "SEC": "S"})
    out["Rdir"] = [d.strip().upper() if isinstance(d, str) and d.strip().upper() in ("E", "W") else "" for d in gdf["DIR_ALPHA"]]
    out["state"] = "WI"
    out["src_row"] = range(len(out))
    return out


def assign_counties(sections, counties, min_frac):
    """Return {section index: {county: frac}} for fracs >= min_frac, plus sliver count."""
    import shapely
    state_c = counties[counties["state"] == sections["state"].iloc[0]].reset_index(drop=True)
    sec_idx, cty_idx = state_c.sindex.query(sections.geometry, predicate="intersects")
    sec_geoms = sections.geometry.values[sec_idx]
    inter = shapely.area(shapely.intersection(sec_geoms, state_c.geometry.values[cty_idx]))
    area = shapely.area(sec_geoms)
    assigned, slivers = defaultdict(dict), 0
    for s, c, a, total in zip(sec_idx, cty_idx, inter, area):
        frac = a / total if total else 0.0
        if frac >= min_frac:
            assigned[int(s)][state_c["county"].iloc[c]] = round(float(frac), 4)
        elif frac > 0:
            slivers += 1
    return assigned, slivers


def build(mn, wi, counties, min_frac, summary):
    """Concatenate study-area sections from both states; fill summary with counts."""
    import geopandas as gpd
    import pandas as pd
    parts = []
    for sections in (mn, wi):
        state = sections["state"].iloc[0]
        assigned, slivers = assign_counties(sections, counties, min_frac)
        keep = sorted(assigned)
        sub = sections.iloc[keep].copy()
        fr = [assigned[i] for i in keep]
        sub["counties"] = [";".join(sorted(d)) for d in fr]
        sub["primary_county"] = [max(d, key=d.get) for d in fr]
        sub["primary_frac"] = [max(d.values()) for d in fr]
        multi = Counter(";".join(sorted(d)) for d in fr if len(d) > 1)
        summary["states"][state] = {
            "source_rows": len(sections), "rows_in_study_counties": len(sub),
            "sections_in_more_than_one_county": sum(multi.values()),
            "multi_county_pairs": dict(sorted(multi.items())),
            "sliver_overlaps_below_min_frac": slivers,
        }
        if state == "MN":
            agree = Counter()
            for county, code, text in zip(sub["primary_county"], sub["COUN"], sub["TRS_DESCRI"]):
                agree[(county, int(code))] += 1
                summary["mn_trs_descri_names_county"][county] = summary["mn_trs_descri_names_county"].get(county, 0) + (
                    county in text.lower() if isinstance(text, str) else 0)
            summary["mn_primary_county_vs_COUN_code"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(agree.items())}
            summary["mn_rdir_code_counts_in_study"] = dict(Counter(map(str, sub["RDIR_code"])))
        parts.append(sub[["state", "counties", "primary_county", "primary_frac", "T", "R", "Rdir", "S", "src_row", "geometry"]])
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    out.insert(out.columns.get_loc("R"), "Tdir", "N")
    for c in ("T", "R", "S", "src_row"):
        out[c] = out[c].astype(int)
    return out


def valid_table(sections):
    valid = defaultdict(lambda: defaultdict(set))
    for st, cs, t, td, r, rd, s in zip(sections["state"], sections["counties"], sections["T"], sections["Tdir"],
                                      sections["R"], sections["Rdir"], sections["S"]):
        if not 1 <= int(s) <= 36:
            continue
        for c in cs.split(";"):
            valid[st][c].add((int(t), td, int(r), rd, int(s)))
    return {st: {c: sorted(map(list, keys)) for c, keys in sorted(cs.items())} for st, cs in sorted(valid.items())}


def state_table(frames):
    """Whole-state valid keys (S in 1-36) for the no-county ablation."""
    out = {}
    for f in frames:
        keys = {(int(t), "N", int(r), rd, int(s)) for t, r, rd, s in zip(f["T"], f["R"], f["Rdir"], f["S"]) if 1 <= int(s) <= 36}
        out[f["state"].iloc[0]] = sorted(map(list, keys))
    return out


def county_stats(valid, sections):
    out = {}
    for st, counties in valid.items():
        for c, rows in counties.items():
            sub = sections[(sections["state"] == st) & sections["counties"].str.split(";").apply(lambda x: c in x)]
            keys = Counter(zip(sub["T"], sub["Tdir"], sub["R"], sub["Rdir"], sub["S"]))
            townships = {(r[0], r[1], r[2], r[3]) for r in rows}
            out[f"{st}:{c}"] = {
                "section_rows": len(sub), "unique_trs_valid": len(rows),
                "trs_with_more_than_one_polygon": sum(v > 1 for v in keys.values()),
                "Rdir_counts": dict(Counter(r[3] or "unknown" for r in rows)),
                "S_outside_1_36_values_excluded": sorted(int(s) for s in sub["S"] if not 1 <= int(s) <= 36),
                "townships": len(townships),
                "townships_T_lt_100": sum(t[0] < 100 for t in townships),
                "townships_T_ge_100": sum(t[0] >= 100 for t in townships),
            }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--mn", default=DEFAULTS["mn"])
    ap.add_argument("--wi", default=DEFAULTS["wi"])
    ap.add_argument("--counties", default=str(DEFAULTS["counties"]))
    ap.add_argument("--min-frac", type=float, default=0.01, help="min share of a section's area inside a county")
    ap.add_argument("--mn-rdir", default="0=W", help="RDIR code mapping, e.g. '0=W' (default) or 'infer'")
    ap.add_argument("--out-dir", default=str(DEFAULTS["out_dir"]))
    ap.add_argument("--summary", default=str(DEFAULTS["summary"]))
    args = ap.parse_args(argv)

    summary = {"inputs": {"min_frac": args.min_frac}, "states": {}, "mn_trs_descri_names_county": {}}
    counties = load_counties(args.counties)
    summary["inputs"]["study_counties_found"] = sorted(f"{s}:{c}" for s, c in zip(counties["state"], counties["county"]))
    mn = load_mn(args.mn, args.mn_rdir, summary)
    wi = load_wi(args.wi)
    summary["wi_dir_alpha_counts_statewide"] = dict(Counter(wi["Rdir"].replace("", "unknown")))
    sections = build(mn, wi, counties, args.min_frac, summary)
    valid = valid_table(sections)
    summary["counties"] = county_stats(valid, sections)
    valid_state = state_table([mn, wi])
    summary["valid_state_keys"] = {st: len(v) for st, v in valid_state.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections.to_file(out_dir / "plss_study_sections.gpkg", layer="sections", driver="GPKG")
    meta = {"crs": CRS, "min_frac": args.min_frac, "Tdir": "N for every row (no field in either layer)",
            "mn_rdir_map": summary["mn_rdir_map"], "sources": ["MN_PLSS.shp", "WI_PLSS.shp", Path(args.counties).name]}
    (out_dir / "plss_valid_trs.json").write_text(json.dumps({"meta": meta, "valid": valid, "valid_state": valid_state}) + "\n", encoding="utf-8")

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out_dir.name}/plss_study_sections.gpkg, plss_valid_trs.json, {Path(args.summary).name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
