"""
S4: geometric weak labels for the subdivision matcher.

Weak label of a deed = the plat in the deed's own county layer with the largest overlap with the
volunteer deed polygon (EPSG:26915). It is NOT independent ground truth: deed polygons were drawn by
volunteers who read the same addition name. Deeds in counties without a plat layer (Washington) and
multi-part deed geometries (excluded from test by design) are not eligible.

Library use: weak_labels(deeds, plats) -> list of dicts. CLI prints counts and distributions only:

    python dev/weak_labels.py --pred <val predictions geojson> [--summary results/matcher/val_weak_labels.json]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import plat_layers as pl  # noqa: E402


def load_deeds(path, crs=pl.CRS):
    """Prediction GeoJSON from dev/infer_try08.py -> list of {county, props, geometry (crs), n_parts}."""
    import geopandas as gpd
    if "test" in Path(path).name.lower():
        sys.exit("refused: the test split is frozen (CLAUDE.md, Test-set freeze)")
    g = gpd.read_file(path).to_crs(crs)
    g["geometry"] = g.geometry.make_valid()
    rows = g.drop(columns="geometry").to_dict("records")
    return [{"county": str(r.get("cnty_name") or "").strip().lower(), "props": r, "geometry": geom,
             "n_parts": int(r.get("n_parts") or 0)} for r, geom in zip(rows, g.geometry)]


def eligibility(d, counties):
    if d["county"] not in counties:
        return "no_plat_layer"
    if d["geometry"] is None or d["geometry"].is_empty or d["geometry"].area <= 0:
        return "no_geometry"
    if d["n_parts"] > 1:
        return "multi_part"
    return "eligible"


def weak_labels(deeds, plats, errors=None):
    tree, errors = pl.plat_sindex(plats), errors if errors is not None else Counter()
    counties = {p["county"] for p in plats}
    out = []
    for i, d in enumerate(deeds):
        rec = {"status": eligibility(d, counties), "plat": None, "max_frac": 0.0, "cover_frac": 0.0, "n_overlap": 0}
        try:
            if rec["status"] == "eligible":
                g = d["geometry"]
                cand = [k for k in tree.query(g, predicate="intersects") if plats[k]["county"] == d["county"]]
                areas = {k: plats[k]["geometry"].intersection(g).area for k in cand}
                areas = {k: a for k, a in areas.items() if a > 0}
                if areas:
                    from shapely.ops import unary_union
                    best = max(areas, key=lambda k: (areas[k], -k))
                    rec.update(plat=best, max_frac=min(areas[best] / g.area, 1.0), n_overlap=len(areas),
                               cover_frac=min(unary_union([plats[k]["geometry"] for k in areas]).intersection(g).area / g.area, 1.0))
        except Exception as e:  # noqa: BLE001
            errors[f"weak_label:{type(e).__name__}"] += 1
        out.append(rec)
    return out


def dist(values, edges):
    import numpy as np
    a = np.asarray(values, float)
    if not len(a):
        return {"n": 0}
    q = np.quantile(a, [0, .1, .25, .5, .75, .9, 1])
    bins = Counter()
    for x in a:
        bins[next((lab for lab, hi in edges if x < hi), edges[-1][0])] += 1
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4), "min,p10,p25,p50,p75,p90,max": [round(float(v), 4) for v in q],
            "bins": {lab: bins.get(lab, 0) for lab, _ in edges}}


FRAC_BINS = [("=0", 1e-12), ("(0,.25)", .25), ("[.25,.5)", .5), ("[.5,.9)", .9), ("[.9,.99)", .99), (">=.99", float("inf"))]


def summarize(deeds, labels):
    by = {}
    for county in sorted({d["county"] for d in deeds}) + ["ALL"]:
        idx = [i for i, d in enumerate(deeds) if county in ("ALL", d["county"])]
        st = Counter(labels[i]["status"] for i in idx)
        el = [i for i in idx if labels[i]["status"] == "eligible"]
        by[county] = {"deeds": len(idx), "status": dict(st),
                      "with_weak_label": sum(labels[i]["plat"] is not None for i in el),
                      "overlap_>=2_plats": sum(labels[i]["n_overlap"] >= 2 for i in el),
                      "max_frac": dist([labels[i]["max_frac"] for i in el], FRAC_BINS),
                      "cover_frac": dist([labels[i]["cover_frac"] for i in el], FRAC_BINS)}
    return by


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--summary")
    args = ap.parse_args(argv)
    errors = Counter()
    plats, _ = pl.load_plats()
    deeds = load_deeds(args.pred)
    res = {"by_county": summarize(deeds, weak_labels(deeds, plats, errors)), "errors": dict(errors)}
    text = json.dumps(res, indent=1)
    print(text)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text + "\n")


if __name__ == "__main__":
    main()
