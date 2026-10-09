"""
Load the five county plat layers (<PARTNER>'s shared shapefiles) for the subdivision matcher.

Returns plat records for src.matcher.PlatIndex: {county, name, aliases, geometry (EPSG:26915), sections}
where sections = set of (S, T, R) PLSS sections the plat overlaps by > 1 m^2 (merged_plss_5070, the paper's
PLSS layer). Also returns the (name, alias) pairs used to learn step-3 rules (Olmsted PLATNAME/LABEL,
Sherburne LEGAL_NAME/ALIAS). Prints nothing.
"""
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MP = REPO_ROOT.parent / "<DATA_ROOT>"
PLAT_ROOT = MP / "<PLAT_LAYER_DIR>"
PLSS = MP / "<RUN_DIR>" / "output" / "try08" / "merged_plss_5070.geojson"
CRS = "EPSG:26915"
# county key (lower case) -> (shapefile, primary name field, alias fields)
LAYERS = {
    "anoka": ("<PLAT_LAYER>/<PLAT_LAYER>.shp", "PLAT", ["SEARCH_NAM", "EXTRACT_NA"]),
    "dakota": ("<PLAT_LAYER>/<PLAT_LAYER>.shp", "PLATNAME", []),
    "milwaukee": ("<PLAT_LAYER>/Subdivision.shp", "Name", []),
    "olmsted": ("<PLAT_LAYER>/<PLAT_LAYER>.shp", "PLATNAME", ["LABEL"]),
    "sherburne": ("<PLAT_LAYER> 2/Plats.shp", "LEGAL_NAME", ["ALIAS"]),
}
RULE_PAIRS = {"olmsted": ("PLATNAME", "LABEL"), "sherburne": ("LEGAL_NAME", "ALIAS")}


def load_plats(root=PLAT_ROOT, plss_path=PLSS, crs=CRS):
    import geopandas as gpd
    plats, pairs = [], []
    for county, (rel, name_f, alias_f) in LAYERS.items():
        g = gpd.read_file(Path(root) / rel)
        if county in RULE_PAIRS:
            a, b = RULE_PAIRS[county]
            pairs += list(zip(g[a], g[b]))
        g = g.to_crs(crs)
        g["geometry"] = g.geometry.make_valid()
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        for _, r in g.iterrows():
            name = r[name_f]
            if not (isinstance(name, str) and len(name.strip()) > 1):
                continue
            plats.append({"county": county, "name": name, "geometry": r.geometry, "sections": set(),
                          "aliases": [r[f] for f in alias_f if isinstance(r[f], str) and r[f].strip()]})
    plss = gpd.read_file(plss_path).to_crs(crs)
    plss["geometry"] = plss.geometry.make_valid()
    pg = gpd.GeoDataFrame({"i": range(len(plats))}, geometry=[p["geometry"] for p in plats], crs=crs)
    j = gpd.sjoin(pg, plss[["SEC", "TWP", "RNG", "geometry"]], predicate="intersects")
    for i, s, t, rg, ri in zip(j["i"], j["SEC"], j["TWP"], j["RNG"], j["index_right"]):
        if plats[i]["geometry"].intersection(plss.geometry.loc[ri]).area > 1.0:
            plats[i]["sections"].add((int(s), int(t), int(rg)))
    return plats, pairs


def plat_sindex(plats):
    from shapely import STRtree
    return STRtree([p["geometry"] for p in plats])


def by_county(plats):
    out = defaultdict(list)
    for i, p in enumerate(plats):
        out[p["county"]].append(i)
    return out
