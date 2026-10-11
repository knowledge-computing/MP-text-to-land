"""
Load the county plat layers (partner plat layers) for the subdivision matcher.

Layer locations, name fields and alias fields come from the local JSON named by MP_PLAT_CONFIG; the PLSS
layer is the paper's merged_plss_5070.geojson under MP_RUN_DIR (dev/private_paths.py). Nothing about the
private folder layout is written here.

Returns plat records for src.matcher.PlatIndex: {county, name, aliases, geometry (EPSG:26915), sections}
where sections = set of (S, T, R) PLSS sections the plat overlaps by > 1 m^2. Also returns the (name, alias)
pairs used to learn step-3 rules (the config's rule_pairs). Prints nothing.
"""
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import private_paths as pp  # noqa: E402

CRS = "EPSG:26915"
PLSS = pp.run_dir("output", "try08", "merged_plss_5070.geojson")


def layer_config(cfg=None):
    """-> (root Path, {county: (relative shapefile, name field, alias fields)}, {county: (name, alias)})."""
    cfg = cfg or pp.plat_config()
    layers = {c.lower(): (v["path"], v["name_field"], list(v.get("alias_fields") or [])) for c, v in cfg["layers"].items()}
    pairs = {c.lower(): tuple(v) for c, v in cfg["rule_pairs"].items()}
    return Path(cfg["root"]), layers, pairs


def load_plats(cfg=None, plss_path=PLSS, crs=CRS):
    import geopandas as gpd
    root, layers, rule_pairs = layer_config(cfg)
    plats, pairs = [], []
    for county, (rel, name_f, alias_f) in layers.items():
        g = gpd.read_file(root / rel)
        if county in rule_pairs:
            a, b = rule_pairs[county]
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
