"""Gazetteer source records and assembly (steps 4-6). Nothing here prints data.

A record is a plain dict, so a gazetteer variant (a subset of sources) can be rebuilt in memory:
  {"type": "name", county, name, aliases, source, contributor, n_parcels, blocks, lots, pairs}
  {"type": "cov",  county, add_cov, add_mod, source, contributor, block, lots}
Sources: partner (partner plat layers), mngeo (MnGeo plan_parcels_open plat_name), public (mp-us-racial-
covenants rows minus val/test deeds), train (our train split minus deeds duplicated in val/test).
"""
import json
from collections import defaultdict
from pathlib import Path

from .blocklot import parse_block, parse_lots
from .data import clean, county_key, norm_doc
from .gazetteer import LAYER, Gazetteer


def partner_records(root, layers):
    """layers: {county: (relative shapefile, primary name field, [alias fields])} (dev/plat_layers.layer_config(), from MP_PLAT_CONFIG)."""
    import pyogrio
    out = []
    for county, (rel, name_f, alias_f) in layers.items():
        df = pyogrio.read_dataframe(Path(root) / rel, columns=[name_f] + list(alias_f), read_geometry=False)
        for r in df.to_dict("records"):
            name = clean(r.get(name_f))
            if not name or len(name) <= 1:
                continue
            out.append({"type": "name", "county": county, "name": name, "source": "partner", "contributor": LAYER,
                        "aliases": [a for a in (clean(r.get(f)) for f in alias_f) if a], "n_parcels": 0,
                        "blocks": [], "lots": [], "pairs": []})
    return out


def mngeo_records(mngeo_dir, counties):
    import pandas as pd
    out = []
    for c in counties:
        df = pd.read_parquet(Path(mngeo_dir) / f"parcels_{c.lower()}.parquet", columns=["plat_name", "block", "lot"])
        df["plat_name"] = df["plat_name"].map(clean)
        df = df[df["plat_name"].notna()]
        df["b"] = df["block"].map(lambda v: parse_block(clean(v)) if clean(v) else None)
        df["l"] = df["lot"].map(lambda v: sorted(parse_lots(clean(v))) if clean(v) else [])
        for name, g in df.groupby("plat_name", sort=True):
            blocks = sorted({b for b in g["b"] if isinstance(b, str)})
            lots = sorted({x for ls in g["l"] for x in ls})
            pairs = sorted({(b, x) for b, ls in zip(g["b"], g["l"]) if isinstance(b, str) for x in ls})
            out.append({"type": "name", "county": county_key(c), "name": name, "source": "mngeo",
                        "contributor": LAYER, "aliases": [], "n_parcels": len(g), "blocks": blocks, "lots": lots,
                        "pairs": pairs})
    return out


def cov_records(rows, source):
    """rows with county, doc_num, add_cov, add_mod, block_mod, lot_mod (public pairs or train split rows)."""
    out = []
    for r in rows:
        mod = clean(r.get("add_mod"))
        if not mod:
            continue
        c = county_key(r.get("county") or r.get("cnty_name"))
        doc = norm_doc(r.get("doc_num"))
        out.append({"type": "cov", "county": c, "add_cov": clean(r.get("add_cov")), "add_mod": mod, "source": source,
                    "contributor": (c, doc) if doc else ("row", f"{source}:{len(out)}"),
                    "block": parse_block(clean(r.get("block_mod"))), "lots": sorted(parse_lots(clean(r.get("lot_mod"))))})
    return out


def build_gazetteer(records, normalizer, sources=None, counties=None):
    g = Gazetteer(normalizer)
    for r in records:
        if (sources is not None and r["source"] not in sources) or (counties is not None and r["county"] not in counties):
            continue
        if r["type"] == "name":
            g.add_name(r["county"], r["name"], r["source"], aliases=r["aliases"], contributor=tuple(r["contributor"]),
                       n_parcels=r["n_parcels"], blocks=r["blocks"], lots=r["lots"],
                       pairs=[tuple(p) for p in r["pairs"]])
        else:
            g.add_cov_alias(r["county"], r["add_cov"], r["add_mod"], r["source"], tuple(r["contributor"]),
                            block=r["block"], lots=r["lots"])
    return g


def save_records(records, path):
    import pandas as pd
    df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()}
                       for r in records])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_records(path):
    import pandas as pd
    out = []
    for r in pd.read_parquet(path).to_dict("records"):
        r = {k: v for k, v in r.items() if not (v is None or (isinstance(v, float) and v != v))}
        for k in ("aliases", "blocks", "lots", "pairs", "contributor"):
            if k in r and isinstance(r[k], str):
                r[k] = json.loads(r[k])
        if r["type"] == "name":
            r.setdefault("aliases", [])
        r.setdefault("block", None)
        r.setdefault("add_cov", None)
        out.append(r)
    return out


def source_counts(records):
    out = defaultdict(lambda: defaultdict(int))
    for r in records:
        out[r["county"]][f"records_{r['source']}"] += 1
    return {k: dict(v) for k, v in out.items()}


def mclio_records(mclio_dir):
    """MCLIO plats (dev/subdiv_07_mclio.py) -> name records for Milwaukee, source "mclio"."""
    import pandas as pd
    f = Path(mclio_dir) / "plats.parquet"
    if not f.exists():
        return []
    df = pd.read_parquet(f, columns=["name", "n_parcels", "blocks", "lots", "pairs"])
    return [{"type": "name", "county": "milwaukee", "name": r["name"], "source": "mclio", "contributor": LAYER,
             "aliases": [], "n_parcels": int(r["n_parcels"]), "blocks": json.loads(r["blocks"]),
             "lots": json.loads(r["lots"]), "pairs": [tuple(p) for p in json.loads(r["pairs"])]}
            for r in df.to_dict("records")]


def plat_polygons(cfg, root, layers, normalizer):
    """(county, key) -> (polygon in EPSG:26915, origin): partner layer polygons when the key has one, else the
    union of MnGeo / MCLIO plats dissolved by name. Prints nothing."""
    import geopandas as gpd
    from shapely.ops import unary_union
    P = cfg["paths"]
    parts = {"partner": defaultdict(list), "parcels": defaultdict(list)}
    for county, (rel, name_f, _) in layers.items():
        g = gpd.read_file(Path(root) / rel, columns=[name_f]).to_crs("EPSG:26915")
        g["geometry"] = g.geometry.make_valid()
        for name, geom in zip(g[name_f], g.geometry):
            key = normalizer(name) if isinstance(name, str) else ""
            if key and geom is not None and not geom.is_empty:
                parts["partner"][(county, key)].append(geom)
    mng = gpd.read_parquet(Path(P["mngeo_dir"]) / "plats.parquet")
    for co, name, geom in zip(mng["co_name"], mng["plat_name"], mng.geometry):
        key = normalizer(name)
        if key and geom is not None:
            parts["parcels"][(county_key(co), key)].append(geom.buffer(0) if not geom.is_valid else geom)
    mcl = Path(P["mclio_dir"]) / "plats.parquet"
    if mcl.exists():
        m = gpd.read_parquet(mcl)
        for name, geom in zip(m["name"], m.geometry):
            key = normalizer(name)
            if key and geom is not None:
                parts["parcels"][("milwaukee", key)].append(geom.buffer(0) if not geom.is_valid else geom)
    out = {}
    for origin in ("parcels", "partner"):   # partner last: it wins
        for k, gs in parts[origin].items():
            out[k] = (unary_union(gs), origin)
    return out
