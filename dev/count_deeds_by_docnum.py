"""
Aggregate-only deed counts under two deed definitions (read-only; descriptive).

  image_ids   a deed = a unique (county, set of image_ids) of a GeoJSON feature (dev/check_split_overlap.py);
              also reported without the county, as the 2025 count did ("unique image_ids" values)
  docnum      a deed = a unique document number taken from the page FILE NAMES: the basename text before the last
              "_" (the key src/data_preprocessing/generate_image_ids_list_from_filenames.py and Tool 3 group pages
              by); a page name without "_" is its own document. Counted per county and without the county.

Per source (each split of the July 2025 6:2:2 split, their union, and the full GeoJSON under GEOJSON_PATH) it
reports how the two definitions disagree:
  split   image_ids deeds whose pages carry more than one document number (docnum splits them)
  merged  document numbers shared by more than one image_ids deed (docnum merges them), and the deeds involved
  components  connected groups of image_ids deeds linked by a shared document number

Privacy: reads only the image_ids and cnty_name properties, never opens OCR files, and prints COUNTS only (no ids,
file names, paths or text). County names are mapped to the six study counties or "other". Errors report the
exception type and a numeric index only. Standard library only.

    python3 dev/count_deeds_by_docnum.py [--out results/split_audit/deed_count_definitions.json]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import check_split_overlap as cso  # noqa: E402  (loaders only)


def docnum(image_id):
    base = image_id.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit("_", 1)[0] if "_" in base else base


def load_full_geojson(root, errors):
    """All *.geojson files in the county folders under root -> [(county, frozenset ids)] + parse stats."""
    records, stats = [], Counter()
    folders = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    stats["folders"] = len(folders)
    for k, folder in enumerate(folders):
        recs, st = cso.load_split(Path(root) / folder, errors)
        stats.update(st)
        county = cso.county_key(folder)
        # trust the feature's cnty_name; fall back to the folder's county when the feature has none
        records += [(c if c not in ("unknown", "other") else county, ids) for c, ids in recs]
    return records, stats


def raw_unique_values(root, errors):
    """Distinct non-empty image_ids property values compared as raw strings (no parsing, order-sensitive), as a
    'unique image_ids' count over the GeoJSON files would see them. -> (distinct values, distinct parsed sets)."""
    raw, sets = set(), set()
    for k, folder in enumerate(sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))):
        for j, f in enumerate(cso.geojson_files(Path(root) / folder)):
            try:
                with open(f, encoding="utf-8") as fh:
                    feats = json.load(fh).get("features", [])
                for feat in feats:
                    v = (feat.get("properties") or {}).get("image_ids")
                    ids = cso.parse_image_ids(v)
                    if ids:
                        raw.add(v if isinstance(v, str) else json.dumps(v))
                        sets.add(frozenset(ids))
            except Exception as e:  # noqa: BLE001
                errors.append(f"raw#{k}/{j}:{type(e).__name__}")
    return len(raw), len(sets)


def count(records):
    deeds = set(records)                                   # (county, frozenset ids)
    pages = {(c, i) for c, ids in deeds for i in ids}
    by_doc = defaultdict(set)                              # (county, docnum) -> image_ids deeds
    for d in deeds:
        for i in d[1]:
            by_doc[(d[0], docnum(i))].add(d)
    docs_per_deed = {d: {docnum(i) for i in d[1]} for d in deeds}
    merged = {k: v for k, v in by_doc.items() if len(v) > 1}
    # connected components of deeds linked by a shared (county, docnum)
    parent = {d: d for d in deeds}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for v in by_doc.values():
        v = list(v)
        for d in v[1:]:
            parent[find(d)] = find(v[0])
    comps = Counter(find(d) for d in deeds)
    county_counts = Counter(c for c, _ in deeds)
    return {
        "features": len(records),
        "pages": len(pages),
        "pages_without_underscore": sum("_" not in i.replace("\\", "/").rsplit("/", 1)[-1] for _, i in pages),
        "deeds_image_ids_county": len(deeds),
        "deeds_image_ids_no_county": len({ids for _, ids in deeds}),
        "deeds_docnum_county": len(by_doc),
        "deeds_docnum_no_county": len({k[1] for k in by_doc}),
        "components_linked_by_docnum": len(comps),
        "split_image_ids_deeds_with_>1_docnum": sum(len(v) > 1 for v in docs_per_deed.values()),
        "split_extra_docnums_from_those_deeds": sum(len(v) - 1 for v in docs_per_deed.values()),
        "merged_docnums_shared_by_>1_deed": len(merged),
        "merged_image_ids_deeds_involved": len(set().union(*merged.values())) if merged else 0,
        "components_with_>1_deed": sum(n > 1 for n in comps.values()),
        "by_county_image_ids_deeds": dict(sorted(county_counts.items())),
        "by_county_docnum_deeds": dict(sorted(Counter(k[0] for k in by_doc).items())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "split_audit" / "deed_count_definitions.json"))
    ap.add_argument("--no-full", action="store_true", help="skip the full GeoJSON under GEOJSON_PATH")
    args = ap.parse_args(argv)
    errors, result, union = [], {"status": "descriptive recount (read-only, aggregates only)"}, []
    for name, path in cso.DEFAULT_SPLITS.items():
        recs, st = cso.load_split(path, errors)
        union += recs
        result[f"split_{name}"] = {**count(recs), "parse": dict(st)}
    result["splits_union"] = count(union)
    if not args.no_full:
        import check_county_id_collisions as cc
        root = os.environ.get("GEOJSON_PATH") or cc.read_env_file(REPO_ROOT / ".env").get("GEOJSON_PATH")
        if root and os.path.isdir(root):
            recs, st = load_full_geojson(root, errors)
            n_raw, n_sets = raw_unique_values(root, errors)
            result["full_geojson"] = {**count(recs), "parse": dict(st),
                                      "distinct_raw_image_ids_strings": n_raw, "distinct_image_ids_sets": n_sets}
        else:
            result["full_geojson"] = "GEOJSON_PATH not found"
    result["errors"] = {"count": len(errors), "types": dict(Counter(e.rsplit(":", 1)[-1] for e in errors))}
    text = json.dumps(result, indent=1)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
