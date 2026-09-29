"""
Aggregate-only evaluation of Tool 3's page grouping against the GeoJSON deeds.

Answers: should Tool 3 group pages by (county, folder, prefix) instead of (county, prefix)?
A GeoJSON feature's image_ids list is taken as the ground truth for "pages of one deed".
Both groupings are scored against it:
  * over-merged group: no single GeoJSON deed contains all of the group's GeoJSON ids
    (separate deeds were merged).
  * split deed: a GeoJSON deed whose OCR pages fall into more than one group.
Folder-level grouping is better if it lowers over-merged groups without adding split deeds.
(Run on 2026-09-29: it did not, so Tool 3 keeps (county, prefix); see PROJECT_MAP.md §7
item 12. The over-merged and split-deed counts are a baseline for future grouping RQs.)

Privacy: it reads OCR file NAMES only (never opens OCR files) and only the image_ids
property of GeoJSON features, and prints COUNTS only: no paths, ids, names or text.
Standard library only, so any Python 3 works. Roots are taken from, in order:
--ocr-root / --geojson-root, then $OCRTXT_PATH / $GEOJSON_PATH, then the repository .env.

    python3 dev/check_page_grouping.py
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

DEV_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEV_DIR.parent


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, DEV_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_collisions = _load_sibling("check_county_id_collisions")  # stdlib-only helpers
read_env_file = _collisions.read_env_file
load_group_files_by_prefix = _collisions.load_group_files_by_prefix


def resolve(cli_value, var, environ=os.environ, env_file=REPO_ROOT / ".env"):
    return cli_value or environ.get(var) or read_env_file(env_file).get(var)


def parse_image_ids(value):
    """Parse a GeoJSON image_ids value like geojson_loader does; None if unparseable."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value.replace("'", '"'))
        except ValueError:
            return None
    if not isinstance(value, list):
        return None
    return [i for i in value if isinstance(i, str) and i.strip()]


def load_geojson_deeds(geojson_root, counties):
    """Return ({county: [set of ids per feature]}, number of unparseable image_ids values)."""
    deeds, unparseable = defaultdict(list), 0
    for county in counties:
        county_dir = os.path.join(geojson_root, county)
        for name in sorted(os.listdir(county_dir)):
            if not name.endswith(".geojson"):
                continue
            with open(os.path.join(county_dir, name), encoding="utf-8") as f:
                features = json.load(f).get("features", [])
            for feature in features:
                ids = parse_image_ids((feature.get("properties") or {}).get("image_ids"))
                if ids is None:
                    unparseable += 1
                elif ids:
                    deeds[county].append(set(ids))
    return deeds, unparseable


def tool3_groups(ocr_root, by_folder=False):
    """Tool 3's page groups as (county, [county-relative ids]).

    Mirrors Tool 3: group_files_by_prefix, then split by county. by_folder=True also splits
    by folder, i.e. the proposed (county, folder, prefix) key.
    """
    groups = []
    for group in load_group_files_by_prefix()(ocr_root):
        buckets = {}
        for path in group:
            parts = Path(os.path.relpath(path, ocr_root)).parts
            county, image_id = (parts[0], "/".join(parts[1:])) if len(parts) > 1 else (None, parts[0])
            key = (county, os.path.dirname(path)) if by_folder else (county,)
            buckets.setdefault(key, (county, []))[1].append(image_id)
        groups.extend(buckets.values())
    return groups


def evaluate(groups, deeds):
    """Score groups against GeoJSON deeds; returns counts only."""
    deed_list = [(county, ids) for county, per_county in deeds.items() for ids in per_county]
    id_to_deeds = defaultdict(set)
    for d, (county, ids) in enumerate(deed_list):
        for i in ids:
            id_to_deeds[(county, i)].add(d)
    id_to_group = {(county, i): g for g, (county, ids) in enumerate(groups) for i in ids}

    c = dict.fromkeys(["groups", "multi_folder_groups", "labeled_groups", "over_merged_groups",
                       "multi_folder_consistent", "multi_folder_over_merged", "multi_folder_unlabeled",
                       "deeds_with_pages", "split_deeds", "exact_deeds"], 0)
    for county, ids in groups:
        c["groups"] += 1
        multi = len({os.path.dirname(i) for i in ids}) > 1
        c["multi_folder_groups"] += multi
        known = [id_to_deeds[(county, i)] for i in ids if (county, i) in id_to_deeds]
        if not known:
            c["multi_folder_unlabeled"] += multi
            continue
        c["labeled_groups"] += 1
        over_merged = not set.intersection(*known)
        c["over_merged_groups"] += over_merged
        c["multi_folder_over_merged"] += multi and over_merged
        c["multi_folder_consistent"] += multi and not over_merged
    for county, ids in deed_list:
        on_disk = {i for i in ids if (county, i) in id_to_group}
        if not on_disk:
            continue
        c["deeds_with_pages"] += 1
        group_ids = {id_to_group[(county, i)] for i in on_disk}
        c["split_deeds"] += len(group_ids) > 1
        c["exact_deeds"] += len(group_ids) == 1 and set(groups[group_ids.pop()][1]) == on_disk
    return c


def summary(ocr_root, geojson_root):
    ocr_counties = {d for d in os.listdir(ocr_root) if os.path.isdir(os.path.join(ocr_root, d))}
    geo_counties = {d for d in os.listdir(geojson_root) if os.path.isdir(os.path.join(geojson_root, d))}
    counties = sorted(ocr_counties & geo_counties)
    deeds, unparseable = load_geojson_deeds(geojson_root, counties)
    current = tool3_groups(ocr_root)
    by_folder = tool3_groups(ocr_root, by_folder=True)
    on_disk = {(county, i) for county, ids in current for i in ids}
    geo_ids = {(county, i) for county, per_county in deeds.items() for ids in per_county for i in ids}
    return {
        "counties": len(counties),
        "ocr_only_counties": len(ocr_counties - geo_counties),
        "geojson_only_folders": len(geo_counties - ocr_counties),
        "ocr_files": len(on_disk),
        "geojson_deeds": sum(len(v) for v in deeds.values()),
        "unparseable": unparseable,
        "geojson_ids": len(geo_ids),
        "geojson_ids_missing": len(geo_ids - on_disk),
        "multi_folder_deeds": sum(len({os.path.dirname(i) for i in ids if (county, i) in on_disk}) > 1
                                  for county, per_county in deeds.items() for ids in per_county),
        "current": evaluate(current, deeds),
        "by_folder": evaluate(by_folder, deeds),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ocr-root", default=None, help="OCR txt root that contains the county folders")
    parser.add_argument("--geojson-root", default=None, help="GeoJSON root that contains the county folders")
    args = parser.parse_args(argv)
    ocr_root = resolve(args.ocr_root, "OCRTXT_PATH")
    geojson_root = resolve(args.geojson_root, "GEOJSON_PATH")
    if not ocr_root or not os.path.isdir(ocr_root) or not geojson_root or not os.path.isdir(geojson_root):
        sys.exit("OCR or GeoJSON root not found (pass --ocr-root/--geojson-root, or set "
                 "OCRTXT_PATH/GEOJSON_PATH in the environment or .env)")

    s = summary(ocr_root, geojson_root)
    cur, fol = s["current"], s["by_folder"]
    print(f"counties compared: {s['counties']} | county folders only under OCR root: {s['ocr_only_counties']}"
          f" | folders only under GeoJSON root: {s['geojson_only_folders']}")
    print(f"OCR txt files: {s['ocr_files']} | GeoJSON deeds (features with image_ids): {s['geojson_deeds']}"
          f" | unparseable image_ids values: {s['unparseable']}")
    print(f"GeoJSON ids: {s['geojson_ids']} | not found as OCR files: {s['geojson_ids_missing']}"
          f" | GeoJSON deeds whose OCR pages span >1 folder: {s['multi_folder_deeds']}")
    print(f"{'':62}{'current':>10}{'by folder':>11}")
    print(f"{'':62}{'(county,':>10}{'(county,':>11}")
    print(f"{'':62}{'prefix)':>10}{'folder,':>11}")
    print(f"{'':62}{'':>10}{'prefix)':>11}")
    rows = [("Tool 3 page groups", "groups"),
            ("  groups spanning >1 folder", "multi_folder_groups"),
            ("  groups with GeoJSON ids", "labeled_groups"),
            ("  over-merged groups (no single GeoJSON deed contains them)", "over_merged_groups"),
            ("GeoJSON deeds with OCR pages", "deeds_with_pages"),
            ("  split across >1 group", "split_deeds"),
            ("  matched exactly by one group", "exact_deeds")]
    for label, key in rows:
        print(f"{label:62}{cur[key]:>10}{fol[key]:>11}")
    print(f"current multi-folder groups -> consistent with one GeoJSON deed: {cur['multi_folder_consistent']}"
          f" | over-merged: {cur['multi_folder_over_merged']} | no GeoJSON ids: {cur['multi_folder_unlabeled']}")


if __name__ == "__main__":
    main()
