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

A "diagnosis" section then classifies the current grouping's mismatches (counts only):
  * GeoJSON hygiene: missing OCR pages, ids listed in more than one deed, exact
    duplicate deeds, ids repeated within one deed, OCR pages no deed references.
  * over-merged groups: merging disjoint deeds (no shared ids: a real merge of separate
    deeds) vs. overlapping deeds (the deeds share an id, so GeoJSON itself chains them).
  * split deeds, one primary reason each: some pages have no "_" (each such page is its
    own group), or the pages have different basename prefixes (split further into
    prefixes that differ only in digits vs. in other text).

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
import re
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
    """Return ({county: [set of ids per feature]}, unparseable image_ids values, ids repeated within a feature)."""
    deeds, unparseable, repeated = defaultdict(list), 0, 0
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
                    repeated += len(ids) - len(set(ids))
                    deeds[county].append(set(ids))
    return deeds, unparseable, repeated


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


def _basename_prefix(image_id):
    """The key group_files_by_prefix uses: basename text before the last "_" (None if no "_")."""
    base = image_id.rsplit("/", 1)[-1]
    return base.rsplit("_", 1)[0] if "_" in base else None


def diagnose(groups, deeds, repeated_within_deed=0):
    """Classify the mismatches between groups and GeoJSON deeds; returns counts only."""
    deed_list = [(county, ids) for county, per_county in deeds.items() for ids in per_county]
    id_to_deeds = defaultdict(set)
    for d, (county, ids) in enumerate(deed_list):
        for i in ids:
            id_to_deeds[(county, i)].add(d)
    id_to_group = {(county, i): g for g, (county, ids) in enumerate(groups) for i in ids}

    h = dict.fromkeys(["deeds", "deeds_with_pages", "deeds_some_pages_missing", "deeds_all_pages_missing",
                       "ids_in_multiple_deeds", "deeds_sharing_ids", "exact_duplicate_deeds",
                       "ids_repeated_within_deed", "unreferenced_ocr_pages"], 0)
    h["deeds"] = len(deed_list)
    h["ids_repeated_within_deed"] = repeated_within_deed
    shared = [k for k, ds in id_to_deeds.items() if len(ds) > 1]
    h["ids_in_multiple_deeds"] = len(shared)
    h["deeds_sharing_ids"] = len(set().union(*(id_to_deeds[k] for k in shared)))
    seen = set()
    for county, ids in deed_list:
        key = (county, frozenset(ids))
        h["exact_duplicate_deeds"] += key in seen
        seen.add(key)
        present = sum((county, i) in id_to_group for i in ids)
        h["deeds_with_pages"] += present > 0
        h["deeds_some_pages_missing"] += 0 < present < len(ids)
        h["deeds_all_pages_missing"] += present == 0
    h["unreferenced_ocr_pages"] = sum(k not in id_to_deeds for k in id_to_group)

    g = dict.fromkeys(["groups", "labeled_groups", "groups_touching_multiple_deeds",
                       "groups_with_unreferenced_and_geojson_pages", "groups_with_only_unreferenced_pages"], 0)
    o = dict.fromkeys(["total", "disjoint_deeds", "overlapping_deeds", "touching_2_deeds",
                       "touching_3plus_deeds", "multi_folder", "with_unreferenced_pages"], 0)
    over_merged = set()
    for gi, (county, ids) in enumerate(groups):
        g["groups"] += 1
        known = [(county, i) for i in ids if (county, i) in id_to_deeds]
        has_unreferenced = len(known) < len(ids)
        if not known:
            g["groups_with_only_unreferenced_pages"] += 1
            continue
        g["labeled_groups"] += 1
        g["groups_with_unreferenced_and_geojson_pages"] += has_unreferenced
        touched = set().union(*(id_to_deeds[k] for k in known))
        g["groups_touching_multiple_deeds"] += len(touched) > 1
        if set.intersection(*(id_to_deeds[k] for k in known)):
            continue
        over_merged.add(gi)
        o["total"] += 1
        overlapping = any(len(id_to_deeds[k]) > 1 for k in known)
        o["overlapping_deeds"] += overlapping
        o["disjoint_deeds"] += not overlapping
        o["touching_2_deeds"] += len(touched) == 2
        o["touching_3plus_deeds"] += len(touched) > 2
        o["multi_folder"] += len({os.path.dirname(i) for i in ids}) > 1
        o["with_unreferenced_pages"] += has_unreferenced

    sp = dict.fromkeys(["total", "no_underscore_pages", "different_prefixes", "prefixes_differ_only_in_digits",
                        "prefixes_differ_in_text", "same_prefix", "multi_folder", "into_2_groups",
                        "into_3plus_groups", "in_over_merged_group", "some_pages_missing"], 0)
    for county, ids in deed_list:
        present = [i for i in ids if (county, i) in id_to_group]
        pieces = {id_to_group[(county, i)] for i in present}
        if len(pieces) < 2:
            continue
        sp["total"] += 1
        prefixes = {_basename_prefix(i) for i in present}
        if None in prefixes:
            sp["no_underscore_pages"] += 1
        elif len(prefixes) > 1:
            sp["different_prefixes"] += 1
            digits_only = len({re.sub(r"[0-9]+", "#", p) for p in prefixes}) == 1
            sp["prefixes_differ_only_in_digits"] += digits_only
            sp["prefixes_differ_in_text"] += not digits_only
        else:
            sp["same_prefix"] += 1
        sp["multi_folder"] += len({os.path.dirname(i) for i in present}) > 1
        sp["into_2_groups"] += len(pieces) == 2
        sp["into_3plus_groups"] += len(pieces) > 2
        sp["in_over_merged_group"] += bool(pieces & over_merged)
        sp["some_pages_missing"] += len(present) < len(ids)
    return {"hygiene": h, "groups": g, "over_merged": o, "split": sp}


def summary(ocr_root, geojson_root):
    ocr_counties = {d for d in os.listdir(ocr_root) if os.path.isdir(os.path.join(ocr_root, d))}
    geo_counties = {d for d in os.listdir(geojson_root) if os.path.isdir(os.path.join(geojson_root, d))}
    counties = sorted(ocr_counties & geo_counties)
    deeds, unparseable, repeated = load_geojson_deeds(geojson_root, counties)
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
        "diagnosis": diagnose(current, deeds, repeated),
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

    h, g, o, sp = (s["diagnosis"][k] for k in ("hygiene", "groups", "over_merged", "split"))
    print()
    print("diagnosis of the current (county, prefix) grouping (counts only)")
    print(f"GeoJSON deeds: {h['deeds']} | with >=1 OCR page: {h['deeds_with_pages']}"
          f" | some pages missing: {h['deeds_some_pages_missing']} | all pages missing: {h['deeds_all_pages_missing']}")
    print(f"ids listed in >1 deed: {h['ids_in_multiple_deeds']} | deeds sharing an id: {h['deeds_sharing_ids']}"
          f" | exact duplicate deeds: {h['exact_duplicate_deeds']} | ids repeated within one deed: {h['ids_repeated_within_deed']}")
    print(f"OCR pages not referenced by any GeoJSON deed: {h['unreferenced_ocr_pages']}")
    print(f"groups: {g['groups']} | with GeoJSON ids: {g['labeled_groups']}"
          f" | containing pages from >1 GeoJSON deed: {g['groups_touching_multiple_deeds']}")
    print(f"  groups with unreferenced OCR pages: alongside GeoJSON pages: {g['groups_with_unreferenced_and_geojson_pages']}"
          f" | only unreferenced pages: {g['groups_with_only_unreferenced_pages']}")
    print(f"over-merged groups: {o['total']}")
    print(f"  merging disjoint deeds (no shared ids): {o['disjoint_deeds']}"
          f" | merging overlapping deeds (deeds share an id): {o['overlapping_deeds']}")
    print(f"  touching 2 deeds: {o['touching_2_deeds']} | 3+ deeds: {o['touching_3plus_deeds']}"
          f" | spanning >1 folder: {o['multi_folder']} | also containing unreferenced pages: {o['with_unreferenced_pages']}")
    print(f"GeoJSON deeds split across >1 group: {sp['total']}")
    print(f"  primary reason -> some pages have no '_': {sp['no_underscore_pages']}"
          f" | different basename prefixes: {sp['different_prefixes']} | same prefix: {sp['same_prefix']}")
    print(f"  different prefixes -> differ only in digits: {sp['prefixes_differ_only_in_digits']}"
          f" | differ in other text: {sp['prefixes_differ_in_text']}")
    print(f"  split into 2 groups: {sp['into_2_groups']} | 3+ groups: {sp['into_3plus_groups']}"
          f" | pages in >1 folder: {sp['multi_folder']} | some pages missing: {sp['some_pages_missing']}"
          f" | also part of an over-merged group: {sp['in_over_merged_group']}")


if __name__ == "__main__":
    main()
