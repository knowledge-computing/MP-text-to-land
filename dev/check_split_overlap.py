"""
Aggregate-only check of train/val/test GeoJSON splits at deed and page level.

Answers two questions about a split such as the July 2025 *6_2_2.geojson files:
  * deed-level sizes: a split file holds one feature per parcel geometry, so its row
    ratio (e.g. 60/20/20) can differ a lot from its deed ratio. A "deed" is a unique
    (county, set of image_ids); a "page" is a unique (county, image_id).
  * leakage: does a deed in one split share pages with a deed in another split, either
    as the identical deed or as an overlapping one (some pages shared)?

Privacy: reads only the image_ids and cnty_name properties of each feature, and prints
COUNTS only (no ids, paths, names or text). County names are mapped to the six study
counties or "other". Standard library only. Errors inside data loops report only the
exception type and a numeric index.

    python3 dev/check_split_overlap.py                      # July 2025 6_2_2 split
    python3 dev/check_split_overlap.py --split train=PATH --split test=PATH --out results/x.json

Each PATH is a .geojson file or a folder whose *.geojson files are read (not recursive).
"""

import argparse
import ast
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import private_paths as pp  # noqa: E402

JULY = pp.run_dir()  # MP_RUN_DIR, resolved only when the default splits are used
DEFAULT_SPLITS = {
    "train": JULY / "train" / "train6_2_2.geojson",
    "val": JULY / "val" / "val6_2_2.geojson",
    "test": JULY / "test" / "test6_2_2.geojson",
}
STUDY_COUNTIES = ("anoka", "dakota", "olmsted", "sherburne", "washington", "milwaukee")


def parse_image_ids(value):
    """Return a list of ids from a list, a JSON list string, or a Python tuple/list string; None if unparseable."""
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):  # numpy arrays from pyogrio list columns
        value = value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(text.replace("'", '"'))
            except ValueError:
                return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    return [i.strip() for i in value if isinstance(i, str) and i.strip()]


def county_key(value):
    if not isinstance(value, str):
        return "unknown"
    low = value.lower()
    return next((c for c in STUDY_COUNTIES if c in low), "other")


def geojson_files(path):
    path = Path(path)
    return sorted(path.glob("*.geojson")) if path.is_dir() else [path]


def load_split(path, errors):
    """Return per-feature records [(county, frozenset ids)] plus parse counters."""
    records, stats = [], Counter()
    for file_index, file in enumerate(geojson_files(path)):
        try:
            with open(file, encoding="utf-8") as f:
                features = json.load(f).get("features", [])
        except Exception as e:  # report type and index only
            errors.append(f"file#{file_index}:{type(e).__name__}")
            continue
        for feat_index, feat in enumerate(features):
            try:
                props = feat.get("properties") or {}
                ids = parse_image_ids(props.get("image_ids"))
                stats["features"] += 1
                if ids is None:
                    stats["features_unparseable_ids"] += 1
                    continue
                if not ids:
                    stats["features_empty_ids"] += 1
                    continue
                if len(set(ids)) < len(ids):
                    stats["features_with_repeated_id"] += 1
                records.append((county_key(props.get("cnty_name")), frozenset(ids)))
            except Exception as e:
                errors.append(f"file#{file_index}/feature#{feat_index}:{type(e).__name__}")
    return records, stats


def summarize(splits):
    errors, deeds, pages, out = [], {}, {}, {"splits": {}, "pairs": {}}
    for name, path in splits.items():
        records, stats = load_split(path, errors)
        per_deed = Counter(records)
        deeds[name] = set(per_deed)
        pages[name] = {(c, i) for c, ids in per_deed for i in ids}
        by_county = defaultdict(lambda: Counter())
        for (c, _), n in per_deed.items():
            by_county[c]["deeds"] += 1
            by_county[c]["features"] += n
        out["splits"][name] = {
            **stats,
            "deeds": len(per_deed),
            "pages": len(pages[name]),
            "deeds_with_multiple_features": sum(1 for n in per_deed.values() if n > 1),
            "max_features_per_deed": max(per_deed.values(), default=0),
            "by_county": {c: dict(v) for c, v in sorted(by_county.items())},
        }
    total = set().union(*deeds.values()) if deeds else set()
    out["all_splits_unique_deeds"] = len(total)
    out["deed_share_of_union"] = {n: round(len(d) / len(total), 4) if total else None for n, d in deeds.items()}
    feats = sum(s["features"] for s in out["splits"].values())
    out["feature_share"] = {n: round(s["features"] / feats, 4) if feats else None for n, s in out["splits"].items()}

    page_owner = {name: defaultdict(set) for name in deeds}
    for name in deeds:
        for deed in deeds[name]:
            for i in deed[1]:
                page_owner[name][(deed[0], i)].add(deed)
    for a, b in combinations(deeds, 2):
        identical = deeds[a] & deeds[b]
        a_touch = {d for d in deeds[a] if any((d[0], i) in pages[b] for i in d[1])}
        b_touch = {d for d in deeds[b] if any((d[0], i) in pages[a] for i in d[1])}
        out["pairs"][f"{a}~{b}"] = {
            "identical_deeds": len(identical),
            f"{a}_deeds_sharing_a_page_with_{b}": len(a_touch),
            f"{b}_deeds_sharing_a_page_with_{a}": len(b_touch),
            f"{a}_deeds_overlapping_not_identical": len(a_touch - identical),
            "shared_pages": len(pages[a] & pages[b]),
        }
    if "test" in deeds:
        others = [n for n in deeds if n != "test"]
        other_pages = set().union(*(pages[n] for n in others)) if others else set()
        other_deeds = set().union(*(deeds[n] for n in others)) if others else set()
        leaked = {d for d in deeds["test"] if any((d[0], i) in other_pages for i in d[1])}
        out["test_leakage"] = {
            "test_deeds": len(deeds["test"]),
            "test_deeds_identical_to_a_non_test_deed": len(deeds["test"] & other_deeds),
            "test_deeds_sharing_any_page_with_non_test": len(leaked),
            "test_pages_seen_outside_test": len(pages["test"] & other_pages),
        }
    out["errors"] = Counter(e.split(":")[-1] for e in errors)
    out["error_count"] = len(errors)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--split", action="append", default=[], metavar="NAME=PATH",
                        help="split name and GeoJSON file/folder; repeatable (default: July 2025 6_2_2 files)")
    parser.add_argument("--out", default=None, help="optional path for the aggregate JSON")
    args = parser.parse_args(argv)
    splits = dict(s.split("=", 1) for s in args.split) if args.split else {k: str(v) for k, v in DEFAULT_SPLITS.items()}
    missing = [n for n, p in splits.items() if not os.path.exists(p)]
    if missing:
        print(f"missing split paths for: {', '.join(missing)}", file=sys.stderr)
        return 2
    result = summarize(splits)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
