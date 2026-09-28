"""
Aggregate-only check for cross-county image_id collisions in the local OCR txt tree.

Answers: would Tool 3 -> 7 -> 9 merge records from different counties on this data?
It only walks file NAMES under the OCR root (never opens those files) and prints COUNTS
only: no paths, ids, names or text.

Standard library only: no python-dotenv or other project dependencies are needed, so any
Python 3 works. The OCR root is taken from, in order:
  1. --root
  2. the OCRTXT_PATH environment variable
  3. OCRTXT_PATH in the repository's .env file

Run from anywhere:

    python3 dev/check_county_id_collisions.py
    python3 dev/check_county_id_collisions.py --root /path/to/ocr/txt/   # override
"""

import argparse
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUPING_MODULE = REPO_ROOT / "src" / "data_preprocessing" / "generate_image_ids_list_from_filenames.py"


def read_env_file(path):
    """Minimal .env reader: KEY=VALUE lines, optional 'export ', quotes, and # comments."""
    values = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, value = line.split("=", 1)
            value = value.strip()
            if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()
            values[key.strip()] = value
    return values


def resolve_root(cli_root, environ=os.environ, env_file=REPO_ROOT / ".env"):
    """Pick the OCR root: --root, then $OCRTXT_PATH, then OCRTXT_PATH in .env."""
    return cli_root or environ.get("OCRTXT_PATH") or read_env_file(env_file).get("OCRTXT_PATH")


def load_group_files_by_prefix():
    """Load Tool 3's grouping function straight from its file.

    Importing it via the `src` package would run src/__init__.py, which needs
    python-dotenv, geopandas and other dependencies. The module itself only imports os.
    """
    spec = importlib.util.spec_from_file_location("generate_image_ids_list_from_filenames", GROUPING_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.group_files_by_prefix


def count_collisions(root):
    """Return aggregate counts only (no paths or ids) for the OCR tree under root."""
    groups = load_group_files_by_prefix()(root)

    def county_and_id(path):
        parts = Path(os.path.relpath(path, root)).parts
        return (parts[0], "/".join(parts[1:])) if len(parts) > 1 else (None, parts[0])

    counties_seen = set()
    mixed_county_groups = multi_folder_groups = 0
    id_counties = defaultdict(set)
    for group in groups:
        counties = {county_and_id(p)[0] for p in group}
        counties_seen |= counties
        if len(counties) > 1:
            mixed_county_groups += 1
        elif len({os.path.dirname(p) for p in group}) > 1:
            multi_folder_groups += 1
        for p in group:
            county, image_id = county_and_id(p)
            id_counties[image_id].add(county)

    colliding_ids = [i for i, c in id_counties.items() if len(c) > 1]
    return {
        "files": sum(len(g) for g in groups),
        "county_folders": len(counties_seen - {None}),
        "files_under_root": sum(1 for c in id_counties.values() if None in c),
        "groups": len(groups),
        "mixed_county_groups": mixed_county_groups,
        "multi_folder_groups": multi_folder_groups,
        "colliding_ids": len(colliding_ids),
        "colliding_ids_no_underscore": sum("_" not in os.path.basename(i) for i in colliding_ids),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--root", default=None,
                        help="OCR txt root that contains the county folders "
                             "(default: $OCRTXT_PATH, then OCRTXT_PATH in the repository .env)")
    args = parser.parse_args(argv)

    root = resolve_root(args.root)
    if not root or not os.path.isdir(root):
        sys.exit("OCR root not found (pass --root, or set OCRTXT_PATH in the environment or .env)")

    c = count_collisions(root)
    print(f"OCR txt files: {c['files']} | county folders: {c['county_folders']}"
          f" | files directly under root: {c['files_under_root']}")
    print(f"Tool 3 page groups: {c['groups']}")
    print(f"  groups mixing >1 county (Tool 3 merges across counties): {c['mixed_county_groups']}")
    print(f"  single-county groups spanning >1 folder: {c['multi_folder_groups']}")
    print(f"county-relative ids present in >1 county: {c['colliding_ids']}")
    print(f"  of which basename has no '_' (not merged by Tool 3, but merged by Tool 7/9): "
          f"{c['colliding_ids_no_underscore']}")


if __name__ == "__main__":
    main()
